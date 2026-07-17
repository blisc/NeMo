#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Find fixed Lhotse bucket batch sizes for EasyMagpie training.

This adapter deliberately profiles the real EasyMagpie training path instead of
using the generic OOMptimizer synthetic schema. It assumes cached target/context
codes, the NeMo AutoModel decoder backend, bf16 autocast, and no activation
checkpointing. The Jason model config does not provide a usable data location,
so every invocation must supply a Lhotse input-config YAML with ``--input-cfg``.
"""

import bisect
import gc
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import click
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.tts.models import EasyMagpieTTSModel


DEFAULT_CONFIG_PATH = Path("examples/tts/conf/magpietts/easy_magpietts_lhotse.yaml")


class BucketList(click.ParamType):
    """Parse a CLI value such as ``'[4.4, 8.0, 12.0]'``."""

    name = "list[float]"

    def convert(self, value, param, ctx):
        if value is None or isinstance(value, list):
            return value
        try:
            parsed = OmegaConf.to_container(OmegaConf.create(value))
            bins = [float(item) for item in parsed]
        except (TypeError, ValueError):
            self.fail(f"expected a list of numbers, got {value!r}", param, ctx)
        if not bins or bins != sorted(bins) or len(set(bins)) != len(bins):
            self.fail("bucket boundaries must be a non-empty, strictly increasing list", param, ctx)
        return bins


def bucket_index(duration: float, bucket_duration_bins: Sequence[float]) -> int | None:
    """Return the index of the first duration upper bound containing ``duration``."""

    index = bisect.bisect_left(bucket_duration_bins, duration)
    return index if index < len(bucket_duration_bins) else None


def expand_singleton_batch(value: Any, batch_size: int) -> Any:
    """Repeat the batch dimension of a nested singleton EasyMagpie batch."""

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        if value.shape[0] != 1:
            raise ValueError(f"Expected singleton tensor batch dimension, got shape={tuple(value.shape)}")
        return value.expand(batch_size, *value.shape[1:]).clone()
    if isinstance(value, dict):
        return {key: expand_singleton_batch(item, batch_size) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected singleton list batch dimension, got len={len(value)}")
        return value * batch_size
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError(f"Expected singleton tuple batch dimension, got len={len(value)}")
        return value * batch_size
    return value


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def _pad_codes(codes: torch.Tensor, num_frames: int) -> torch.Tensor:
    if codes.ndim != 3 or codes.shape[0] != 1:
        raise ValueError(f"Expected cached codes with shape (1, C, T), got {tuple(codes.shape)}")
    if codes.shape[-1] >= num_frames:
        return codes[..., :num_frames]
    return F.pad(codes, (0, num_frames - codes.shape[-1]), value=0)


def pad_cached_codes_for_bucket(
    batch: dict[str, Any],
    bucket_duration: float,
    context_duration: float,
    frames_per_second: float,
) -> dict[str, Any]:
    """Pad cached target/context codes to conservative bucket-boundary lengths."""

    required = {"audio_codes", "audio_codes_lens", "context_audio_codes", "context_audio_codes_lens"}
    missing = required.difference(batch)
    if missing:
        raise ValueError(
            "EasyMagpie OOMptimizer requires cached target and context codes; "
            f"the batch is missing {sorted(missing)}"
        )

    result = deepcopy(batch)
    target_frames = max(1, math.ceil(bucket_duration * frames_per_second))
    context_frames = max(1, math.ceil(context_duration * frames_per_second))
    result["audio_codes"] = _pad_codes(result["audio_codes"], target_frames)
    result["audio_codes_lens"] = torch.full_like(result["audio_codes_lens"], target_frames)
    result["context_audio_codes"] = _pad_codes(result["context_audio_codes"], context_frames)
    result["context_audio_codes_lens"] = torch.full_like(result["context_audio_codes_lens"], context_frames)
    return result


@dataclass
class BatchSizeSearch:
    """Exponential search followed by binary search over integer batch sizes."""

    current: int = 2
    relative_gap_threshold: float = 0.05
    max_ok: int | None = None
    min_oom: int | None = None

    def __post_init__(self):
        if self.current < 1:
            raise ValueError("The starting batch size must be positive.")
        if not 0.0 <= self.relative_gap_threshold <= 1.0:
            raise ValueError("The relative gap threshold must be in [0, 1].")

    @property
    def done(self) -> bool:
        if self.max_ok is None or self.min_oom is None:
            return False
        gap = (self.min_oom - self.max_ok) / self.min_oom
        return self.min_oom - self.max_ok <= 1 or gap <= self.relative_gap_threshold

    def record(self, oom: bool) -> None:
        if self.done:
            raise RuntimeError("Cannot advance a completed batch-size search.")
        if oom:
            self.min_oom = self.current if self.min_oom is None else min(self.min_oom, self.current)
            if self.current == 1 and self.max_ok is None:
                raise RuntimeError("A singleton batch does not fit on this GPU.")
            self.current = max(1, self.current // 2) if self.max_ok is None else (self.max_ok + self.min_oom) // 2
        else:
            self.max_ok = self.current if self.max_ok is None else max(self.max_ok, self.current)
            self.current = self.current * 2 if self.min_oom is None else math.ceil((self.max_ok + self.min_oom) / 2)


def _profiling_dataloader_config(train_ds_cfg: DictConfig, bins: Sequence[float]) -> DictConfig:
    result = deepcopy(train_ds_cfg)
    with open_dict(result.dataset):
        result.dataset.batch_duration = None
        result.dataset.quadratic_duration = None
        result.dataset.bucket_duration_bins = list(bins)
        result.dataset.bucket_batch_size = [1] * len(bins)
        result.dataset.use_bucketing = True
        result.dataset.drop_last = False
        result.dataset.num_workers = 0
        result.dataset.pin_memory = False
    return result


def collect_bucket_examples(
    dataloader,
    bucket_duration_bins: Sequence[float],
    frames_per_second: float,
    max_batches: int,
) -> list[dict[str, Any]]:
    """Keep the longest observed cached-code singleton from every duration bucket."""

    examples: list[dict[str, Any] | None] = [None] * len(bucket_duration_bins)
    observed_durations = [-1.0] * len(bucket_duration_bins)
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        if "audio_codes" not in batch or "audio_codes_lens" not in batch:
            raise ValueError("Encountered uncached target audio while scanning profiling examples.")
        if "context_audio_codes" not in batch or "context_audio_codes_lens" not in batch:
            raise ValueError("Encountered uncached context audio while scanning profiling examples.")
        if batch["audio_codes"].shape[0] != 1:
            raise ValueError("The profiling dataloader must yield singleton batches.")

        duration = float(batch["audio_codes_lens"][0]) / frames_per_second
        index = bucket_index(duration, bucket_duration_bins)
        if index is not None and duration > observed_durations[index]:
            examples[index] = move_to_device(batch, torch.device("cpu"))
            observed_durations[index] = duration
        if all(item is not None for item in examples):
            break

    missing = [bucket_duration_bins[index] for index, item in enumerate(examples) if item is None]
    if missing:
        raise RuntimeError(
            f"Did not observe examples for bucket upper bounds {missing} after scanning {max_batches} batches. "
            "Increase --scan-batches or verify that the configured data mixture contains those durations."
        )
    return examples  # type: ignore[return-value]


def _validate_assumptions(cfg: DictConfig) -> None:
    if cfg.model.get("decoder_type") != "nemo_automodel":
        raise ValueError("This adapter only supports model.decoder_type=nemo_automodel.")
    if not cfg.model.get("load_cached_codes_if_available", False):
        raise ValueError("This adapter requires model.load_cached_codes_if_available=true.")
    for option in ("activation_checkpointing", "gradient_checkpointing", "use_activation_checkpointing"):
        if cfg.model.get(option, False):
            raise ValueError(f"This adapter assumes activation checkpointing is disabled, but model.{option}=true.")


def _load_config(config_path: Path, input_cfg: Path, overrides: Sequence[str]) -> DictConfig:
    """Load model settings and replace the placeholder training input configuration."""

    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    with open_dict(cfg.model.train_ds.dataset):
        cfg.model.train_ds.dataset.input_cfg = str(input_cfg)
    return cfg


def _profile_step(model, optimizer, template, batch_size: int, device: torch.device) -> tuple[bool, float]:
    batch = None
    loss = None
    oom = False
    torch.cuda.reset_peak_memory_stats(device)
    try:
        batch = move_to_device(expand_singleton_batch(template, batch_size), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model.training_step(batch, 0)
        loss.sum().backward()
        optimizer.step()
    except torch.cuda.OutOfMemoryError:
        oom = True
    finally:
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        del loss, batch
        optimizer.zero_grad(set_to_none=True)
        if oom:
            gc.collect()
            torch.cuda.empty_cache()
    return oom, peak_gib


@click.command(context_settings={"show_default": True})
@click.option("--config-path", type=click.Path(path_type=Path, exists=True), default=DEFAULT_CONFIG_PATH)
@click.option(
    "--input-cfg",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    required=True,
    help="Lhotse input-config YAML; replaces model.train_ds.dataset.input_cfg.",
)
@click.option("--buckets", type=BucketList(), help="Defaults to model.train_ds.dataset.bucket_duration_bins.")
@click.option("--scan-batches", type=click.IntRange(min=1), default=5000)
@click.option("--start-batch-size", type=click.IntRange(min=1), default=2)
@click.option("--threshold", type=click.FloatRange(min=0.0, max=1.0), default=0.05)
@click.option(
    "--memory-fraction",
    type=click.FloatRange(min=0.01, max=1.0),
    default=0.85,
    help="CUDA allocator limit; leaves headroom for DDP and non-training allocations.",
)
@click.option("--device", default="cuda:0")
@click.option("--override", "overrides", multiple=True, help="OmegaConf dot-list override; may be repeated.")
def main(
    config_path: Path,
    input_cfg: Path,
    buckets: list[float] | None,
    scan_batches: int,
    start_batch_size: int,
    threshold: float,
    memory_fraction: float,
    device: str,
    overrides: tuple[str, ...],
) -> None:
    """Profile fixed bucket batch sizes for an EasyMagpie Lhotse training config."""

    cfg = _load_config(config_path, input_cfg, overrides)
    _validate_assumptions(cfg)

    if buckets is None:
        buckets = [float(item) for item in cfg.model.train_ds.dataset.bucket_duration_bins]
    if not buckets or buckets != sorted(buckets) or len(set(buckets)) != len(buckets):
        raise click.ClickException("Bucket boundaries must be a non-empty, strictly increasing list.")

    torch_device = torch.device(device)
    if torch_device.type != "cuda" or torch_device.index is None:
        raise click.ClickException("--device must be an indexed CUDA device such as cuda:0.")
    torch.cuda.set_device(torch_device)
    torch.cuda.set_per_process_memory_fraction(memory_fraction, torch_device)
    torch.set_float32_matmul_precision("high")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision="bf16-mixed",
        max_steps=1,
        barebones=True,
    )
    model = EasyMagpieTTSModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    model = model.to(torch_device)
    model.train()
    model.log = lambda *args, **kwargs: None
    model.log_dict = lambda *args, **kwargs: None

    frame_seconds = model.codec_model_samples_per_frame / model.sample_rate
    frames_per_second = 1.0 / frame_seconds
    dataloader_cfg = _profiling_dataloader_config(cfg.model.train_ds, buckets)
    dataloader = model.get_lhotse_dataloader(dataloader_cfg, mode="train")
    model._train_dl = dataloader
    click.echo(f"Scanning up to {scan_batches} singleton cached-code batches...")
    examples = collect_bucket_examples(dataloader, buckets, frames_per_second, scan_batches)
    templates = [
        pad_cached_codes_for_bucket(
            example,
            bucket_duration=bucket,
            context_duration=float(cfg.model.context_duration_max),
            frames_per_second=frames_per_second,
        )
        for example, bucket in zip(examples, buckets)
    ]

    optimizer, _ = model.setup_optimization(cfg.model.optim)
    for param_group in optimizer.param_groups:
        param_group["lr"] = 0.0

    click.echo("Initializing gradients and AdamW state with a singleton batch...")
    oom, _ = _profile_step(model, optimizer, templates[-1], 1, torch_device)
    if oom:
        raise click.ClickException("A singleton batch from the largest duration bucket does not fit on this GPU.")

    profile: list[int] = [0] * len(buckets)
    next_start = start_batch_size
    for index in reversed(range(len(buckets))):
        bucket = buckets[index]
        search = BatchSizeSearch(current=max(1, next_start), relative_gap_threshold=threshold)
        click.echo(f"Profiling bucket <= {bucket:g}s")
        while not search.done:
            oom, peak_gib = _profile_step(model, optimizer, templates[index], search.current, torch_device)
            click.echo(f"  batch_size={search.current}: {'OOM' if oom else 'OK'} (peak={peak_gib:.2f} GiB)")
            search.record(oom)
        assert search.max_ok is not None
        profile[index] = search.max_ok
        next_start = search.max_ok * 2
        click.echo(f"  selected batch_size={search.max_ok}")

    click.echo("\nAdd these values under model.train_ds.dataset:")
    click.echo("bucket_duration_bins: [" + ", ".join(f"{item:g}" for item in buckets) + "]")
    click.echo("bucket_batch_size: [" + ", ".join(str(item) for item in profile) + "]")
    click.echo("batch_duration: null")
    click.echo("quadratic_duration: null")


if __name__ == "__main__":
    main()
