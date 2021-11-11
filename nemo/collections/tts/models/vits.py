from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict

import numpy as np
import torch
from hydra.utils import instantiate
import omegaconf
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger, WandbLogger
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import attentions
import monotonic_align
from torch.cuda.amp import autocast

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

from nemo.collections.asr.data.audio_to_text import FastPitchDataset
from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.helpers.helpers import plot_spectrogram_to_numpy, regulate_len
from nemo.collections.tts.models.base import TextToWaveform
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    MelSpectrogramType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.optim.lr_scheduler import NoamAnnealing
from nemo.utils import logging
from nemo.collections.tts.models.base import TextToWaveform
from nemo.collections.tts.losses.vits_losses import DiscriminatorLoss, FeatureLoss, GeneratorLoss, KlLoss
import nemo.collections.tts.modules.vits_modules as modules


HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False

@dataclass
class VitsConfig:
    parser: Dict[Any, Any] = MISSING
    preprocessor: Dict[Any, Any] = MISSING
    input_fft: Dict[Any, Any] = MISSING
    output_fft: Dict[Any, Any] = MISSING
    duration_predictor: Dict[Any, Any] = MISSING
    pitch_predictor: Dict[Any, Any] = MISSING


class Vits(TextToWaveform):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        self._parser = parsers.make_parser(
            labels=cfg.labels,
            name='en',
            unk_id=-1,
            blank_id=-1,
            do_normalize=True,
            abbreviation_version="fastpitch",
            make_table=False,
        )

        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(VitsConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        self.audio_to_melspec_precessor = instantiate(cfg.preprocessor)
        self.melspec_fn = instantiate(cfg.preprocessor, highfreq=None, use_grads=True)

        self.encoder = instantiate(cfg.input_fft)
        self.duration_predictor = instantiate(cfg.duration_predictor)
        self.pitch_predictor = instantiate(cfg.pitch_predictor)

        self.generator = instantiate(cfg.generator)
        self.multiperioddisc = modules.MultiPeriodDiscriminator()
        self.feat_matching_loss = FeatureLoss()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()

        self.max_token_duration = cfg.max_token_duration

        self.pitch_emb = torch.nn.Conv1d(
            1,
            cfg.symbols_embedding_dim,
            kernel_size=cfg.pitch_embedding_kernel_size,
            padding=int((cfg.pitch_embedding_kernel_size - 1) / 2),
        )

        # Store values precomputed from training data for convenience
        self.register_buffer('pitch_mean', torch.zeros(1))
        self.register_buffer('pitch_std', torch.zeros(1))

        self.mel_loss_coeff = cfg.mel_loss_coeff

        self.log_train_images = False
        self.logged_real_samples = False
        self._tb_logger = None
        self.hann_window = None
        self.splice_length = cfg.splice_length
        self.sample_rate = cfg.sample_rate
        self.hop_size = cfg.hop_size

    def parse(self, str_input: str) -> torch.tensor:
        # TODO: Implement
        pass

    def configure_optimizers(self):
        self.optim_g = torch.optim.AdamW(
            self.net_g.parameters(),
            self._cfg.model.lr,
            betas=self._cfg.model.betas,
            eps=self._cfg.model.eps)
        self.optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            self._cfg.model.lr,
            betas=self._cfg.model.betas,
            eps=self._cfg.model.eps)

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=self._cfg.model.lr_decay)
        scheduler_g_dict = {
            'scheduler': scheduler_g,
            'interval': 'step',
        }
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=self._cfg.model.lr_decay)
        scheduler_d_dict = {
            'scheduler': scheduler_d,
            'interval': 'step'
        }
        return [self.optim_g, self.optim_d], [scheduler_g_dict, scheduler_d_dict]

    def forward(self, batch, batch_idx):
        with torch.no_grad():
            (x, x_lengths, spec, spec_lengths, y, y_lengths) = batch

            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]

            y_hat, attn, mask, *_ = self.net_g.module.infer(x, x_lengths, max_len=1000)
            y_hat_lengths = mask.sum([1, 2]).long() * self._cfg.model.hop_size

        return y_hat[0, :, :y_hat_lengths[0]]

    def training_step(self, batch, batch_idx):
        (x, x_lengths, spec, spec_lengths, y, y_lengths) = batch

        with autocast(enabled=False):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(x, x_lengths, spec, spec_lengths)
            mel = modules.spec_to_mel_torch(
                spec,
                self._cfg.model.train_ds.filter_length,
                self._cfg.model.n_mel_channels,
                self._cfg.model.sample_rate,
                self._cfg.model.mel_fmin,
                self._cfg.model.mel_fmax
            )
            y_mel = commons.slice_segments(mel, ids_slice, self._cfg.model.segment_size // self._cfg.model.hop_size)
            y_hat_mel = modules.mel_spectrogram_torch(
                y_hat.squeeze(1),
                self._cfg.model.train_ds.filter_length,
                self._cfg.model.n_mel_channels,
                self._cfg.model.sample_rate,
                self._cfg.model.hop_size,
                self._cfg.model.preprocessing.n_window_size,
                self._cfg.model.mel_fmin,
                self._cfg.model.mel_fmax
            )
            y = commons.slice_segments(y, ids_slice * self._cfg.model.hop_size, self._cfg.model.segment_size)  # slice
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = DiscriminatorLoss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc

        # train discriminator
        self.optim_d.zero_grad()
        self.manual_backward(loss_disc_all)
        commons.clip_grad_value_(self.net_d.parameters(), None)
        self.optim_d.step()

        with autocast(enabled=True):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
        with autocast(enabled=False):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self._cfg.model.c_mel
            loss_kl = KlLoss(z_p, logs_q, m_p, logs_p, z_mask) * self._cfg.model.c_kl

            loss_fm = FeatureLoss(fmap_r, fmap_g)
            loss_gen, losses_gen = GeneratorLoss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        # train generator
        self.optim_g.zero_grad()
        self.manual_backward(loss_gen_all)
        commons.clip_grad_value_(self.net_g.parameters(), None)
        self.optim_d.step()

        schedulers = self.lr_schedulers()
        if schedulers is not None:
            sch1, sch2 = schedulers
            sch1.step()
            sch2.step()

    def validation_step(self, batch, batch_idx):
        (x, x_lengths, spec, spec_lengths, y, y_lengths) = batch

        y_hat, attn, mask, *_ = self.net_g.module.infer(x, x_lengths, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * self.hps.data.hop_length

        # Note to modify the functions / use the ones in NeMo, we need the lengths
        mel, mel_lengths = self.audio_to_melspec_precessor(x, x_lengths)
        y_hat_mel, y_hat_mel_lengths = self.audio_to_melspec_precessor(y, y_lengths)

        loss_mel = F.l1_loss(mel, y_hat_mel)

        self.log_dict({"val_loss": loss_mel}, on_epoch=True, sync_dist=True)

        # plot audio once per epoch
        if batch_idx == 0 and isinstance(self.logger, WandbLogger) and HAVE_WANDB:
            clips = []
            specs = []

            for i in range(min(5, y.shape[0])):
                clips += [
                    wandb.Audio(
                        y[i, : y_lengths[i]].data.cpu().numpy(),
                        caption=f"real audio {i}",
                        sample_rate=self.hps.data.sampling_rate,
                    ),
                    wandb.Audio(
                        y_hat[i, : y_hat_lengths[i]].data.cpu().numpy().astype('float32'),
                        caption=f"generated audio {i}",
                        sample_rate=self.hps.data.sampling_rate,
                    ),
                ]

                specs += [
                    wandb.Image(
                        plot_spectrogram_to_numpy(y_hat_mel[i, :, : y_hat_mel_lengths[i]].data.cpu().numpy()),
                        caption=f"output mel {i}",
                    ),
                    wandb.Image(
                        plot_spectrogram_to_numpy(mel[i, :, : mel_lengths[i]].cpu().numpy()),
                        caption=f"gt mel {i}",
                    ),
                ]

            self.logger.experiment.log({"audio": clips, "specs": specs})

    @staticmethod
    def _loader(cfg):
        try:
            _ = cfg.dataset.manifest_filepath
        except omegaconf.errors.MissingMandatoryValue:
            logging.warning("manifest_filepath was skipped. No dataset for this model.")
            return None

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        list_of_models = []
        # TODO: List available models??
        return list_of_models

    def convert_text_to_waveform(self, *, tokens):
        #  TODO: Convert text to waveforms
        pass

