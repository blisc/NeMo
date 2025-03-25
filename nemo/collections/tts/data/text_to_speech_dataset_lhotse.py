# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import random
import re
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import torch
from lhotse import CutSet
from lhotse.dataset.collation import collate_matrices
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse
from megatron.core import parallel_state

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    beta_binomial_prior_distribution,
    normalize_volume,
    stack_tensors,
)
from nemo.utils import logging

SUPPORTED_CODEC_MODEL_NAMES = ["21fpsCausalDecoder", "12fpsCausalDecoder"]


def check_speaker_format(item: str):
    # enforce the format as example like "| Language:en Dataset:HiFiTTS Speaker:9136_other |".
    pattern = r"\| Language:\w+ Dataset:[\w\d\W]+ Speaker:[\w\d\W]+ \|"
    return bool(re.match(pattern, item))


def collate_vectors(items, max_length: int, padding_value):
    vectors = collate_vectors_lhotse(items, padding_value=padding_value)
    if max_length > vectors.size(1):
        vectors = torch.cat(
            [vectors, padding_value * torch.ones(vectors.size(0), max_length - vectors.size(1), dtype=vectors.dtype)],
            dim=1,
        )
    if items[0].shape[0] < 1:
        vectors = vectors.long()
    return vectors

def normalize_volume_torch(audio, volume_level: float = 0.95):
    """Apply peak normalization to the input audio.
    """
    if not (0.0 <= volume_level <= 1.0):
        raise ValueError(f"Volume must be in range [0.0, 1.0], received {volume_level}")

    if audio.size == 0:
        return audio

    max_sample = torch.max(torch.abs(audio))
    if max_sample == 0:
        return audio

    return volume_level * (audio / torch.max(torch.abs(audio)))

def build_lhotse_dataloader(dataset, data_cfg, is_eval=False):
    """Buld dataloader given an input dataset."""
    return get_lhotse_dataloader_from_config(
        data_cfg,
        global_rank=parallel_state.get_data_parallel_rank(),
        world_size=parallel_state.get_data_parallel_world_size(),
        dataset=dataset,
    )


class MagpieTTSLhotseDataset(torch.utils.data.Dataset):
    """
    Class for processing and loading text to speech training examples.

    Args:
        sample_rate: Sample rate to load audio as. If the audio is stored at a different sample rate, then it will
            be resampled.
        text_tokenizer: Tokenizer to apply to the text field.
        speaker_path: Optional, path to JSON file with speaker indices, for multi-speaker training. Can be created with
            scripts.dataset_processing.tts.create_speaker_map.py
        featurizers: Optional, list of featurizers to load feature data from. Should be the same config provided
            when running scripts.dataset_processing.tts.compute_features.py before training.
        feature_processors: Optional, list of feature processors to run on training examples.
        align_prior_hop_length: Optional int, hop length of audio features.
            If provided alignment prior will be calculated and included in batch output. Must match hop length
            of audio features used for training.
        min_duration: Optional float, if provided audio files in the training manifest shorter than 'min_duration'
            will be ignored.
        max_duration: Optional float, if provided audio files in the training manifest longer than 'max_duration'
            will be ignored.
        volume_norm: Whether to apply volume normalization to loaded audio.
    """
    def __init__(
        self,
        sample_rate: int,
        align_prior_hop_length: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        volume_norm: bool = True,
        codec_model_downsample_factor: int = None,
        bos_id: int = None,
        eos_id: int = None,
        audio_bos_id: int = None,
        audio_eos_id: int = None,
        prior_scaling_factor: float = None,
        load_cached_codes_if_available: bool = True,
        dataset_type: str = 'train',
        tokenizer_config=None,
        load_16khz_audio: bool = True,
        use_text_conditioning_tokenizer: bool = False,
        pad_context_text_to_max_duration: bool = False,
        context_duration_min: float = 3.0,
        context_duration_max: float = 10.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.text_tokenizer = None
        self.align_prior_hop_length = align_prior_hop_length
        self.volume_norm = volume_norm

        self.bos_id = bos_id
        self.eos_id = eos_id
        self.audio_bos_id = audio_bos_id
        self.audio_eos_id = audio_eos_id
        self.codec_model_downsample_factor = codec_model_downsample_factor
        self.include_align_prior = prior_scaling_factor is not None
        self.prior_scaling_factor = prior_scaling_factor
        self.load_cached_codes_if_available = load_cached_codes_if_available
        self.dataset_type = dataset_type
        self.tokenizer_config = tokenizer_config
        self.load_16khz_audio = load_16khz_audio
        self.use_text_conditioning_tokenizer = use_text_conditioning_tokenizer
        self.text_conditioning_tokenizer = None
        self.pad_context_text_to_max_duration = pad_context_text_to_max_duration
        self.context_duration_min = context_duration_min
        self.context_duration_max = context_duration_max

    def __getitem__(self, cuts):
        cuts = cuts.sort_by_duration()

        logging.debug(f"Len: {len(cuts)}")

        # load audios and text
        num_codec_frames = []
        align_priors = []
        context_audios = []
        context_audios_lens = []
        target_audios = []
        target_audios_lens = []
        target_audios_16khz = []
        target_audios_16khz_lens = []
        context_text_tokens = []
        context_text_tokens_lens = []
        has_text_context_list = []
        target_text_tokens = []
        target_text_tokens_lens = []

        for i, cut in enumerate(cuts):
            # load target/answer audio
            answer_audio = torch.FloatTensor(cut.target_audio.resample(self.sample_rate).load_audio()).squeeze(0)
            if self.volume_norm:
                answer_audio = normalize_volume_torch(answer_audio)

            answer_audio = torch.nn.functional.pad(
                answer_audio,
                (0, self.codec_model_downsample_factor - (answer_audio.shape[0] % self.codec_model_downsample_factor)),
                value=0
            ).unsqueeze(0)

            answer_audio_len = answer_audio.shape[1]
            target_audios.append(answer_audio)
            target_audios_lens.append(answer_audio_len)
            num_frames = int(answer_audio_len / self.codec_model_downsample_factor) + 1 # +1 for EOS
            num_codec_frames.append(num_frames)

            # load context audio
            context_audio = torch.FloatTensor(cut.resample(self.sample_rate).load_audio()).squeeze(0)
            if self.volume_norm:
                context_audio = normalize_volume_torch(context_audio)

            context_audio = torch.nn.functional.pad(
                context_audio,
                (0, self.codec_model_downsample_factor - (context_audio.shape[0] % self.codec_model_downsample_factor)),
                value=0
            ).unsqueeze(0)
            context_audios_len = context_audio.shape[1]
            context_audios.append(context_audio)
            context_audios_lens.append(context_audios_len)

            # load context text
            if cut.supervisions[0].speaker == "user":
                if self.use_text_conditioning_tokenizer:
                    context_text = cut.supervisions[0].text
                    context_tokenizer = self.text_conditioning_tokenizer if self.text_conditioning_tokenizer else self.text_tokenizer
                    # check if the text is not empty
                    if context_text.replace(" ", ""):
                        context_text = self.text_conditioning_tokenizer(context_text)['input_ids']
                        has_text_context_list.append(True)
                    else:
                        context_text = self.text_conditioning_tokenizer("[NO TEXT CONTEXT]")['input_ids']
                        has_text_context_list.append(False)

                    if self.pad_context_text_to_max_duration:
                        _required_len = int(self.context_duration_max * self.sample_rate / self.codec_model_downsample_factor) + 2 # +2 for BOS and EOS
                        if len(context_text) < _required_len:
                            _pad_id = self.text_conditioning_tokenizer.pad_token_id
                            context_text += [_pad_id] * (_required_len - len(context_text))
                        else:
                            context_text = context_text[:_required_len]

                    context_text = torch.tensor(context_text, dtype=torch.int32)
                    context_text_len = context_text.shape[0]
                    context_text_tokens.append(context_text)
                    context_text_tokens_lens.append(context_text_len)
            else:
                raise Exception("First speaker should be user")

            if cut.supervisions[1].speaker == "agent":
                target_text = cut.supervisions[1].text
                # check if the text is not empty
                if target_text.replace(" ", ""):
                    tokenizer_name = "english_phoneme" # Default to english phoneme tokenizer
                    if getattr(cut, "tokenizer_names", None):
                        # Pick a random tokenizer from the list of tokenizers
                        tokenizer_name = random.choice(cut.tokenizer_names)

                    target_text = self.text_tokenizer.encode(text=target_text, tokenizer_name=tokenizer_name)
                    target_text = target_text + [self.eos_id]
                else:
                    target_text = [self.eos_id]

                target_text = torch.tensor(target_text, dtype=torch.int32)
                target_text_len = target_text.shape[0]
                target_text_tokens.append(target_text)
                target_text_tokens_lens.append(target_text_len)
            else:
                raise Exception("Second speaker should be agent")

            if self.include_align_prior:
                # align_prior = self.beta_binomial_interpolator(spec_len, text_len)
                align_prior = beta_binomial_prior_distribution(phoneme_count=target_text_len, mel_count=num_frames, scaling_factor=self.prior_scaling_factor)
                align_prior = torch.tensor(align_prior, dtype=torch.float32)
                align_priors.append(align_prior)

            if self.load_16khz_audio:
                target_audio_16khz = librosa.resample(answer_audio.squeeze(0).numpy(), orig_sr=self.sample_rate, target_sr=16000)
                target_audio_16khz = torch.FloatTensor(target_audio_16khz).unsqueeze(0)
                target_audio_16khz_len = target_audio_16khz.shape[1]
                target_audios_16khz.append(target_audio_16khz)
                target_audios_16khz_lens.append(target_audio_16khz_len)

        # collate target/agent audios
        target_audios = collate_vectors(
            [a.squeeze(0) for a in target_audios], max_length=max(target_audios_lens), padding_value=0.0
        ).float()
        target_audios_lens = torch.IntTensor(target_audios_lens)
        num_codec_frames = torch.IntTensor(num_codec_frames)

        # collate context/user audios
        context_audios = collate_vectors(
            [a.squeeze(0) for a in context_audios], max_length=max(context_audios_lens), padding_value=0.0
        ).float()
        context_audios_lens = torch.IntTensor(context_audios_lens)

        # collate context/user text
        if self.use_text_conditioning_tokenizer:
            context_text_tokens = collate_vectors(context_text_tokens, max_length=max(context_text_tokens_lens), padding_value=self.text_tokenizer.pad)
            context_text_tokens_lens = torch.IntTensor(context_text_tokens_lens)

        # collate target/agent text
        target_text_tokens = collate_vectors(target_text_tokens, max_length=max(target_text_tokens_lens), padding_value=self.text_tokenizer.pad)
        target_text_tokens_lens = torch.IntTensor(target_text_tokens_lens)

        # collate align prior
        if self.include_align_prior:
            spec_max_len = max([prior.shape[0] for prior in align_priors])
            text_max_len = max([prior.shape[1] for prior in align_priors])
            align_priors = stack_tensors(align_priors, max_lens=[text_max_len, spec_max_len],)

        # collate 16khz target/agent audio
        if self.load_16khz_audio:
            target_audios_16khz = collate_vectors(
                [a.squeeze(0) for a in target_audios_16khz], max_length=max(target_audios_16khz_lens), padding_value=0.0
            ).float()
            target_audios_16khz_lens = torch.IntTensor(target_audios_16khz_lens)

        batch_dict = {
            # "dataset_names": dataset_names,
            # "audio_filepaths": audio_filepath_list,
            "sample_ids": list(cuts.ids),
            "text": target_text_tokens,
            "text_lens": target_text_tokens_lens,
            'audio': target_audios,
            'audio_lens': target_audios_lens,
            # 'audio_codes': batch_audio_codes
            # 'audio_codes_lens': batch_audio_codes_len
            'context_audio': context_audios,
            'context_audio_lens': context_audios_lens,
            # 'context_audio_codes': batch_context_audio_codes
            # 'context_audio_codes_lens': batch_context_audio_codes_len
        }

        if self.include_align_prior:
            batch_dict["align_prior_matrix"] = align_priors

        if self.load_16khz_audio:
            batch_dict['audio_16khz'] = target_audios_16khz
            batch_dict['audio_lens_16khz'] = target_audios_16khz_lens

        if self.use_text_conditioning_tokenizer:
            batch_dict['context_text_tokens'] = context_text_tokens
            batch_dict['context_text_len'] = context_text_tokens_lens
            batch_dict['has_text_context'] = torch.BoolTensor(has_text_context_list)

        return batch_dict


    def collate_fn(self, batch: List[dict]):
        return batch


class MagpieTTSMonologueLhotseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        # dataset_meta: Dict,
        sample_rate: int,
        # weighted_sampling_steps_per_epoch: Optional[int] = None, TODO @xueyang: this should be moved to lhotse sampler.
        # min_duration: Optional[float] = None,
        # max_duration: Optional[float] = None,
        volume_norm: bool = True,
        codec_model_downsample_factor: int = None,
        codec_model_name: str = "21fpsCausalDecoder",
        bos_id: int = None,
        eos_id: int = None,
        pad_id: int = None,
        audio_bos_id: int = None,
        audio_eos_id: int = None,
        context_audio_bos_id: int = None,
        context_audio_eos_id: int = None,
        num_audio_codebooks: int = None,
        prior_scaling_factor: float = None,
        load_cached_codes_if_available: bool = True,
        dataset_type: str = 'train',
        load_16khz_audio: bool = True,
        pad_context_text_to_max_duration: bool = False,
        context_duration_min: float = 3.0,
        context_duration_max: float = 10.0,
        # tokenizer_name: str = "english_phoneme",
        # tokenizer_config=None,
        use_text_conditioning_tokenizer: bool = False,
        # text_tokenizer=None,
        # text_conditioning_tokenizer=None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        # self.min_duration = min_duration
        # self.max_duration = max_duration
        # self.text_tokenizer = text_tokenizer
        self.volume_norm = volume_norm
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.audio_bos_id = audio_bos_id
        self.audio_eos_id = audio_eos_id
        self.context_audio_bos_id = context_audio_bos_id
        self.context_audio_eos_id = context_audio_eos_id

        if codec_model_name not in SUPPORTED_CODEC_MODEL_NAMES:
            raise ValueError(f"Invalid `codec_model_name`: {codec_model_name}.")
        self.codec_model_name = codec_model_name
        self.codec_model_downsample_factor = codec_model_downsample_factor
        self.num_audio_codebooks = num_audio_codebooks

        self.include_align_prior = prior_scaling_factor is not None
        self.prior_scaling_factor = prior_scaling_factor
        self.load_cached_codes_if_available = load_cached_codes_if_available
        self.dataset_type = dataset_type
        # self.tokenizer_config = tokenizer_config
        self.load_16khz_audio = load_16khz_audio
        if use_text_conditioning_tokenizer is True:
            raise NotImplementedError("Initialization of text context tokenizer has not been implemented yet.")
        self.use_text_conditioning_tokenizer = use_text_conditioning_tokenizer
        # self.text_conditioning_tokenizer = text_conditioning_tokenizer
        self.pad_context_text_to_max_duration = pad_context_text_to_max_duration
        self.context_duration_min = context_duration_min
        self.context_duration_max = context_duration_max
        # self.tokenizer_name = tokenizer_name

    def get_num_audio_samples_to_slice(self, duration, sample_rate):
        num_codec_frames = int(duration * sample_rate / self.codec_model_downsample_factor)
        num_audio_samples = num_codec_frames * self.codec_model_downsample_factor
        return num_audio_samples

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List]]:
        # TODO @xueyang: need to confirm where to add such filter.
        # filtered_cuts_in_duration = cuts.filter(
        #     lambda x: self.min_duration <= x.recording.duration <= self.max_duration
        # )

        # define list to store batched information
        dataset_name_list = []
        audio_list = []
        audio_len_list = []
        audio_list_16khz = []
        audio_len_list_16khz = []
        token_list = []
        token_len_list = []
        prior_list = []
        audio_codes_list = []
        audio_codes_len_list = []
        context_audio_list = []
        context_audio_len_list = []
        context_audio_codes_list = []
        context_audio_codes_len_list = []
        context_text_tokens_list = []
        context_text_tokens_len_list = []
        context_has_text_context_list = []
        reward_list = []
        raw_text_list = []
        target_codes_field = f"codes_{self.codec_model_name}"
        context_codes_field = f"context_codes_{self.codec_model_name}"
        for cut in cuts:
            speaker = cut.supervisions[0].speaker
            if not check_speaker_format(speaker):
                raise ValueError(f"Invalid format in cut.supervisions[0].speaker: {speaker}")
            dataset_name = speaker.strip().split()[2].split(":")[-1]
            dataset_name_list.append(dataset_name)

            # target audio or target codes
            if self.load_cached_codes_if_available and cut.has_custom(target_codes_field):
                audio_codes = torch.from_numpy(cut.load_custom(target_codes_field)).long()  # (8, T)
                audio_bos_tensor = torch.full((audio_codes.shape[0], 1), self.audio_bos_id, dtype=audio_codes.dtype)
                audio_eos_tensor = torch.full((audio_codes.shape[0], 1), self.audio_eos_id, dtype=audio_codes.dtype)
                audio_codes = torch.cat([audio_bos_tensor, audio_codes, audio_eos_tensor], dim=1)
                audio_codes_len = audio_codes.shape[1]
                spec_len = audio_codes.shape[1] + 1  # +1 for EOS
                audio_codes_list.append(
                    audio_codes.T
                )  # transpose to (T, 8) in order to use collate_matrices to process batch.
                audio_codes_len_list.append(audio_codes_len)
            else:
                # Only load audio if codes are not available
                audio_array = cut.recording.resample(self.sample_rate).load_audio().squeeze(0)
                if self.volume_norm:
                    audio_array = normalize_volume(audio_array)
                audio = torch.from_numpy(audio_array)
                # Pad audio to be multiple of downsample factor
                audio = torch.nn.functional.pad(
                    audio,
                    (0, self.codec_model_downsample_factor - (audio.shape[0] % self.codec_model_downsample_factor)),
                    value=0,
                )
                audio_len = audio.shape[0]
                spec_len = int(audio_len / self.codec_model_downsample_factor) + 1  # +1 for EOS
                audio_list.append(audio)
                audio_len_list.append(audio_len)

            # context audio or context codes
            if self.load_cached_codes_if_available and cut.has_custom(context_codes_field):
                # TODO @xueyang: dev branch applied Tensor.long(), i.e. torch.int64 which is not necessary.
                # load audios and text
                context_audio_codes = torch.from_numpy(cut.load_custom(context_codes_field)).long()  # (8, T)
                # Sample random duration between self.context_duration_min and self.context_duration_max
                _context_duration_to_slice = random.uniform(self.context_duration_min, self.context_duration_max)
                _num_frames_to_slice = int(
                    _context_duration_to_slice * self.sample_rate / self.codec_model_downsample_factor
                )
                if _num_frames_to_slice < context_audio_codes.shape[1]:
                    start_idx = random.randint(0, context_audio_codes.shape[1] - _num_frames_to_slice)
                    context_audio_codes = context_audio_codes[:, start_idx : start_idx + _num_frames_to_slice]
                else:
                    # Repeat the audio if it is shorter than the desired duration
                    _num_repeats = int(np.ceil(_num_frames_to_slice / context_audio_codes.shape[1]))
                    # context_audio_codes is a tensor of shape (num_codebooks, T)
                    context_audio_codes_repeated = context_audio_codes.repeat(1, _num_repeats)
                    context_audio_codes = context_audio_codes_repeated[:, :_num_frames_to_slice]

                context_bos_tensor = torch.full(
                    (context_audio_codes.shape[0], 1), self.context_audio_bos_id, dtype=context_audio_codes.dtype
                )
                context_eos_tensor = torch.full(
                    (context_audio_codes.shape[0], 1), self.context_audio_eos_id, dtype=context_audio_codes.dtype
                )
                context_audio_codes = torch.cat([context_bos_tensor, context_audio_codes, context_eos_tensor], dim=1)
                context_audio_codes_len = context_audio_codes.shape[1]
                context_audio_codes_list.append(
                    context_audio_codes.T
                )  # transpose to (T, 8) in order to use collate_matrices to process batch.
                context_audio_codes_len_list.append(context_audio_codes_len)
            elif cut.has_custom("context_recording"):
                # Only load audio if codes are not available
                context_audio_array = cut.context_recording.resample(self.sample_rate).load_audio().squeeze(0)
                if self.volume_norm:
                    context_audio_array = normalize_volume(context_audio_array)
                _context_duration_to_slice = random.uniform(self.context_duration_min, self.context_duration_max)
                _num_samples_to_slice = self.get_num_audio_samples_to_slice(
                    _context_duration_to_slice, self.sample_rate
                )
                if _num_samples_to_slice < len(context_audio_array):
                    start_idx = random.randint(0, len(context_audio_array) - _num_samples_to_slice)
                    context_audio_array = context_audio_array[start_idx : start_idx + _num_samples_to_slice]
                else:
                    # Repeat the audio if it is shorter than the desired duration
                    _num_repeats = int(np.ceil(_num_samples_to_slice / len(context_audio_array)))
                    context_audio_array = np.tile(context_audio_array, _num_repeats)
                    context_audio_array = context_audio_array[:_num_samples_to_slice]
                context_audio = torch.from_numpy(context_audio_array)
                context_audio_len = context_audio.shape[0]
                context_audio_list.append(context_audio)
                context_audio_len_list.append(context_audio_len)
            else:
                # We always want to have context_audio_codes if available for multi-encoder model. These are ignored
                # for singlencoder model.
                # If context audio is not available, just use a dummy context_audio_codes
                # (Will be used in text context scenario)
                # TODO @xueyang: verified that this block should cover below 3 conditions which were handled well.
                #  1. load_cached_codes_if_available and ["context_audio_codes_path", "context_audio_filepath"] not in data.manifest_entry;
                #        assign to example["context_audio_codes"] and example["context_audio_codes_len"]
                #  2. load_cached_codes_if_available is not True and "context_audio_codes_path" in data.manifest_entry;
                #        assign to example["context_audio"] and example["context_audio_len"]
                #  3. load_cached_codes_if_available is not True and ["context_audio_codes_path", "context_audio_filepath"] not in data.manifest_entry;
                #        assign to example["context_audio"] and example["context_audio_len"]
                if self.load_cached_codes_if_available:
                    context_bos_tensor = torch.full(
                        (self.num_audio_codebooks, 1), self.context_audio_bos_id, dtype=torch.int32
                    )
                    context_eos_tensor = torch.full(
                        (self.num_audio_codebooks, 1), self.context_audio_eos_id, dtype=torch.int32
                    )
                    context_audio_codes = torch.cat([context_bos_tensor, context_eos_tensor], dim=1)
                    context_audio_codes_len = context_audio_codes.shape[1]
                    context_audio_codes_list.append(
                        context_audio_codes.T
                    )  # transpose to (T, 8) in order to use collate_matrices to process batch.
                    context_audio_codes_len_list.append(context_audio_codes_len)
                else:
                    # @shehzeenh: Added this condition so that a batch does not have a mix of context_audio and context_audio_codes
                    context_audio = torch.zeros(self.codec_model_downsample_factor, dtype=torch.float32)
                    context_audio_len = context_audio.shape[0]
                    context_audio_list.append(context_audio)
                    context_audio_len_list.append(context_audio_len)

            if self.load_16khz_audio:
                if cut.has_custom("context_recording"):
                    # use context audio for SV model
                    audio_array_16khz = cut.context_recording.resample(16_000).load_audio().squeeze(0)
                    if self.volume_norm:
                        audio_array_16khz = normalize_volume(audio_array_16khz)
                else:
                    # Otherwise, load the target audio for SV model.
                    audio_array_16khz = cut.recording.resample(16_000).load_audio().squeeze(0)
                    if self.volume_norm:
                        audio_array_16khz = normalize_volume(audio_array_16khz)
                _context_duration_to_slice = random.uniform(self.context_duration_min, self.context_duration_max)
                _num_samples_to_slice = int(_context_duration_to_slice * 16_000)
                if _num_samples_to_slice < len(audio_array_16khz):
                    start_idx = random.randint(0, len(audio_array_16khz) - _num_samples_to_slice)
                    audio_array_16khz = audio_array_16khz[start_idx : start_idx + _num_samples_to_slice]
                audio_16khz = torch.from_numpy(audio_array_16khz)
                audio_len_16khz = audio_16khz.shape[0]
                audio_list_16khz.append(audio_16khz)
                audio_len_list_16khz.append(audio_len_16khz)

            if self.use_text_conditioning_tokenizer:
                raise NotImplementedError("Initialization of text context tokenizer has not been implemented yet.")
                if cut.supervisions[0].has_custom("context_text"):
                    context_text_tokens = self.text_conditioning_tokenizer(cut.supervisions[0].context_text)[
                        'input_ids'
                    ]
                    has_text_context = True
                else:
                    context_text_tokens = self.text_conditioning_tokenizer("[NO TEXT CONTEXT]")['input_ids']
                    has_text_context = False
                if self.pad_context_text_to_max_duration:
                    _required_len = (
                        int(self.context_duration_max * self.sample_rate / self.codec_model_downsample_factor) + 2
                    )  # +2 for BOS and EOS
                    if len(context_text_tokens) < _required_len:
                        _pad_id = self.text_conditioning_tokenizer.pad_token_id
                        context_text_tokens += [_pad_id] * (_required_len - len(context_text_tokens))
                    else:
                        # TODO @xueyang: It seems counter intuition if trimming the text context tokens to the required
                        #  context length. For example, the context_tokens after trimming may correspond to the partial
                        #  context_text like "Speaker and Emotion: | Language:en Dataset(trimmed :Riva Speaker:Rodney_DROP |)"
                        context_text_tokens = context_text_tokens[:_required_len]
                context_text_tokens = torch.tensor(context_text_tokens, dtype=torch.int32)
                context_text_tokens_len = context_text_tokens.shape[0]
                context_text_tokens_list.append(context_text_tokens)
                context_text_tokens_len_list.append(context_text_tokens_len)
                context_has_text_context_list.append(has_text_context)

            # tokenize transcript
            # TODO @xueyang: temporally apply raw text. will check to change if normalized text is available.
            raw_text = cut.supervisions[0].text
            raw_text_list.append(raw_text)
            # if cut.supervisions[0].has_custom("tokenizer_names"):
                # Pick a random tokenizer from the list of tokenizers
                # self.tokenizer_name =
                # random.choice(cut.supervisions[0].tokenizer_names)
            # import ipdb; ipdb.set_trace()
            # if cut.tokenizer_names:
                # Pick a random tokenizer from the list of tokenizers
                # self.tokenizer_name = random.choice(cut.tokenizer_names)
            # tokens = self.text_tokenizer.encode(text=raw_text, tokenizer_name=self.tokenizer_name)
            # tokens = self.text_tokenizer(text=raw_text, tokenizer_name=self.tokenizer_name)
            # cut.tokens is obtained from nemo/collections/common/data/lhotse/dataloader.py::tokenize
            tokens = cut.tokens + [self.eos_id]  # Not adding BOS id
            tokens = torch.tensor(tokens, dtype=torch.int32)
            text_len = tokens.shape[0]
            token_list.append(tokens)
            token_len_list.append(text_len)

            if self.include_align_prior:
                # align_prior = self.beta_binomial_interpolator(spec_len, text_len)
                align_prior = beta_binomial_prior_distribution(
                    phoneme_count=text_len, mel_count=spec_len, scaling_factor=self.prior_scaling_factor
                )
                align_prior = torch.tensor(align_prior, dtype=torch.float32)
                prior_list.append(align_prior)

            if cut.supervisions[0].has_custom("reward"):
                reward = cut.supervisions[0].reward
                reward_list.append(reward)

        # collate vectors and matrices here.
        batch_dict = {
            "dataset_names": dataset_name_list,
            "raw_texts": raw_text_list,
            "text": collate_vectors_lhotse(token_list, padding_value=self.pad_id),  # (B, max_len)
            "text_lens": torch.IntTensor(token_len_list),
        }

        # audio for SV.
        if len(audio_list_16khz) > 0:
            batch_dict["audio_16khz"] = collate_vectors_lhotse(audio_list_16khz, padding_value=0.0)
            batch_dict["audio_lens_16khz"] = torch.IntTensor(audio_len_list_16khz)

        # target audio and codes
        if len(audio_list) > 0:
            batch_dict["audio"] = collate_vectors_lhotse(audio_list, padding_value=0.0)
            batch_dict["audio_lens"] = torch.IntTensor(audio_len_list)
        if len(audio_codes_list) > 0:
            # transpose back to (B, 8, T) from (B, T, 8).
            batch_dict["audio_codes"] = collate_matrices(audio_codes_list, padding_value=0).transpose(1, 2)
            batch_dict["audio_codes_lens"] = torch.IntTensor(audio_codes_len_list)

        # context audio and codes
        if len(context_audio_list) > 0:
            batch_dict["context_audio"] = collate_vectors_lhotse(context_audio_list, padding_value=0.0)
            batch_dict["context_audio_lens"] = torch.IntTensor(context_audio_len_list)
        if len(context_audio_codes_list) > 0:
            # transpose back to (B, 8, T) from (B, T, 8).
            batch_dict["context_audio_codes"] = collate_matrices(context_audio_codes_list, padding_value=0).transpose(1, 2)
            batch_dict["context_audio_codes_lens"] = torch.IntTensor(context_audio_codes_len_list)

        if self.use_text_conditioning_tokenizer:
            batch_dict['context_text_tokens'] = collate_vectors_lhotse(context_text_tokens_list, padding_value=self.text_conditioning_tokenizer.pad_token_id)
            batch_dict['context_text_tokens_lens'] = torch.IntTensor(context_text_tokens_len_list)
            batch_dict['has_text_context'] = torch.BoolTensor(context_has_text_context_list)

        if self.include_align_prior:
            spec_max_len = max([prior.shape[0] for prior in prior_list])
            text_max_len = max([prior.shape[1] for prior in prior_list])
            batch_dict["align_prior_matrix"] = stack_tensors(prior_list, max_lens=[text_max_len, spec_max_len])

        if len(reward_list) > 0:
            batch_dict['rewards'] = torch.FloatTensor(reward_list)

        # Assert only ONE of context_audio or context_audio_codes in the batch
        assert ('audio' in batch_dict) ^ ('audio_codes' in batch_dict)

        # Assert only ONE of context_audio or context_audio_codes in the batch
        if 'context_audio' in batch_dict:
            assert 'context_audio_codes' not in batch_dict
        if 'context_audio_codes' in batch_dict:
            assert 'context_audio' not in batch_dict

        return batch_dict