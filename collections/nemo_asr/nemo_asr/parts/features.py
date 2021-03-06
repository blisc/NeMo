# Taken straight from Patter https://github.com/ryanleary/patter
# TODO: review, and copyright and fix/add comments
import itertools
import math

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch_stft import STFT

from .perturb import AudioAugmentor
from .segment import AudioSegment

CONSTANT = 1e-5


def normalize_batch(x, seq_len, normalize_type):
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                             device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                            device=x.device)
        for i in range(x.shape[0]):
            x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, :seq_len[i].item()].mean()
            x_std[i] = x[i, :, :seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
    else:
        return x


def splice_frames(x, frame_splicing):
    """ Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
    return torch.cat(seq, dim=1)


class WaveformFeaturizer(object):
    def __init__(self, sample_rate=16000, int_values=False, augmentor=None,
                 speed_perturb=False):
        self.augmentor = augmentor if augmentor is not None else \
            AudioAugmentor()
        self.sample_rate = sample_rate
        self.int_values = int_values
        self.speed_perturb = speed_perturb

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(self, file_path, offset=0, duration=0, trim=False):
        target_sr = self.sample_rate
        if self.speed_perturb:
            target_sr *= np.random.uniform(0.85, 1.15)
        audio = AudioSegment.from_file(
            file_path,
            target_sr=target_sr,
            int_values=self.int_values,
            offset=offset, duration=duration, trim=trim)
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)
        return torch.tensor(audio_segment.samples, dtype=torch.float)

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None

        sample_rate = input_config.get("sample_rate", 16000)
        int_values = input_config.get("int_values", False)

        return cls(sample_rate=sample_rate, int_values=int_values,
                   augmentor=aa)


class FeaturizerFactory(object):
    def __init__(self):
        pass

    @classmethod
    def from_config(cls, input_cfg, perturbation_configs=None):
        return WaveformFeaturizer.from_config(
            input_cfg,
            perturbation_configs=perturbation_configs)


class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.
    See AudioToMelSpectrogramPreprocessor for args.
    """
    def __init__(
            self, *,
            sample_rate=16000,
            n_window_size=320,
            n_window_stride=160,
            window="hann",
            normalize="per_feature",
            n_fft=None,
            preemph=0.97,
            nfilt=64,
            lowfreq=0,
            highfreq=None,
            log=True,
            dither=CONSTANT,
            pad_to=16,
            max_duration=16.7,
            frame_splicing=1,
            stft_conv=False,
            pad_value=0,
            mag_power=2.,
            logger=None
    ):
        super(FilterbankFeatures, self).__init__()
        if (n_window_size is None or n_window_stride is None
                or not isinstance(n_window_size, int)
                or not isinstance(n_window_stride, int)
                or n_window_size <= 0 or n_window_stride <= 0):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints.")
        if logger:
            logger.info(f"PADDING: {pad_to}")
        else:
            print(f"PADDING: {pad_to}")

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_conv = stft_conv

        if stft_conv:
            if logger:
                logger.info("STFT using conv")
            else:
                print("STFT using conv")

            # Create helper class to patch forward func for use with AMP
            class STFTPatch(STFT):
                def __init__(self, *params, **kw_params):
                    super(STFTPatch, self).__init__(*params, **kw_params)

                def forward(self, input_data):
                    return super(STFTPatch, self).transform(input_data)

            self.stft = STFTPatch(self.n_fft, self.hop_length,
                                  self.win_length, window)

        else:
            print("STFT using torch")
            torch_windows = {
                'hann': torch.hann_window,
                'hamming': torch.hamming_window,
                'blackman': torch.blackman_window,
                'bartlett': torch.bartlett_window,
                'none': None,
            }
            window_fn = torch_windows.get(window, None)
            window_tensor = window_fn(self.win_length,
                                      periodic=False) if window_fn else None
            self.register_buffer("window", window_tensor)
            self.stft = lambda x: torch.stft(
                            x, n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            center=True,
                            window=self.window.to(dtype=torch.float))

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        # self.speed_perturb = None
        # if speed_perturb:
        #     self.speed_perturb = SpeedAugmentation(
        #         segments=speed_perturb_segs,
        #         min_segment_size=speed_perturb_min,
        #         max_segment_size=speed_perturb_max,
        #         global_=speed_perturb_global)

        highfreq = highfreq or sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(sample_rate, self.n_fft, n_mels=nfilt,
                                fmin=lowfreq, fmax=highfreq),
            dtype=torch.float).unsqueeze(0)
        # self.fb = filterbanks
        # self.window = window_tensor
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(
            torch.tensor(max_duration * sample_rate, dtype=torch.float))
        max_pad = pad_to - (max_length % pad_to)
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

    def get_seq_len(self, seq_len):
        return torch.ceil(seq_len / self.hop_length).to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    @torch.no_grad()
    def forward(self, x, seq_len):
        seq_len = self.get_seq_len(seq_len.float())

        # dither
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat(
                (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]),
                dim=1)

        x = self.stft(x)

        # if self.speed_perturb and self.training:
        #     mag = x[0].cpu().numpy()
        #     phase = x[1].cpu().numpy()
        #     x, seq_len = self.speed_perturb(mag*np.exp(phase*1j), seq_len)
        #     x = x.cuda()
        #     seq_len = seq_len.cuda()
        # else:
        #     x = x[0]
        x = x[0]

        # get power spectrum
        if self.mag_power != 1.:
            x = x.pow(self.mag_power)
        if not self.stft_conv:
            x = x.sum(-1)

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            x = torch.log(torch.clamp(x, min=1e-5))

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of
        # `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(x.device)
        mask = mask.expand(x.size(0), max_len) >= seq_len.unsqueeze(1)
        x = x.masked_fill(
            mask.unsqueeze(1).type(torch.bool).to(device=x.device),
            self.pad_value)
        del mask
        pad_to = self.pad_to
        if not self.training:
            pad_to = 16
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)),
                                  value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt),
                                      value=self.pad_value)
        return x


class SpeedAugmentation(nn.Module):
    """ This is not speed perturbation. This is time stretch. Not the same!"""
    def __init__(
            self, *,
            segments=0,
            min_segment_size=10,
            max_segment_size=None,
            global_=False,
            num_processes=8
    ):
        super().__init__()
        self.global_ = global_
        self.segments = segments
        self.min_segment_size = min_segment_size
        self.max_segment_size = max_segment_size
        try:
            self.pool = mp.Pool(processes=num_processes)
            self.mp = True
            print("WOW, multiprocessing")
        except Exception as e:
            print(e)
            print(":(, multiprocessing failed")
            self.mp = False

    @staticmethod
    def do_perturb(args):
        spec, spec_len, args = args
        global_, segments, min_segment_size, max_segment_size = args
        if global_:
            rate_min = max(spec_len / 1680, 0.85)
            rate = np.random.uniform(rate_min, 1.15)
            return librosa.core.phase_vocoder(spec[:, 0:spec_len], rate)
        # else
        for _ in range(segments):
            min_seg_length = int(min_segment_size * spec_len)
            max_seg_length = int(max_segment_size * spec_len)
            slice_start = np.random.randint(
                spec_len - min_seg_length - 1)
            if slice_start + max_seg_length >= spec_len:
                max_seg_length = spec_len - slice_start
            slice_length = np.random.randint(
                min_seg_length,
                max_seg_length)
            slice_end = slice_start + slice_length
            rate_min = max(
                slice_length / (1680 - spec_len + slice_length), 0.85)
            slice_ = spec[:, slice_start:slice_end]
            rate = np.random.uniform(rate_min, 1.15)
            slice_perturbed = librosa.core.phase_vocoder(slice_, rate)
            spec = np.concatenate(
                (
                    spec[:, :slice_start],
                    slice_perturbed,
                    spec[:, slice_end:spec_len]
                ),
                axis=1)
            spec_len = spec.shape[1]
        return spec

    def forward(self, input_spec, input_spec_length):
        input_spec_length = input_spec_length.cpu().numpy().astype(int)
        if self.mp:
            output_specs = self.pool.map(
                SpeedAugmentation.do_perturb,
                zip(
                    input_spec, input_spec_length,
                    itertools.repeat((
                        self.global_, self.segments, self.min_segment_size,
                        self.max_segment_size))
                ))
        else:
            output_specs = []
            for i in range(input_spec.shape[0]):
                output_specs.append(SpeedAugmentation.do_perturb((
                    input_spec[i], input_spec_length[i],
                    (
                        self.global_, self.segments, self.min_segment_size,
                        self.max_segment_size
                    )
                )))

        # Collate
        max_len = -1
        for i, spec in enumerate(output_specs):
            input_spec_length[i] = spec.shape[1]
            if input_spec_length[i] > max_len:
                max_len = input_spec_length[i]
        output_spec = torch.zeros(input_spec.shape[0], input_spec.shape[1],
                                  max_len, dtype=torch.float)
        input_spec_length = torch.from_numpy(input_spec_length)
        for i in range(input_spec.shape[0]):
            output_spec[i].narrow(1, 0, input_spec_length[i]).copy_(
                torch.from_numpy(np.absolute(output_specs[i])))

        return output_spec, input_spec_length
