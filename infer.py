import time
import argparse

import numpy as np
import torch
import tqdm
from scipy.io.wavfile import write

from nemo.collections.asr.data.audio_to_text import AudioToCharDataset, _AudioTextDataset
from nemo.collections.tts.models import HifiGanModel, Tacotron2Model
from nemo.collections.tts.modules.hifigan_modules import Generator


parser = argparse.ArgumentParser()
parser.add_argument('-bs', '--batchsize', type=int, default=1)
parser.add_argument('-c', '--checkpoint', type=str)
parser.add_argument('-v', '--vocodercheckpoint', type=str)
parser.add_argument('-m', '--manifest', type=str)

args = parser.parse_args()


class MeasureTime(list):
    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.cuda.synchronize()
        self.append(time.perf_counter() - self.t0)

    def __add__(self, other):
        assert len(self) == len(other)
        return MeasureTime(sum(ab) for ab in zip(self, other))


torch.backends.cudnn.benchmark = True

t2 = Tacotron2Model.restore_from(args.checkpoint)
t2 = t2.cuda().eval()
t2.decoder.gate_threshold = 0.25
labels = t2.cfg.labels

vocoder = Generator(
    resblock=1,
    upsample_rates=[8, 8, 4, 2],
    upsample_kernel_sizes=[16, 16, 4, 4],
    upsample_initial_channel=512,
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
)
nemo_gen_keys = [k for k in vocoder.state_dict().keys()]
adlr_gen_ckpt = torch.load(args.vocodercheckpoint)
adlr_gen_keys = adlr_gen_ckpt.keys()

new_nemo_ckpt = {nemo_key: adlr_gen_ckpt[adlr_key] for adlr_key, nemo_key in zip(adlr_gen_keys, nemo_gen_keys)}
vocoder.load_state_dict(new_nemo_ckpt)
vocoder = vocoder.cuda().eval()


dataset = _AudioTextDataset(
    args.manifest,
    parser=t2.parser,
    sample_rate=44100,
    bos_id=len(labels),
    eos_id=len(labels) + 1,
    pad_id=len(labels) + 2,
    max_utts=128,
)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=args.batchsize, collate_fn=dataset.collate_fn, num_workers=0, shuffle=False
)
measures = MeasureTime()
vocoder_measures = MeasureTime()
device = torch.device("cuda")
all_utterances = 0
all_samples = 0

batches = []
for b in dataloader:
    batches.append(b)


def infer_vocoder(model, spec: torch.Tensor):
    audio = model(spec).squeeze(1)
    return audio


print("warming up model")
for _ in tqdm.tqdm(range(10)):
    for batch in batches:
        _, _, text, text_length = batch
        text = text.to(device)
        text_length = text_length.to(device)
        with torch.no_grad():
            spectrogram, _ = t2.generate_spectrogram(tokens=text, token_len=text_length)
            audios = infer_vocoder(model=vocoder, spec=spectrogram)


print("testing model")
for _ in tqdm.tqdm(range(100)):
    for batch in batches:
        _, _, text, text_length = batch
        text = text.to(device)
        text_length = text_length.to(device)
        with torch.no_grad(), measures:
            spectrogram, lengths = t2.generate_spectrogram(tokens=text, token_len=text_length)
            audios = infer_vocoder(model=vocoder, spec=spec, with_bias_denoise=args.with_bias_denoise)
        # with vocoder_measures:

        all_utterances += text.size(0)
        all_samples += lengths.sum().item() * 512

# gm = np.sort(np.asarray(measures))
# rtf = all_samples / (all_utterances * gm.mean() * 44100)
# print(f"RTF: {rtf}")

spec_gen_m = np.sort(np.asarray(measures))
# vocoder_m = np.sort(np.asarray(vocoder_measures))
full_pipeline_m = spec_gen_m  # + vocoder_m
full_pipeline_rtf = all_samples / (all_utterances * full_pipeline_m.mean() * args.sample_rate)
print(f"RTF: {full_pipeline_rtf}")
