import time
import argparse

import numpy as np
import torch
import tqdm
from scipy.io.wavfile import write

from nemo.collections.asr.data.audio_to_text import AudioToCharDataset, _AudioTextDataset
from nemo.collections.tts.models import HifiGanModel, Tacotron2Model


parser = argparse.ArgumentParser()
parser.add_argument('-bs', '--batchsize', type=int, default=1)
parser.add_argument('-c', '--checkpoint', type=str)
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


dataset = _AudioTextDataset(
    args.manifest,
    parser=t2.parser,
    sample_rate=44100,
    bos_id=len(labels),
    eos_id=len(labels) + 1,
    pad_id=len(labels) + 2,
)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=args.batchsize, collate_fn=dataset.collate_fn, num_workers=0, shuffle=False
)
measures = MeasureTime()
device = torch.device("cuda")
all_utterances = 0
all_samples = 0

batches = []
for b in dataloader:
    batches.append(b)

print("warming up model")
for _ in tqdm.tqdm(range(1)):
    for i, batch in tqdm.tqdm(enumerate(batches)):
        _, _, text, text_length = batch
        text = text.to(device)
        with torch.no_grad():
            spectrogram, _ = t2.generate_spectrogram(tokens=text)


print("testing model")
for _ in tqdm.tqdm(range(2)):
    for batch in tqdm.tqdm(batches):
        _, _, text, text_length = batch
        text = text.to(device)
        with torch.no_grad(), measures:
            spectrogram, lengths = t2.generate_spectrogram(tokens=text)
        all_utterances += 1
        all_samples += lengths.sum().item() * 512

gm = np.sort(np.asarray(measures))
rtf = all_samples / (all_utterances * gm.mean() * 44100)
print(f"RTF: {rtf}")
