# Audio dataset and corresponding functions taken from Patter
# https://github.com/ryanleary/patter
# TODO: review, and copyright and fix/add comments
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from .manifest import ManifestEN, TFManifest


def seq_collate_fn(batch):
    """collate batch of audio sig, audio len, tokens, tokens len

    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).

    """
    _, audio_lengths, _, tokens_lengths = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens = [], []
    for sig, sig_len, tokens_i, tokens_i_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


def audio_seq_collate_fn(batch):
    """
    Collate a batch (iterable of (sample tensor, label tensor) tuples) into
    properly shaped data tensors
    :param batch:
    :return: inputs (batch_size, num_features, seq_length), targets,
    input_lengths, target_sizes
    """
    # sort batch by descending sequence length (for packed sequences later)
    batch.sort(key=lambda x: x[0].size(0), reverse=True)

    # init tensors we need to return
    inputs = []
    input_lengths = []
    target_sizes = []
    targets = []
    metadata = []

    # iterate over minibatch to fill in tensors appropriately
    for i, sample in enumerate(batch):
        input_lengths.append(sample[0].size(0))
        inputs.append(sample[0])
        target_sizes.append(len(sample[1]))
        targets.extend(sample[1])
        metadata.append(sample[2])
    targets = torch.tensor(targets, dtype=torch.long)
    inputs = torch.stack(inputs)
    input_lengths = torch.stack(input_lengths)
    target_sizes = torch.stack(target_sizes)

    return inputs, targets, input_lengths, target_sizes, metadata


class AudioDataset(Dataset):
    def __init__(
            self,
            manifest_filepath,
            labels,
            featurizer,
            max_duration=None,
            min_duration=None,
            max_utts=0,
            blank_index=-1,
            unk_index=-1,
            normalize=True,
            trim=False,
            eos_id=None,
            logger=False,
            load_audio=True,
            manifest_class=ManifestEN):
        """
        Dataset that loads tensors via a json file containing paths to audio
        files, transcripts, and durations (in seconds). Each new line is a
        different sample. Example below:

        {"audio_filepath": "/path/to/audio.wav", "text_filepath":
        "/path/to/audio.txt", "duration": 23.147}
        ...
        {"audio_filepath": "/path/to/audio.wav", "text": "the
        transcription", offset": 301.75, "duration": 0.82, "utt":
        "utterance_id", "ctm_utt": "en_4156", "side": "A"}

        Args:
            manifest_filepath: Path to manifest json as described above. Can
                be comma-separated paths.
            labels: String containing all the possible characters to map to
            featurizer: Initialized featurizer class that converts paths of
                audio to feature tensors
            max_duration: If audio exceeds this length, do not include in
                dataset
            min_duration: If audio is less than this length, do not include
                in dataset
            max_utts: Limit number of utterances
            blank_index: blank character index, default = -1
            unk_index: unk_character index, default = -1
            normalize: whether to normalize transcript text (default): True
            eos_id: Id of end of sequence symbol to append if not None
            load_audio: Boolean flag indicate whether do or not load audio
        """
        m_paths = manifest_filepath.split(',')
        self.manifest = manifest_class(m_paths, labels,
                                       max_duration=max_duration,
                                       min_duration=min_duration,
                                       max_utts=max_utts,
                                       blank_index=blank_index,
                                       unk_index=unk_index,
                                       normalize=normalize)
        self.featurizer = featurizer
        self.trim = trim
        self.eos_id = eos_id
        self.load_audio = load_audio
        if logger:
            logger.info(
                "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} "
                "hours.".format(
                    self.manifest.duration / 3600,
                    self.manifest.filtered_duration / 3600))

    def __getitem__(self, index):
        sample = self.manifest[index]
        if self.load_audio:
            duration = sample['duration'] if 'duration' in sample else 0
            offset = sample['offset'] if 'offset' in sample else 0
            features = self.featurizer.process(sample['audio_filepath'],
                                               offset=offset,
                                               duration=duration,
                                               trim=self.trim)
            f, fl = features, torch.tensor(features.shape[0]).long()
            # f = f / (torch.max(torch.abs(f)) + 1e-5)
        else:
            f, fl = None, None

        t, tl = sample["tokens"], len(sample["tokens"])
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return \
            f, fl, \
            torch.tensor(t).long(), torch.tensor(tl).long()

    def __len__(self):
        return len(self.manifest)


class TranscriptDataset(Dataset):
    """A dataset class that reads and returns the text of a file.

    Args:
        path: (str) Path to file with newline separate strings of text
        labels (list): List of string labels to use when to str2int translation
        eos_id (int): Label position of end of string symbol
    """
    def __init__(self, path, labels, eos_id):
        _, ext = os.path.splitext(path)
        if ext == '.csv':
            texts = pd.read_csv(path)['transcript'].tolist()
        else:
            with open(path, 'r') as f:
                texts = f.readlines()
        texts = [l.strip().lower() for l in texts if len(l)]
        self.texts = texts

        self.char2num = {c: i for i, c in enumerate(labels)}
        self.eos_id = eos_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        char2num = self.char2num
        return torch.tensor(
            [char2num[c] for c in self.texts[item]
             if c in char2num] + [self.eos_id],
            dtype=torch.long
        )


class TFAudioDataset(Dataset):
    def __init__(self, manifest_filepath, labels, featurizer,
                 max_duration=None,
                 min_duration=None, max_utts=0, normalize=True,
                 trim=False, eos_id=None, logger=False, load_audio=True,
                 tokenizer=None):
        """
        Dataset that loads tensors via a json file containing paths to audio
        files, transcripts, and durations
        (in seconds). Each new line is a different sample. Example below:

        {"audio_filepath": "/path/to/audio.wav", "text_filepath":
        "/path/to/audio.txt", "duration": 23.147}
        ...
        {"audio_filepath": "/path/to/audio.wav", "text": "the
        transcription", offset": 301.75, "duration": 0.82, "utt":
        "utterance_id",
        "ctm_utt": "en_4156", "side": "A"}

        Args:
            manifest_filepath: Path to manifest json as described above. Can
            be coma-separated paths.
            labels: String containing all the possible characters to map to
            featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
            max_duration: If audio exceeds this length, do not include in
            dataset
            min_duration: If audio is less than this length, do not include
            in dataset
            max_utts: Limit number of utterances
            normalize: whether to normalize transcript text (default): True
            eos_id: Id of end of sequence symbol to append if not None
            load_audio: Boolean flag indicate whether do or not load audio
        """
        m_paths = manifest_filepath.split(',')
        self.manifest = TFManifest(m_paths, labels,
                                   max_duration=max_duration,
                                   min_duration=min_duration,
                                   max_utts=max_utts,
                                   normalize=normalize,
                                   tokenizer=tokenizer,
                                   logger=logger)
        self.featurizer = featurizer
        self.trim = trim
        self.eos_id = eos_id
        self.load_audio = load_audio
        if logger:
            logger.info(
                "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} "
                "hours.".format(
                    self.manifest.duration / 3600,
                    self.manifest.filtered_duration / 3600))

    def __getitem__(self, index):
        sample = self.manifest[index]
        if self.load_audio:
            duration = sample['duration'] if 'duration' in sample else 0
            offset = sample['offset'] if 'offset' in sample else 0
            features = self.featurizer.process(sample['audio_filepath'],
                                               offset=offset,
                                               duration=duration,
                                               trim=self.trim)
            f, fl = features, torch.tensor(features.shape[0]).long()
            # f = f / (torch.max(torch.abs(f)) + 1e-5)
        else:
            f, fl = None, None

        t = sample["tokenizer_transcript"]
        tl = len(t) - 1
        decoder_outputs = t[1:]
        decoder_inputs = t[:-1]

        char_transcript = sample["char_transcript"]
        char_transcript_l = len(sample["char_transcript"])

        return \
            f, fl, \
            torch.tensor(decoder_inputs).long(),\
            torch.tensor(decoder_outputs).long(),\
            torch.tensor(tl).long(),\
            torch.tensor(char_transcript).long(),\
            torch.tensor(char_transcript_l).long(),

    def __len__(self):
        return len(self.manifest)


def tfaudio_seq_collate_fn(batch):
    def find_max_len(seq, index):
        max_len = -1
        for item in seq:
            if item[index].size(0) > max_len:
                max_len = item[index].size(0)
        return max_len

    batch_size = len(batch)

    audio_signal, audio_lengths = None, None
    if batch[0][0] is not None:
        max_audio_len = find_max_len(batch, 0)

        audio_signal = torch.zeros(batch_size, max_audio_len,
                                   dtype=torch.float)
        audio_lengths = []
        for i, s in enumerate(batch):
            audio_signal[i].narrow(0, 0, s[0].size(0)).copy_(s[0])
            audio_lengths.append(s[1])
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

    max_transcript_len = find_max_len(batch, 2)
    max_char_transcript_len = find_max_len(batch, 5)

    decoder_in = torch.zeros(batch_size, max_transcript_len, dtype=torch.long)
    decoder_out = torch.zeros(batch_size, max_transcript_len, dtype=torch.long)
    transcript_lengths = []
    char_transcript = torch.zeros(
        batch_size, max_char_transcript_len, dtype=torch.long)
    char_transcript_lengths = []
    for i, s in enumerate(batch):
        decoder_in[i].narrow(0, 0, s[2].size(0)).copy_(s[2])
        decoder_out[i].narrow(0, 0, s[3].size(0)).copy_(s[3])
        transcript_lengths.append(s[4])
        char_transcript[i].narrow(0, 0, s[5].size(0)).copy_(s[5])
        char_transcript_lengths.append(s[6])
    transcript_lengths = torch.tensor(transcript_lengths, dtype=torch.long)
    char_transcript_lengths = torch.tensor(
        char_transcript_lengths, dtype=torch.long)

    return audio_signal, audio_lengths, decoder_in, decoder_out,\
        transcript_lengths, char_transcript, char_transcript_lengths
