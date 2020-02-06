# Audio dataset and corresponding functions taken from Patter
# https://github.com/ryanleary/patter
# TODO: review, and copyright and fix/add comments
import kaldi_io
import os
import pandas as pd
import string
import random
import torch
from torch.utils.data import Dataset

from .manifest import ManifestBase, ManifestEN, TFManifest


def seq_collate_fn(batch, token_pad_value=0):
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
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=token_pad_value)
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
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        load_audio: Boolean flag indicate whether do or not load audio
    """

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
        bos_id=None,
        eos_id=None,
        logger=False,
        load_audio=True,
        manifest_class=ManifestEN,
    ):
        m_paths = manifest_filepath.split(',')
        self.manifest = manifest_class(
            m_paths,
            labels,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            blank_index=blank_index,
            unk_index=unk_index,
            normalize=normalize,
        )
        self.featurizer = featurizer
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.load_audio = load_audio
        if logger:
            logger.info(
                "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} "
                "hours.".format(self.manifest.duration / 3600, self.manifest.filtered_duration / 3600)
            )

    def __getitem__(self, index):
        sample = self.manifest[index]
        if self.load_audio:
            duration = sample['duration'] if 'duration' in sample else 0
            offset = sample['offset'] if 'offset' in sample else 0
            features = self.featurizer.process(
                sample['audio_filepath'], offset=offset, duration=duration, trim=self.trim
            )
            f, fl = features, torch.tensor(features.shape[0]).long()
            # f = f / (torch.max(torch.abs(f)) + 1e-5)
        else:
            f, fl = None, None

        t, tl = sample["tokens"], len(sample["tokens"])
        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __len__(self):
        return len(self.manifest)


class TFAudioDataset(Dataset):
    def __init__(
        self,
        manifest_filepath,
        labels,
        featurizer,
        max_duration=None,
        min_duration=None,
        max_utts=0,
        normalize=True,
        trim=False,
        bos_id=None,
        eos_id=None,
        logger=False,
        load_audio=True,
        tokenizer=None,
        mlm_prob=0,
        tokenizer_vocab_size=None,
    ):
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
        self.tokenizer = tokenizer
        self.manifest = TFManifest(
            m_paths,
            labels,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            normalize=normalize,
            tokenizer=tokenizer,
            logger=logger,
        )
        self.featurizer = featurizer
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.vocab_size = tokenizer_vocab_size
        self.load_audio = load_audio
        self.mlm_prob = mlm_prob
        if logger:
            logger.info(
                "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} "
                "hours.".format(self.manifest.duration / 3600, self.manifest.filtered_duration / 3600)
            )

    def __getitem__(self, index):
        sample = self.manifest[index]
        if self.load_audio:
            duration = sample['duration'] if 'duration' in sample else 0
            offset = sample['offset'] if 'offset' in sample else 0
            features = self.featurizer.process(
                sample['audio_filepath'], offset=offset, duration=duration, trim=self.trim
            )
            f, fl = features, torch.tensor(features.shape[0]).long()
            # f = f / (torch.max(torch.abs(f)) + 1e-5)
        else:
            f, fl = None, None

        t = sample["tokenizer_transcript"]
        tl = len(t) - 1

        if self.mlm_prob:
            # decoder_inputs should be decoder_outputs but with masks
            # tl are now the masks
            decoder_outputs = t
            decoder_inputs, tl = self.mask_ids(t)
        else:
            decoder_outputs = t[1:]
            decoder_inputs = t[:-1]

        char_transcript = sample["char_transcript"]
        char_transcript_l = len(sample["char_transcript"])

        return (
            f,
            fl,
            torch.tensor(decoder_inputs).long(),
            torch.tensor(decoder_outputs).long(),
            torch.tensor(tl).long(),
            torch.tensor(char_transcript).long(),
            torch.tensor(char_transcript_l).long(),
        )

    def mask_ids(self, ids):
        """
        Args:
          ids: list of token ids representing a chunk of text
        Returns:
          masked_ids: list of input tokens with some of the entries masked
            according to the following protocol from the original BERT paper:
            each token is masked with a probability of 15% and is replaced with
            1) the [MASK] token 80% of the time,
            2) random token 10% of the time,
            3) the same token 10% of the time.
          output_mask: list of binary variables which indicate what tokens has
            been masked (to calculate the loss function for these tokens only)
        """

        # Whole-word masking by default, as it gives better performance.
        cand_indexes = [[ids[0]]]
        for tid in ids[1:]:
            token = self.tokenizer.ids_to_tokens([tid])[0]
            is_suffix = token.startswith('\u2581')
            if is_suffix:
                # group together with its previous token to form a whole-word
                cand_indexes[-1].append(tid)
            else:
                cand_indexes.append([tid])

        masked_ids, output_mask = [], []
        mask_id = self.tokenizer.token_to_id("<mask>")

        for word_ids in cand_indexes:
            is_special = (word_ids[0] == self.bos_id) or (word_ids[0] == self.eos_id)
            if is_special or (random.random() > self.mlm_prob):
                output_mask.extend([0] * len(word_ids))
                masked_ids.extend(word_ids)
            else:
                output_mask.extend([1] * len(word_ids))
                p = random.random()
                # for 80%, replace with mask
                if p < 0.8:
                    masked_ids.extend([mask_id] * len(word_ids))
                # for 10%, replace by a random token
                elif p < 0.9:
                    for _ in word_ids:
                        # randomly select a valid word
                        random_word = random.randrange(self.vocab_size)
                        while random_word in (self.bos_id, self.eos_id):
                            random_word = random.randrange(self.vocab_size)
                        masked_ids.append(random_word)
                # for 10%, use same token
                else:
                    masked_ids.extend(word_ids)

        return masked_ids, output_mask

    def __len__(self):
        return len(self.manifest)


class MLMAudioDataset(Dataset):
    def __init__(
        self,
        manifest_filepath,
        labels,
        featurizer,
        max_duration=None,
        min_duration=None,
        max_utts=0,
        normalize=True,
        trim=False,
        bos_id=None,
        eos_id=None,
        logger=False,
        load_audio=True,
        tokenizer=None,
        mlm_prob=0,
        tokenizer_vocab_size=None,
        _eval=False,
    ):
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
        self.tokenizer = tokenizer
        self.manifest = TFManifest(
            m_paths,
            labels,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            normalize=normalize,
            tokenizer=tokenizer,
            logger=logger,
        )
        self.featurizer = featurizer
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.vocab_size = tokenizer_vocab_size
        self.load_audio = load_audio
        self.mlm_prob = mlm_prob
        self.eval = _eval
        if logger:
            logger.info(
                "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} "
                "hours.".format(self.manifest.duration / 3600, self.manifest.filtered_duration / 3600)
            )

    def __getitem__(self, index):
        sample = self.manifest[index]
        if self.load_audio:
            duration = sample['duration'] if 'duration' in sample else 0
            offset = sample['offset'] if 'offset' in sample else 0
            features = self.featurizer.process(
                sample['audio_filepath'], offset=offset, duration=duration, trim=self.trim
            )
            f, fl = features, torch.tensor(features.shape[0]).long()
            # f = f / (torch.max(torch.abs(f)) + 1e-5)
        else:
            f, fl = None, None

        t = sample["tokenizer_transcript"]
        tl = len(t) - 1

        if self.eval:
            decoder_outputs = t
            output_mask = [1] * len(decoder_outputs)
            output_mask[1] = 0
            mask_id = self.tokenizer.token_to_id("<mask>")
            decoder_inputs = [mask_id] * len(decoder_outputs)
            decoder_inputs[0] = self.bos_id
        elif self.mlm_prob:
            # decoder_inputs should be decoder_outputs but with masks
            # tl are now the masks
            decoder_outputs = t
            decoder_inputs, output_mask = self.mask_ids(t)
        else:
            decoder_outputs = t[1:]
            decoder_inputs = t[:-1]

        char_transcript = sample["char_transcript"]
        char_transcript_l = len(sample["char_transcript"])

        return (
            f,
            fl,
            torch.tensor(decoder_inputs).long(),
            torch.tensor(decoder_outputs).long(),
            torch.tensor(tl).long(),
            torch.tensor(output_mask).long(),
            torch.tensor(char_transcript).long(),
            torch.tensor(char_transcript_l).long(),
        )

    def mask_ids(self, ids):
        """
        Args:
          ids: list of token ids representing a chunk of text
        Returns:
          masked_ids: list of input tokens with some of the entries masked
            according to the following protocol from the original BERT paper:
            each token is masked with a probability of 15% and is replaced with
            1) the [MASK] token 80% of the time,
            2) random token 10% of the time,
            3) the same token 10% of the time.
          output_mask: list of binary variables which indicate what tokens has
            been masked (to calculate the loss function for these tokens only)
        """

        # Whole-word masking by default, as it gives better performance.
        cand_indexes = [[ids[0]]]
        for tid in ids[1:]:
            token = self.tokenizer.ids_to_tokens([tid])[0]
            is_suffix = token.startswith('\u2581')
            if tid in [self.bos_id, self.eos_id]:
                cand_indexes.append([tid])
            elif not is_suffix:
                # group together with its previous token to form a whole-word
                cand_indexes[-1].append(tid)
            else:
                cand_indexes.append([tid])

        masked_ids, output_mask = [], []
        mask_id = self.tokenizer.token_to_id("<mask>")

        for word_ids in cand_indexes:
            is_special = (word_ids[0] == self.bos_id) or (word_ids[0] == self.eos_id)
            if is_special or (random.random() > self.mlm_prob):
                output_mask.extend([0] * len(word_ids))
                masked_ids.extend(word_ids)
            else:
                output_mask.extend([1] * len(word_ids))
                p = random.random()
                # for 80%, replace with mask
                if p < 0.8:
                    masked_ids.extend([mask_id] * len(word_ids))
                # for 10%, replace by a random token
                elif p < 0.9:
                    for _ in word_ids:
                        # randomly select a valid word
                        random_word = random.randrange(self.vocab_size)
                        while random_word in (self.bos_id, self.eos_id):
                            random_word = random.randrange(self.vocab_size)
                        masked_ids.append(random_word)
                # for 10%, use same token
                else:
                    masked_ids.extend(word_ids)

        return masked_ids, output_mask

    def __len__(self):
        return len(self.manifest)


def mlmaudio_seq_collate_fn(batch):
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

        audio_signal = torch.zeros(batch_size, max_audio_len, dtype=torch.float)
        audio_lengths = []
        for i, s in enumerate(batch):
            audio_signal[i].narrow(0, 0, s[0].size(0)).copy_(s[0])
            audio_lengths.append(s[1])
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

    max_transcript_len = find_max_len(batch, 2)
    max_char_transcript_len = find_max_len(batch, 6)

    decoder_in = torch.zeros(batch_size, max_transcript_len, dtype=torch.long)
    decoder_out = torch.zeros(batch_size, max_transcript_len, dtype=torch.long)
    output_mask = torch.zeros(batch_size, max_transcript_len, dtype=torch.long)
    transcript_lengths = []
    char_transcript = torch.zeros(batch_size, max_char_transcript_len, dtype=torch.long)
    char_transcript_lengths = []
    for i, s in enumerate(batch):
        decoder_in[i].narrow(0, 0, s[2].size(0)).copy_(s[2])
        decoder_out[i].narrow(0, 0, s[3].size(0)).copy_(s[3])
        transcript_lengths.append(s[4])
        output_mask[i].narrow(0, 0, s[5].size(0)).copy_(s[5])
        char_transcript[i].narrow(0, 0, s[6].size(0)).copy_(s[6])
        char_transcript_lengths.append(s[7])
    transcript_lengths = torch.tensor(transcript_lengths, dtype=torch.long)
    char_transcript_lengths = torch.tensor(char_transcript_lengths, dtype=torch.long)

    return (
        audio_signal,
        audio_lengths,
        decoder_in,
        decoder_out,
        transcript_lengths,
        output_mask,
        char_transcript,
        char_transcript_lengths,
    )


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

        audio_signal = torch.zeros(batch_size, max_audio_len, dtype=torch.float)
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
    char_transcript = torch.zeros(batch_size, max_char_transcript_len, dtype=torch.long)
    char_transcript_lengths = []
    for i, s in enumerate(batch):
        decoder_in[i].narrow(0, 0, s[2].size(0)).copy_(s[2])
        decoder_out[i].narrow(0, 0, s[3].size(0)).copy_(s[3])
        transcript_lengths.append(s[4])
        char_transcript[i].narrow(0, 0, s[5].size(0)).copy_(s[5])
        char_transcript_lengths.append(s[6])
    transcript_lengths = torch.tensor(transcript_lengths, dtype=torch.long)
    char_transcript_lengths = torch.tensor(char_transcript_lengths, dtype=torch.long)

    return (
        audio_signal,
        audio_lengths,
        decoder_in,
        decoder_out,
        transcript_lengths,
        char_transcript,
        char_transcript_lengths,
    )


class KaldiFeatureDataset(Dataset):
    """
    Dataset that provides basic Kaldi-compatible dataset loading. Assumes that
    the files `feats.scp`, `text`, and (optionally) `utt2dur` exist, as well
    as the .ark files that `feats.scp` points to.

    Args:
        kaldi_dir: Path to directory containing the aforementioned files.
        labels: All possible characters to map to.
        min_duration: If audio is shorter than this length, drop it. Only
            available if the `utt2dur` file exists.
        max_duration: If audio is longer than this length, drop it. Only
            available if the `utt2dur` file exists.
        max_utts: Limits the number of utterances.
        unk_index: unk_character index, default = -1
        blank_index: blank character index, default = -1
        normalize: whether to normalize transcript text. Defaults to True.
        eos_id: Id of end of sequence symbol to append if not None.
    """

    def __init__(
        self,
        kaldi_dir,
        labels,
        min_duration=None,
        max_duration=None,
        max_utts=0,
        unk_index=-1,
        blank_index=-1,
        normalize=True,
        eos_id=None,
        logger=None,
    ):
        self.eos_id = eos_id
        self.unk_index = unk_index
        self.blank_index = blank_index
        self.labels_map = {label: i for i, label in enumerate(labels)}

        data = []
        duration = 0.0
        filtered_duration = 0.0

        # Read Kaldi features (MFCC, PLP) using feats.scp
        feats_path = os.path.join(kaldi_dir, 'feats.scp')
        id2feats = {utt_id: torch.from_numpy(feats) for utt_id, feats in kaldi_io.read_mat_scp(feats_path)}

        # Get durations, if utt2dur exists
        utt2dur_path = os.path.join(kaldi_dir, 'utt2dur')
        id2dur = {}
        if os.path.exists(utt2dur_path):
            with open(utt2dur_path, 'r') as f:
                for line in f:
                    utt_id, dur = line.split()
                    id2dur[utt_id] = float(dur)
        elif max_duration or min_duration:
            raise ValueError(
                f"KaldiFeatureDataset max_duration or min_duration is set but"
                f" utt2dur file not found in {kaldi_dir}."
            )
        elif logger:
            logger.info(
                f"Did not find utt2dur when loading data from " f"{kaldi_dir}. Skipping dataset duration calculations."
            )

        # Match transcripts to features
        text_path = os.path.join(kaldi_dir, 'text')
        with open(text_path, 'r') as f:
            for line in f:
                split_idx = line.find(' ')
                utt_id = line[:split_idx]

                audio_features = id2feats.get(utt_id)

                if audio_features is not None:

                    text = line[split_idx:].strip()
                    if normalize:
                        text = ManifestEN.normalize_text(text, labels)
                    dur = id2dur[utt_id] if id2dur else None

                    # Filter by duration if specified & utt2dur exists
                    if min_duration and dur < min_duration:
                        filtered_duration += dur
                        continue
                    if max_duration and dur > max_duration:
                        filtered_duration += dur
                        continue

                    sample = {
                        'utt_id': utt_id,
                        'text': text,
                        'tokens': ManifestBase.tokenize_transcript(
                            text, self.labels_map, self.unk_index, self.blank_index
                        ),
                        'audio': audio_features.t(),
                        'duration': dur,
                    }

                    data.append(sample)
                    duration += dur

                    if max_utts > 0 and len(data) >= max_utts:
                        print(f"Stop parsing due to max_utts ({max_utts})")
                        break

        if logger and id2dur:
            # utt2dur durations are in seconds
            logger.info(
                f"Dataset loaded with {duration/60 : .2f} hours. " f"Filtered {filtered_duration/60 : .2f} hours."
            )

        self.data = data

    def __getitem__(self, index):
        sample = self.data[index]
        f = sample['audio']
        fl = torch.tensor(f.shape[1]).long()
        t, tl = sample['tokens'], len(sample['tokens'])

        if self.eos_id is not None:
            t.append(self.eos_id)
            tl += 1

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __len__(self):
        return len(self.data)


class TranscriptDataset(Dataset):
    """A dataset class that reads and returns the text of a file.

    Args:
        path: (str) Path to file with newline separate strings of text
        labels (list): List of string labels to use when to str2int translation
        eos_id (int): Label position of end of string symbol
    """

    def __init__(self, path, labels, bos_id=None, eos_id=None, lowercase=True):
        _, ext = os.path.splitext(path)
        if ext == '.csv':
            texts = pd.read_csv(path)['transcript'].tolist()
        elif ext == '.json':
            # Assume it is in ASR manifest form
            texts = []
            for item in ManifestBase.json_item_gen([path]):
                if 'text' in item:
                    text = item['text']
                elif 'text_filepath' in item:
                    text = ManifestBase.load_transcript(item['text_filepath'])
                else:
                    raise ValueError(f"{self} obtained an invalid manifest.")
                texts.append(text)
        else:
            with open(path, 'r') as f:
                texts = f.readlines()
        if lowercase:
            texts = [l.strip().lower() for l in texts if len(l)]
        self.texts = texts

        self.char2num = {c: i for i, c in enumerate(labels)}
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        tokenized_text = ManifestBase.tokenize_transcript(self.texts[item], self.char2num, -1, -1)
        if self.bos_id:
            tokenized_text = [self.bos_id] + tokenized_text
        if self.eos_id:
            tokenized_text = tokenized_text + [self.eos_id]
        return (torch.tensor(tokenized_text, dtype=torch.long), torch.tensor(len(tokenized_text), dtype=torch.long))
