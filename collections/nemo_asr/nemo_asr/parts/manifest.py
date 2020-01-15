# Taken straight from Patter https://github.com/ryanleary/patter
# TODO: review, and copyright and fix/add comments
import json
import nemo
import string

import nemo_nlp
from .cleaners import clean_text


class ManifestBase():
    def __init__(self,
                 manifest_paths,
                 labels,
                 max_duration=None,
                 min_duration=None,
                 sort_by_duration=False,
                 max_utts=0,
                 blank_index=-1,
                 unk_index=-1,
                 normalize=True):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sort_by_duration = sort_by_duration
        self.max_utts = max_utts
        self.blank_index = blank_index
        self.unk_index = unk_index
        self.normalize = normalize
        self.labels_map = {label: i for i, label in enumerate(labels)}

        data = []
        duration = 0.0
        filtered_duration = 0.0

        for item in self.json_item_gen(manifest_paths):
            if min_duration and item['duration'] < min_duration:
                filtered_duration += item['duration']
                continue
            if max_duration and item['duration'] > max_duration:
                filtered_duration += item['duration']
                continue

            # load and normalize transcript text, i.e. `text`
            text = ""
            if 'text' in item:
                text = item['text']
            elif 'text_filepath' in item:
                text = self.load_transcript(item['text_filepath'])
            else:
                filtered_duration += item['duration']
                continue
            if normalize:
                text = self.normalize_text(text, labels)
            if not isinstance(text, str):
                nemo.logging.warning(
                    "WARNING: Got transcript: {}. It is not a "
                    "string. Dropping data point".format(text)
                )
                filtered_duration += item['duration']
                continue
            # item['text'] = text

            # tokenize transcript text
            item["tokens"] = self.tokenize_transcript(
                    text, self.labels_map, self.unk_index, self.blank_index)

            # support files using audio_filename
            if 'audio_filename' in item and 'audio_filepath' not in item:
                nemo.logging.warning(
                    "Malformed manifest: The key audio_filepath was not "
                    "found in the manifest. Using audio_filename instead."
                )
                item['audio_filepath'] = item['audio_filename']

            data.append(item)
            duration += item['duration']

            if max_utts > 0 and len(data) >= max_utts:
                nemo.logging.info(
                    'Stop parsing due to max_utts ({})'.format(max_utts))
                break

        if sort_by_duration:
            data = sorted(data, key=lambda x: x['duration'])
        self._data = data
        self._size = len(data)
        self._duration = duration
        self._filtered_duration = filtered_duration

    @staticmethod
    def normalize_text(text, labels, logger=None):
        """for the base class remove surrounding whitespace only"""
        return text.strip()

    @staticmethod
    def tokenize_transcript(transcript, labels_map, unk_index, blank_index):
        """tokenize transcript to convert words/characters to indices"""
        # allow for special labels such as "<NOISE>"
        special_labels = set([l for l in labels_map.keys() if len(l) > 1])
        tokens = []
        # split by word to find special tokens
        for i, word in enumerate(transcript.split(" ")):
            if i > 0:
                tokens.append(labels_map.get(" ", unk_index))
            if word in special_labels:
                tokens.append(labels_map.get(word))
                continue
            # split by character to get the rest of the tokens
            for char in word:
                tokens.append(labels_map.get(char, unk_index))
        # if unk_index == blank_index, OOV tokens are removed from transcript
        tokens = [x for x in tokens if x != blank_index]
        return tokens

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self._data)

    @staticmethod
    def json_item_gen(manifest_paths):
        for manifest_path in manifest_paths:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    yield json.loads(line)

    @staticmethod
    def load_transcript(transcript_path):
        with open(transcript_path, 'r', encoding="utf-8") as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        return transcript

    @property
    def duration(self):
        return self._duration

    @property
    def filtered_duration(self):
        return self._filtered_duration

    @property
    def data(self):
        return list(self._data)


class ManifestEN(ManifestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def normalize_text(text, labels):
        # Punctuation to remove
        punctuation = string.punctuation
        # Define punctuation that will be handled by text cleaner
        punctuation_to_replace = {
            "+": "plus",
            "&": "and",
            "%": "percent"
        }
        for char in punctuation_to_replace:
            punctuation = punctuation.replace(char, "")
        # We might also want to consider:
        # @ -> at
        # -> number, pound, hashtag
        # ~ -> tilde
        # _ -> underscore

        # If a punctuation symbol is inside our vocab, we do not remove
        # from text
        for l in labels:
            punctuation = punctuation.replace(l, "")

        # Turn all other punctuation to whitespace
        table = str.maketrans(punctuation, " " * len(punctuation))

        try:
            text = clean_text(text, table, punctuation_to_replace)
        except BaseException:
            nemo.logging.warning("WARNING: Normalizing {} failed".format(text))
            return None

        return text


class TFManifest(object):
    def __init__(self, manifest_paths, labels, max_duration=None,
                 min_duration=None, sort_by_duration=False, max_utts=0,
                 normalize=True, tokenizer=None, logger=None):
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.blank_index = -1
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, nemo_nlp.YouTokenToMeTokenizer):
            self.bos_id = self.tokenizer.bos_id()
            self.eos_id = self.tokenizer.eos_id()
        elif isinstance(self.tokenizer, nemo_nlp.SentencePieceTokenizer):
            self.bos_id = self.tokenizer.token_to_id("<s>")
            self.eos_id = self.tokenizer.token_to_id("</s>")
        self.max_len = 0
        ids = []
        duration = 0.0
        filtered_duration = 0.0

        for manifest_path in manifest_paths:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    data = json.loads(line)
                    if min_duration is not None and data['duration'] \
                            < min_duration:
                        filtered_duration += data['duration']
                        continue
                    if max_duration is not None and data['duration'] \
                            > max_duration:
                        filtered_duration += data['duration']
                        continue

                    # Prune and normalize according to transcript
                    transcript_text = data[
                        'text'] if "text" in data else self.load_transcript(
                        data['text_filepath'])
                    if normalize:
                        transcript_text = ManifestEN.normalize_text(
                            transcript_text,
                            labels=labels)
                    if not isinstance(transcript_text, str):
                        print(
                            "WARNING: Got transcript: {}. It is not a "
                            "string. Dropping data point".format(
                                transcript_text))
                        filtered_duration += data['duration']
                        continue
                    data["tokenizer_transcript"] = self.tokenize(
                        transcript_text)
                    data["char_transcript"] = self.parse_transcript(
                        transcript_text)
                    # data['transcript_text'] = transcript_text
                    # print(transcript_text)

                    # support files using audio_filename
                    if 'audio_filename' in data:
                        data['audio_filepath'] = data['audio_filename']
                    ids.append(data)
                    duration += data['duration']

                    if max_utts > 0 and len(ids) >= max_utts:
                        print(
                            'Stopping parsing %s as max_utts=%d' % (
                                manifest_path, max_utts))
                        break

        if logger:
            logger.info(f"Max t_len: {self.max_len}")
        else:
            print(f"Max t_len: {self.max_len}")
        if sort_by_duration:
            ids = sorted(ids, key=lambda x: x['duration'])
        self._data = ids
        self._size = len(ids)
        self._duration = duration
        self._filtered_duration = filtered_duration

    def load_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding="utf-8") as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        return transcript

    def parse_transcript(self, transcript):
        chars = [self.labels_map.get(x, self.blank_index)
                 for x in list(transcript)]
        transcript = list(filter(lambda x: x != self.blank_index, chars))
        return transcript

    def tokenize(self, transcript):
        t = self.tokenizer.text_to_ids(transcript)
        t = [self.bos_id] + t + [self.eos_id]
        if len(t) > self.max_len:
            self.max_len = len(t)
        return t

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self._data)

    @property
    def duration(self):
        return self._duration

    @property
    def filtered_duration(self):
        return self._filtered_duration

    @property
    def data(self):
        return list(self._data)
