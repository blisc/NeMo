from itertools import chain
from pprint import pformat

import numpy as np
import torch

from nemo.backends.pytorch.common.metrics import char_lm_metrics
from nemo_asr.metrics import word_error_rate
from ..helpers import __gather_predictions, __gather_transcripts

ENG_MWN = 5.3


def process_evaluation_batch(tensors, global_vars, labels, specials, tb_writer=None, write_attn=True):
    loss, log_probs = ([],) * 2
    transcripts, transcript_texts = ([],) * 2
    predictions, prediction_texts = ([],) * 2
    attention_weights = []
    for k, v in tensors.items():
        if 'loss' in k:
            loss = v
        elif 'log_probs' in k:
            log_probs = v
        elif ('transcripts' in k) or ('texts' in k):
            transcripts = v
            transcript_texts = __decode(v, labels, specials)
        elif 'predictions' in k:
            # predictions = v
            prediction_texts = __decode(v, labels, specials)
        elif 'attention_weights' in k:
            attention_weights = v

    global_vars.setdefault('loss', [])
    global_vars['loss'].extend(loss)
    bpc, ppl = char_lm_metrics(log_probs, transcripts, transcript_texts, specials['pad_id'])
    global_vars.setdefault('bpc', [])
    global_vars['bpc'].extend(bpc)
    global_vars.setdefault('ppl', [])
    global_vars['ppl'].extend(ppl)
    global_vars.setdefault('transcript_texts', [])
    global_vars['transcript_texts'].extend(transcript_texts)
    global_vars.setdefault('prediction_texts', [])
    global_vars['prediction_texts'].extend(prediction_texts)

    # TODO: Add step number?
    if tb_writer is not None and len(attention_weights) and write_attn:
        sample_len = len(prediction_texts[0][0])
        if sample_len > 0:
            attention_weights = attention_weights[0][0, :sample_len, :]
            tb_writer.add_image('image/eval_attention_weights', attention_weights, dataformats='HW')


def process_evaluation_epoch(
    global_vars, metrics=('loss', 'bpc', 'ppl'), calc_wer=False, logger=None, mode='eval', tag='none'
):
    tag = '_'.join(tag.lower().strip().split())
    return_dict = {}
    for metric in metrics:
        value = torch.mean(torch.stack(global_vars[metric])).item()
        return_dict[f'metric/{mode}_{metric}_{tag}'] = value

    # TODO: Delete?
    bpc = return_dict[f'metric/{mode}_bpc_{tag}']
    return_dict[f'metric/{mode}_ppl_{tag}'] = 2 ** (bpc * ENG_MWN)

    if calc_wer:
        transcript_texts = list(chain(*global_vars['transcript_texts']))
        prediction_texts = list(chain(*global_vars['prediction_texts']))

        if logger:
            logger.info(f'Ten examples (transcripts and predictions)')
            logger.info(transcript_texts[:10])
            logger.info(prediction_texts[:10])

        wer = word_error_rate(hypotheses=prediction_texts, references=transcript_texts)
        return_dict[f'metric/{mode}_wer_{tag}'] = wer

    if logger:
        logger.info(pformat(return_dict))

    return return_dict


def process_evaluation_batch_xf(tensors, global_vars, tokenizer, labels):
    transcript_texts = []
    prediction_texts = []
    global_vars.setdefault('seqloss', [])
    global_vars.setdefault('transcript_texts', [])
    global_vars.setdefault('prediction_texts', [])
    tensor_list = list(tensors.values())[1:]  # Ignore IS_FROM_DIST_EVAL
    global_vars['seqloss'].extend(tensor_list[0])

    for t in tensor_list[1]:
        t = t.cpu().numpy().tolist()
        for sentence in t:
            transcript_texts.append(tokenizer.ids_to_text(sentence))
    for t in tensor_list[2]:
        t = t.cpu().numpy().tolist()
        for sentence in t:
            prediction_texts.append(tokenizer.ids_to_text(sentence))
    global_vars['transcript_texts'].extend(transcript_texts)
    global_vars['prediction_texts'].extend(prediction_texts)

    if len(tensor_list) > 3:
        ctc_transcript_texts = []
        ctc_prediction_texts = []
        global_vars.setdefault('ctcloss', [])
        global_vars.setdefault('ctc_transcript_texts', [])
        global_vars.setdefault('ctc_prediction_texts', [])
        global_vars['ctcloss'].extend(tensor_list[3])
        global_vars['ctc_prediction_texts'] += __gather_predictions(tensor_list[6], labels=labels)
        global_vars['ctc_transcript_texts'] += __gather_transcripts(tensor_list[4], tensor_list[5], labels=labels)


def process_evaluation_epoch_xf(global_vars, calc_wer=True, logger=None, mode='eval', tag='none'):
    tag = '_'.join(tag.lower().strip().split())
    return_dict = {}
    ctc = False
    metrics = ["seqloss"]
    if "ctcloss" in global_vars:
        ctc = True
        metrics.append("ctcloss")

    for metric in metrics:
        value = torch.mean(torch.stack(global_vars[metric])).item()
        return_dict[f'metric/{mode}_{metric}_{tag}'] = value

    if ctc:
        return_dict[f'metric/{mode}_loss_{tag}'] = (
            return_dict[f'metric/{mode}_seqloss_{tag}'] + return_dict[f'metric/{mode}_ctcloss_{tag}']
        )
    else:
        return_dict[f'metric/{mode}_loss_{tag}'] = return_dict[f'metric/{mode}_seqloss_{tag}']

    if calc_wer:
        transcripts = global_vars['transcript_texts']
        predictions = global_vars['prediction_texts']

        wer = word_error_rate(hypotheses=predictions, references=transcripts)
        return_dict[f'metric/{mode}_seq_wer_{tag}'] = wer

        if logger:
            choices = np.random.randint(len(transcripts), size=10)
            pstring = "Ten examples (transcripts and predictions)\n"
            pexamples = [f"{i}:\nseq t:{transcripts[c]}\nseq p:{predictions[c]}\n" for i, c in enumerate(choices)]

        if ctc:
            transcripts = global_vars['ctc_transcript_texts']
            predictions = global_vars['ctc_prediction_texts']

            wer = word_error_rate(hypotheses=predictions, references=transcripts)
            return_dict[f'metric/{mode}_ctc_wer_{tag}'] = wer

            if logger:
                pexamples = [
                    pexamples[i] + f'ctc t:{transcripts[c]}\nctc p:{predictions[c]}\n' for i, c in enumerate(choices)
                ]

        if logger:
            logger.info(pstring + "".join(pexamples).strip())

    if logger:
        logger.info("\n" + pformat(return_dict))

    return return_dict


def __decode(tensors_list, labels, specials):
    labels_map = dict([(i, labels[i]) for i in range(len(labels)) if i not in set(specials.values())])
    results = []
    for tensor in tensors_list:
        tensor = tensor.long().cpu()
        hypotheses = []
        for i in range(tensor.shape[0]):
            hypothesis = ''.join([labels_map[c] for c in tensor[i].numpy().tolist() if c in labels_map])
            hypotheses.append(hypothesis)

        results.append(hypotheses)

    return results
