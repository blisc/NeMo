# Copyright (c) 2019 NVIDIA Corporation
import argparse
import math
import os
import random
from functools import partial
from pprint import pformat
import warnings
import time

import numpy as np
import torch
from ruamel.yaml import YAML

import nemo
import nemo.utils.argparse as nm_argparse
from nemo.utils.lr_policies import SquareAnnealing
import nemo_asr
from nemo_asr.las.helpers import process_evaluation_batch_xf, \
    process_evaluation_epoch_xf
from nemo_asr.helpers import word_error_rate
import nemo_nlp
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        parents=[nm_argparse.NemoArgParser()],
        description='GarNet with BERT',
        conflict_handler='resolve')
    parser.set_defaults(
        checkpoint_dir=None,
        optimizer="novograd",
        batch_size=32,
        eval_batch_size=24,
        weight_decay=1e-3,
        lr=1e-2,
        amp_opt_level="O1",
        create_tb_writer=True
    )

    # Overwrite default args
    parser.add_argument("--model_config", type=str, required=True,
                        help="model configuration file: model.yaml")

    # Create new args
    parser.add_argument("--exp_name", default="GarNet", type=str)
    parser.add_argument("--tokenizer_file", required=True, type=str)
    parser.add_argument("--random_seed", default=None, type=float)
    parser.add_argument('--encoder_checkpoint', default=None, type=str)
    parser.add_argument('--decoder_checkpoint', default=None, type=str)
    parser.add_argument('--beam_size', default=1, type=int)
    parser.add_argument('--enable_ctc_loss', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--decoder_layers', type=int, default=6)
    parser.add_argument('--decoder_d_inner', type=int, default=2048)
    parser.add_argument('--load_dir', required=True, type=str)

    args = parser.parse_args()
    if args.max_steps is not None:
        raise ValueError("GarNet uses num_epochs instead of max_steps")

    return args


def parse_cfg(args):
    yaml = YAML(typ="safe")
    with open(args.model_config) as f:
        garnet_params = yaml.load(f)

    return garnet_params


def create_dag_and_callbacks(args, garnet_params, neural_factory):
    char_labels = garnet_params["labels"]
    logger = neural_factory.logger

    total_cpus = os.cpu_count()
    cpu_per_traindl = max(int(total_cpus / neural_factory.world_size), 1)

    # Defining nodes
    tokenizer = nemo_nlp.SentencePieceTokenizer(
        model_path=args.tokenizer_file)
    # tokenizer = nemo_nlp.NemoBertTokenizer(
    #     pretrained_model="bert-base-uncased")  # + "-vocab.txt")
    if tokenizer.token_to_id("<pad>") != 0:
        raise ValueError(f'{tokenizer.token_to_id("<pad>")}')
    if args.debug:
        garnet_params['AudioToTextDataLayer']['train'][
            'normalize_transcripts'] = False

    data_eval = nemo_asr.MLMAudioToTextDataLayer(
        manifest_filepath=args.eval_datasets[0],
        batch_size=args.eval_batch_size,
        tokenizer=tokenizer,
        bos_id=tokenizer.token_to_id("<s>"),
        eos_id=tokenizer.token_to_id("</s>"),
        tokenizer_vocab_size=tokenizer.vocab_size,
        _eval=True,
        **garnet_params['AudioToTextDataLayer']['eval']
    )
    data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
        **garnet_params['AudioPreprocessing']
    )

    encoder = nemo_asr.JasperEncoder(
        **garnet_params['JasperEncoder']
    )

    vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

    connector = nemo_asr.JasperRNNConnector(
        in_channels=garnet_params['JasperEncoder']['jasper'][-1]['filters'],
        out_channels=512,
    )

    decoder = nemo_nlp.TransformerDecoderInferNM(
        d_model=512,
        d_inner=args.decoder_d_inner,
        num_layers=args.decoder_layers,
        num_attn_heads=8,
        ffn_dropout=0.1,
        vocab_size=vocab_size,
        max_seq_length=196,
        embedding_dropout=0.1,
        learn_positional_encodings=True,
        first_sub_layer="self_attention",
        attn_score_dropout=0.1,
        attn_layer_dropout=0.1,
        bidirectional=True
    )
    t_log_softmax = nemo_nlp.TokenClassifier(
        num_classes=vocab_size,
        hidden_size=512
    )

    loss = nemo_nlp.MaskedLanguageModelingLossNM()
    int_to_seq = nemo_asr.IntToSeq()
    int_to_seq2 = nemo_asr.IntToSeq2()
    greedy_decoder = nemo_asr.GreedyCTCDecoder()

    audios, audio_lens, decoder_in, decoder_out, t_len, out_mask, char_t, char_l = data_eval()
    processed_audios, processed_audio_lens = data_preprocessor(
        input_signal=audios,
        length=audio_lens
    )
    encoded, enc_length_0 = encoder(
        audio_signal=processed_audios,
        length=processed_audio_lens
    )
    enc_length = int_to_seq(x=encoded, length=enc_length_0)
    t_len = int_to_seq2(x=decoder_in, length=t_len)
    connected_encoded = connector(tensor=encoded)

    if args.enable_ctc_loss:
        ctc_decoder = nemo_asr.JasperDecoderForCTC(
            feat_in=garnet_params["JasperEncoder"]["jasper"][-1]["filters"],
            num_classes=len(char_labels)
        )
        ctc_loss = nemo_asr.CTCLossNM(num_classes=len(char_labels))

        ctc_log_probs = ctc_decoder(encoder_output=encoded)
        ctc_predictions = greedy_decoder(log_probs=ctc_log_probs)
        ctc_loss_tensor = ctc_loss(
            log_probs=ctc_log_probs,
            targets=char_t,
            input_length=enc_length_0,
            target_length=char_l)

    logits = decoder(
        input_ids_tgt=decoder_in,
        hidden_states_src=connected_encoded,
        input_mask_src=enc_length,
        input_mask_tgt=t_len,
    )
    log_probs = t_log_softmax(hidden_states=logits)
    eval_loss = loss(
        logits=log_probs,
        output_ids=decoder_out,
        output_mask=out_mask)

    predictions = greedy_decoder(log_probs=log_probs)


    assert(t_log_softmax.mlp.last_linear_layer.weight.shape ==
           decoder.embedding_layer.token_embedding.weight.shape)
    t_log_softmax.mlp.last_linear_layer.weight = \
        decoder.embedding_layer.token_embedding.weight

    return (predictions, decoder_out, ctc_predictions), tokenizer


def main():
    # Parse args
    args = parse_args()
    garnet_params = parse_cfg(args)

    # Define factory
    neural_factory = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        cudnn_benchmark=args.cudnn_benchmark,
        create_tb_writer=False,
    )
    logger = neural_factory.logger

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        logger.info(f'Using seed {args.random_seed}')

    # Defining computational graph
    tensors, tokenizer = create_dag_and_callbacks(
        args, garnet_params, neural_factory)

    # Optimize
    start = time.time()
    evaluated_tensors = neural_factory.infer(
        tensors=list(tensors),
        checkpoint_dir=args.load_dir,
    )

    predictions = []
    for t in evaluated_tensors[0]:
        t = t.cpu().numpy().tolist()
        for k in t:
            # import ipdb; ipdb.set_trace()
            predictions.append(tokenizer.ids_to_text(k))

    references = []
    for t in evaluated_tensors[1]:
        t = t.cpu().numpy().tolist()
        for k in t:
            references.append(tokenizer.ids_to_text(k))

    wer = word_error_rate(hypotheses=predictions, references=references)
    logger.info("Greedy WER {:.2f}%".format(wer*100))

    end = time.time()
    print(f"Total time: {end-start}s")


if __name__ == '__main__':
    main()
