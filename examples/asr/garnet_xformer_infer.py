# Copyright (c) 2019 NVIDIA Corporation
import argparse
import math
import os
import random
from functools import partial
from pprint import pformat
import time

import numpy as np
import torch
from ruamel.yaml import YAML

import nemo
import nemo.utils.argparse as nm_argparse
from nemo.utils.lr_policies import SquareAnnealing
import nemo_asr
from nemo_asr.las.helpers import process_evaluation_batch_xf, process_evaluation_epoch_xf
from nemo_asr.helpers import word_error_rate
import nemo_nlp


def parse_args():
    parser = argparse.ArgumentParser(
        parents=[nm_argparse.NemoArgParser()], description='GarNet with Transformer', conflict_handler='resolve'
    )
    parser.set_defaults(
        checkpoint_dir=None,
        optimizer="novograd",
        batch_size=32,
        eval_batch_size=32,
        weight_decay=1e-5,
        lr=1e-3,
        amp_opt_level="O1",
        create_tb_writer=True,
    )

    # Overwrite default args
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=0,
        help="number of epochs to train. You should specify" "either num_epochs or max_steps",
    )
    parser.add_argument("--model_config", type=str, required=True, help="model configuration file: model.yaml")

    # Create new args
    parser.add_argument("--exp_name", default="GarNet", type=str)
    parser.add_argument("--tokenizer_file", required=True, type=str)
    parser.add_argument("--random_seed", default=None, type=float)
    parser.add_argument('--encoder_checkpoint', default=None, type=str)
    parser.add_argument('--decoder_checkpoint', default=None, type=str)
    parser.add_argument('--beam_size', default=1, type=int)
    parser.add_argument('--enable_ctc_loss', action="store_true")
    parser.add_argument('--log_freq', default=250, type=int)
    parser.add_argument('--load_dir', required=True, type=str)

    args = parser.parse_args()
    if args.max_steps is not None:
        raise ValueError("GarNet uses num_epochs instead of max_steps")

    return args


def parse_cfg(args):
    yaml = YAML(typ="safe")
    with open(args.model_config) as f:
        garnet_params = yaml.load(f)

    garnet_params["optimization"]["params"]["lr"] = args.lr
    garnet_params["optimization"]["params"]["weight_decay"] = args.weight_decay
    garnet_params["optimization"]["params"]["num_epochs"] = args.num_epochs

    return garnet_params


def create_dag(args, garnet_params, neural_factory):
    char_labels = garnet_params["labels"]

    total_cpus = os.cpu_count()
    cpu_per_traindl = max(int(total_cpus / neural_factory.world_size), 1)

    # Defining nodes
    tokenizer = nemo_nlp.YouTokenToMeTokenizer(model_path=args.tokenizer_file)
    # tokenizer = nemo_nlp.BERTTokenizer(
    #     pretrained_model="bert-base-uncased")  # + "-vocab.txt")
    assert tokenizer.pad_id() == 0, f"{tokenizer.pad_id}"
    data = nemo_asr.TFAudioToTextDataLayer(
        manifest_filepath=args.eval_datasets[0],
        labels=char_labels,
        batch_size=args.batch_size,
        num_workers=cpu_per_traindl,
        tokenizer=tokenizer,
        **garnet_params['AudioToTextDataLayer']['eval'],
    )
    data_preprocessor = nemo_asr.AudioPreprocessing(**garnet_params['AudioPreprocessing'])
    encoder = nemo_asr.JasperEncoder(
        feat_in=garnet_params["AudioPreprocessing"]["features"], **garnet_params['JasperEncoder']
    )
    connector = nemo_asr.JasperRNNConnector(
        in_channels=garnet_params['JasperEncoder']['jasper'][-1]['filters'], out_channels=512
    )

    # Transformer Decoder
    vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)
    decoder = nemo_nlp.TransformerDecoderNM(
        d_model=512,
        d_inner=2048,
        num_layers=6,
        num_attn_heads=8,
        fully_connected_dropout=0.1,
        vocab_size=vocab_size,
        max_seq_length=512,
        embedding_dropout=0.1,
        learn_positional_encodings=True,
        first_sub_layer="self_attention",
        attn_score_dropout=0.1,
        attn_layer_dropout=0.1,
    )
    t_log_softmax = nemo_nlp.TransformerLogSoftmaxNM(vocab_size=vocab_size, d_model=512)
    decoder.restore_from("<update_me>")
    t_log_softmax.log_softmax.dense.weight = decoder.embedding_layer.token_embedding.weight

    loss = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(pad_id=tokenizer.pad_id(), label_smoothing=0.1)
    beam_translator = nemo_nlp.BeamSearchTranslatorNM(
        decoder=decoder,
        log_softmax=t_log_softmax,
        max_seq_length=512,
        beam_size=args.beam_size,
        length_penalty=0.0,
        bos_token=tokenizer.bos_id(),
        pad_token=tokenizer.pad_id(),
        eos_token=tokenizer.eos_id(),
    )
    int_to_seq = nemo_asr.IntToSeq()
    int_to_seq2 = nemo_asr.IntToSeq2()

    # Creating DAG
    audios, audio_lens, decoder_in, decoder_out, t_len = data()
    processed_audios, processed_audio_lens = data_preprocessor(input_signal=audios, length=audio_lens)
    encoded, enc_length = encoder(audio_signal=processed_audios, length=processed_audio_lens)
    enc_length = int_to_seq(x=encoded, length=enc_length)
    t_len = int_to_seq2(x=decoder_in, length=t_len)
    encoded = connector(tensor=encoded)
    # logits = decoder(
    #     input_ids_tgt=decoder_in,
    #     hidden_states_src=encoded,
    #     input_mask_src=enc_length,
    #     input_mask_tgt=t_len,
    # )
    # log_probs = t_log_softmax(hidden_states=logits)
    # train_loss = loss(
    #     log_probs=log_probs,
    #     target_ids=decoder_out
    # )
    beam_trans = beam_translator(hidden_states_src=encoded, input_mask_src=enc_length)

    return [beam_trans, decoder_out], tokenizer


def construct_name(args, cfg):
    name = '{}_{}_{}_{}'.format(
        cfg['model'], args.exp_name, 'bs' + str(args.batch_size), 'epochs' + str(args.num_epochs)
    )
    if args.work_dir:
        name = os.path.join(args.work_dir, name)
    return name


def main():
    # Parse args
    args = parse_args()
    garnet_params = parse_cfg(args)
    name = construct_name(args, garnet_params)

    # Define factory
    neural_factory = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        cudnn_benchmark=args.cudnn_benchmark,
        log_dir=name,
        create_tb_writer=True,
        files_to_copy=[args.model_config, __file__],
    )
    logger = neural_factory.logger
    tb_writer = neural_factory.tb_writer

    logger.info(f'Name: {name}')
    logger.info(f'Args to be passed to job #{args.local_rank}:')
    logger.info(pformat(vars(args)))

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        logger.info(f'Using seed {args.random_seed}')

    # Defining computational graph
    tensors, tokenizer = create_dag(args, garnet_params, neural_factory)

    start = time.time()
    evaluated_tensors = neural_factory.infer(tensors=tensors, checkpoint_dir=args.load_dir,)

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
    logger.info("Greedy WER {:.2f}%".format(wer * 100))

    end = time.time()
    print(f"Total time: {end-start}s")


if __name__ == '__main__':
    main()
