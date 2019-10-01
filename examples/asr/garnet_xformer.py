# Copyright (c) 2019 NVIDIA Corporation
import argparse
import math
import os
import random
from functools import partial
from pprint import pformat
import warnings

import numpy as np
import torch
from ruamel.yaml import YAML

import nemo
import nemo.utils.argparse as nm_argparse
from nemo.utils.lr_policies import SquareAnnealing
import nemo_asr
from nemo_asr.las.helpers import process_evaluation_batch_xf, \
    process_evaluation_epoch_xf
import nemo_nlp
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        parents=[nm_argparse.NemoArgParser()],
        description='GarNet with Transformer',
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
    parser.add_argument("--num_epochs", type=int, default=None, required=True,
                        help="number of epochs to train. You should specify"
                             "either num_epochs or max_steps")
    parser.add_argument("--model_config", type=str, required=True,
                        help="model configuration file: model.yaml")
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="training dataset path")
    parser.add_argument("--beta1", type=float,
                        help="Adam/AdamW/NovoGrad beta1")
    parser.add_argument("--beta2", type=float, default=0.25,
                        help="Adam/AdamW/NovoGrad beta2")

    # Create new args
    parser.add_argument("--exp_name", default="GarNet", type=str)
    parser.add_argument("--tokenizer_file", required=True, type=str)
    parser.add_argument("--random_seed", default=None, type=float)
    parser.add_argument('--encoder_checkpoint', default=None, type=str)
    parser.add_argument('--decoder_checkpoint', default=None, type=str)
    parser.add_argument('--beam_size', default=1, type=int)
    parser.add_argument('--enable_ctc_loss', action="store_true")
    parser.add_argument('--log_freq', default=250, type=int)
    parser.add_argument('--add_time_dir', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--decoder_layers', type=int, default=6)

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
    betas = garnet_params["optimization"]["params"]["betas"]
    if args.beta1:
        betas[0] = args.beta1
    if args.beta2:
        betas[1] = args.beta2
    garnet_params["optimization"]["params"]["betas"] = betas

    return garnet_params


def create_dag_and_callbacks(args, garnet_params, neural_factory):
    char_labels = garnet_params["labels"]
    logger = neural_factory.logger

    total_cpus = os.cpu_count()
    cpu_per_traindl = max(int(total_cpus / neural_factory.world_size), 1)

    # Defining nodes
    tokenizer = nemo_nlp.YouTokenToMeTokenizer(
        model_path=args.tokenizer_file)
    # tokenizer = nemo_nlp.NemoBertTokenizer(
        # pretrained_model="bert-base-uncased")  # + "-vocab.txt")
    assert tokenizer.pad_id() == 0, f"{tokenizer.pad_id}"
    if args.debug:
        garnet_params['AudioToTextDataLayer']['train'][
            'normalize_transcripts'] = False
    data = nemo_asr.TFAudioToTextDataLayer(
        manifest_filepath=args.train_dataset,
        batch_size=args.batch_size,
        num_workers=cpu_per_traindl,
        tokenizer=tokenizer,
        **garnet_params['AudioToTextDataLayer']['train']
    )
    data_preprocessor = nemo_asr.AudioPreprocessing(
        **garnet_params['AudioPreprocessing']
    )
    data_augmentation = nemo_asr.SpectrogramAugmentation(
        **garnet_params['SpectrogramAugmentation']
    )
    encoder = nemo_asr.JasperEncoder(
        **garnet_params['JasperEncoder']
    )
    if args.encoder_checkpoint is not None \
            and os.path.exists(args.encoder_checkpoint):
        encoder.restore_from(args.encoder_checkpoint, args.local_rank)
        logger.info(
            f'Loaded weights for encoder'
            f' from {args.encoder_checkpoint}')
        if garnet_params['JasperEncoder']['freeze']:
            encoder.freeze()
            logger.info(f'Freeze encoder weights')
    vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

    connector = nemo_asr.JasperRNNConnector(
        in_channels=garnet_params['JasperEncoder']['jasper'][-1]['filters'],
        out_channels=512
    )
    decoder = nemo_nlp.TransformerDecoderNM(
        d_model=512,
        d_inner=2048,
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
    )
    t_log_softmax = nemo_nlp.TransformerLogSoftmaxNM(
        vocab_size=vocab_size,
        d_model=512
    )

    # connector = nemo_asr.JasperRNNConnector(
    #     in_channels=garnet_params['JasperEncoder']['jasper'][-1]['filters'],
    #     out_channels=768
    # )
    # decoder = nemo_nlp.TransformerDecoderNM(
    #     d_model=768,
    #     d_inner=3072,
    #     num_layers=6,
    #     num_attn_heads=12,
    #     ffn_dropout=0.1,
    #     vocab_size=vocab_size,
    #     max_seq_length=512,
    #     embedding_dropout=0.1,
    #     learn_positional_encodings=True,
    #     first_sub_layer="self_attention",
    #     attn_score_dropout=0.1,
    #     attn_layer_dropout=0.1,
    # )
    # t_log_softmax = nemo_nlp.TransformerLogSoftmaxNM(
    #     vocab_size=vocab_size,
    #     d_model=768
    # )

    t_log_softmax.log_softmax.dense.weight = \
        decoder.embedding_layer.token_embedding.weight
    if args.decoder_checkpoint is not None \
            and os.path.exists(args.decoder_checkpoint):
        decoder.restore_from(args.decoder_checkpoint, args.local_rank)
        logger.info(f'Loaded weights for decoder'
                    f' from {args.decoder_checkpoint}')
        # if cfg['DecoderRNN']['freeze']:
        #     decoder.freeze()
        #     if VERBOSE:
        #         print(f'Freeze decoder weights')
        #     if cfg['decoder']['unfreeze_attn']:
        #         for _, param in decoder.attention.named_parameters():
        #             param.requires_grad = True
        #         if VERBOSE:
        #             print(f'Unfreeze decoder attn weights')

    loss = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(
        pad_id=tokenizer.pad_id(),
        label_smoothing=0.1
    )
    beam_translator = nemo_nlp.BeamSearchTranslatorNM(
        decoder=decoder,
        log_softmax=t_log_softmax,
        max_seq_length=196,
        beam_size=args.beam_size,
        length_penalty=0.0,
        bos_token=tokenizer.bos_id(),
        pad_token=tokenizer.pad_id(),
        eos_token=tokenizer.eos_id()
    )
    int_to_seq = nemo_asr.IntToSeq()
    int_to_seq2 = nemo_asr.IntToSeq2()

    # Creating DAG
    audios, audio_lens, decoder_in, decoder_out, t_len, char_t, char_l = data()
    processed_audios, processed_audio_lens = data_preprocessor(
        input_signal=audios,
        length=audio_lens
    )
    augmented_spec = data_augmentation(input_spec=processed_audios)
    encoded, enc_length_0 = encoder(
        audio_signal=augmented_spec,
        length=processed_audio_lens
    )
    enc_length = int_to_seq(x=encoded, length=enc_length_0)
    t_len = int_to_seq2(x=decoder_in, length=t_len)
    connected_encoded = connector(tensor=encoded)
    logits = decoder(
        input_ids_tgt=decoder_in,
        hidden_states_src=connected_encoded,
        input_mask_src=enc_length,
        input_mask_tgt=t_len,
    )
    log_probs = t_log_softmax(hidden_states=logits)
    train_loss = loss(
        log_probs=log_probs,
        target_ids=decoder_out
    )
    train_loss = [train_loss]
    callbacks = []
    # Callbacks
    train_callback = nemo.core.SimpleLossLoggerCallback(
        tensors=train_loss,
        print_func=lambda x: logger.info(f"Train SeqLoss: {x[0].item()}"),
        get_tb_values=lambda x: [
            ("a_loss", x[0]),
            ("b_seqloss", x[0])],
        step_freq=args.log_freq,
        tb_writer=neural_factory.tb_writer
    )

    if args.enable_ctc_loss:
        logger.info("Training with joint CTC and seqloss")
        ctc_decoder = nemo_asr.JasperDecoderForCTC(
            feat_in=garnet_params["JasperEncoder"]["jasper"][-1]["filters"],
            num_classes=len(char_labels)
        )
        ctc_loss = nemo_asr.CTCLossNM(num_classes=len(char_labels))
        greedy_decoder = nemo_asr.GreedyCTCDecoder()

        ctc_log_probs = ctc_decoder(encoder_output=encoded)
        ctc_predictions = greedy_decoder(log_probs=ctc_log_probs)
        ctc_loss_tensor = ctc_loss(
            log_probs=ctc_log_probs,
            targets=char_t,
            input_length=enc_length_0,
            target_length=char_l)
        train_loss = [train_loss[0], ctc_loss_tensor]
        train_callback = nemo.core.SimpleLossLoggerCallback(
            tensors=train_loss,
            print_func=lambda x: logger.info(
                f"\n"
                f"\t\tTrain Loss: {x[0].item() + x[1].item():4f}\n"
                f"\t\tTrain SeqLoss: {x[0].item():4f}\n"
                f"\t\tTrain CTCLoss: {x[1].item():4f}"),
            get_tb_values=lambda x: [
                ("a_loss", x[0]+x[1]),
                ("b_seqloss", x[0]),
                ("c_ctcloss", x[1])],
            step_freq=args.log_freq,
            tb_writer=neural_factory.tb_writer
        )

    callbacks.append(train_callback)

    if args.eval_datasets:
        for val_path in args.eval_datasets:
            data_eval = nemo_asr.TFAudioToTextDataLayer(
                manifest_filepath=val_path,
                batch_size=args.eval_batch_size,
                tokenizer=tokenizer,
                **garnet_params['AudioToTextDataLayer']['eval']
            )
            audios, audio_lens, decoder_in, decoder_out, t_len, char_t, char_l\
                = data_eval()
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
            logits = decoder(
                input_ids_tgt=decoder_in,
                hidden_states_src=connected_encoded,
                input_mask_src=enc_length,
                input_mask_tgt=t_len,
            )
            log_probs = t_log_softmax(hidden_states=logits)
            eval_loss = loss(
                log_probs=log_probs,
                target_ids=decoder_out
            )
            beam_trans = beam_translator(
                hidden_states_src=connected_encoded, input_mask_src=enc_length
            )

            tensors = [eval_loss, decoder_out, beam_trans]

            if args.enable_ctc_loss:
                ctc_log_probs = ctc_decoder(encoder_output=encoded)
                ctc_predictions = greedy_decoder(log_probs=ctc_log_probs)
                ctc_loss_tensor = ctc_loss(
                    log_probs=ctc_log_probs,
                    targets=char_t,
                    input_length=enc_length_0,
                    target_length=char_l)
                tensors.extend(
                    [ctc_loss_tensor, char_t, char_l, ctc_predictions])

            eval_callback = nemo.core.EvaluatorCallback(
                eval_tensors=list(tensors),
                user_iter_callback=partial(
                    process_evaluation_batch_xf,
                    tokenizer=tokenizer,
                    labels=char_labels
                ),
                user_epochs_done_callback=partial(
                    process_evaluation_epoch_xf,
                    tag=os.path.basename(val_path),
                    calc_wer=True,
                    logger=logger
                ),
                eval_step=args.eval_freq,
                tb_writer=neural_factory.tb_writer
            )
            callbacks.append(eval_callback)
    else:
        logger.warning("No val dataset")

    saver_callback = nemo.core.CheckpointCallback(
        folder=neural_factory.checkpoint_dir,
        step_freq=args.checkpoint_save_freq
    )
    callbacks.append(saver_callback)

    return train_loss, callbacks, len(data)


def construct_name(args, cfg):
    world_size = 1
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    name = '{}_{}_{}_{}_{}'.format(
        cfg['model'],
        args.exp_name,
        'bs' + str(args.batch_size * world_size * args.iter_per_step),
        'epochs' + str(args.num_epochs),
        'ctc' + str(args.enable_ctc_loss)
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
        add_time_to_log_dir=args.add_time_dir
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
    loss, callbacks, num_data = create_dag_and_callbacks(
        args, garnet_params, neural_factory)

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    steps_per_epoch = int(num_data / (batch_size * neural_factory.world_size))
    total_steps = num_epochs * steps_per_epoch

    neural_factory.logger.info(f'Num data: {num_data}')
    neural_factory.logger.info(f'Steps per epoch: {steps_per_epoch}')
    neural_factory.logger.info(f'Total steps: {total_steps}')
    neural_factory.logger.info(
        f'Optimization Params:\n{pformat(garnet_params["optimization"])}')

    # Optimize
    neural_factory.train(
        tensors_to_optimize=loss,
        callbacks=callbacks,
        lr_policy=SquareAnnealing(
            total_steps,
            min_lr=garnet_params['optimization']['min_lr'],
            warmup_steps=(
                garnet_params['optimization']['warmup_epochs']
                * steps_per_epoch
            )
        ),
        optimizer=args.optimizer,
        optimization_params=garnet_params['optimization']['params'],
        batches_per_step=args.iter_per_step
    )


if __name__ == '__main__':
    main()
