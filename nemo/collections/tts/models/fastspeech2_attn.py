from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from hydra.utils import instantiate
from torch import nn
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
import numpy as np


from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging
from nemo.collections.tts.losses.tacotron2loss import L2MelLoss, L1MelLoss
from nemo.collections.tts.helpers.helpers import (
    plot_spectrogram_to_numpy,
    get_mask_from_lengths,
    plot_alignment_to_numpy,
)

# from nemo.collections.tts.modules.fastspeech2_submodules import VariancePredictor, LengthRegulator
from nemo.collections.tts.modules.fastspeech2_v2 import FFTBlocks, FFTBlocksWithEncDecAttn
from nemo.collections.tts.modules.fastspeech2_v3 import ConvAttention, Loss2, ABLoss, SingleHeadAttention, mas, Loss3
from nemo.core.classes import ModelPT


class FastSpeech2AttnModel(ModelPT):
    """FastSpeech 2 model used to convert between text (phonemes) and mel-spectrograms."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        # schema = OmegaConf.structured(FastSpeech2Config)
        # # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        # if isinstance(cfg, dict):
        #     cfg = OmegaConf.create(cfg)
        # elif not isinstance(cfg, DictConfig):
        #     raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # # Ensure passed cfg is compliant with schema
        # OmegaConf.merge(cfg, schema)

        self.audio_to_melspec_precessor = instantiate(self._cfg.preprocessor)
        self.phone_embedding = nn.Embedding(len(self._cfg.labels) + 1, 256, padding_idx=len(self._cfg.labels))
        self.encoder = FFTBlocks(max_seq_len=512, name="enc")
        # self.attention = ConvAttention()
        self.attention = SingleHeadAttention(attn_layer_dropout=self._cfg.attn_dropout)
        # self.duration_predictor = VariancePredictor(d_model=256, d_inner=256, kernel_size=3, dropout=0.5)
        self.mel_decoder = FFTBlocks(max_seq_len=2048, name="dec")
        self.mel_linear = nn.Linear(256, 80, bias=True)

        self.loss = L1MelLoss()
        self.loss2 = None
        if self._cfg.add_loss2:
            self.loss2 = Loss2()
        self.loss3 = Loss3()

        self.log_train_images = False
        self.logged_real_samples = False
        self.binarize_attention = False
        self.train_duration_predictor = False
        self._tb_logger = None
        typecheck.set_typecheck_enabled(enabled=False)

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            self._tb_logger = tb_logger
        return self._tb_logger

    def forward(self, *, spec, spec_len, text, text_length, attn_prior=None):
        if self.training and spec.shape[2] > attn_prior.shape[1] and attn_prior is not None:
            attn_prior = torch.nn.functional.pad(attn_prior, (0, 0, 0, spec.shape[2] - attn_prior.shape[1], 0, 0))
        embedded_tokens = self.phone_embedding(text)
        encoded_text, _ = self.encoder(embedded_tokens, text_length)

        # # ConvAttention
        # attn_mask = get_mask_from_lengths(text_length) == 0
        # attn_logprob = self.attention(spec, encoded_text.transpose(1, 2), attn_prior)
        # attn = attn_logprob.squeeze().masked_fill(attn_mask.unsqueeze(1), -float("inf"))
        # attn = torch.nn.functional.softmax(attn, dim=2)
        # context = torch.bmm(attn, encoded_text)

        # SingleHeadAttention
        context, attn, attn2 = self.attention(
            spec.transpose(1, 2),
            encoded_text,
            encoded_text,
            get_mask_from_lengths(text_length),
            prior=attn_prior,
            binarize=self.binarize_attention,
            in_len=text_length,
            out_len=spec_len,
        )

        output, _ = self.mel_decoder(context, spec_len)
        ouput = self.mel_linear(output)

        return ouput, attn, attn2

    def training_step(self, batch, batch_idx):
        f, fl, t, tl, attn_prior = batch
        spec, spec_len = self.audio_to_melspec_precessor(f, fl)

        mel, attn, attn2 = self(spec=spec, spec_len=spec_len, text=t, text_length=tl, attn_prior=attn_prior)
        attn_logprob = torch.log(attn + 1e-8).unsqueeze(1)

        # Loss
        loss1 = self.loss(spec_pred=mel, spec_target=spec, spec_target_len=spec_len, pad_value=-11.52)
        self.log(name="train_mel_loss", value=loss1)
        output_dict = {"loss": loss1}
        if self.global_step < 1000:
            if self.global_step % 100 == 0:
                attn_logprob_tolog = attn_logprob[0].data.cpu().numpy().squeeze()
                self.tb_logger.add_image(
                    "train_attn", plot_alignment_to_numpy(attn_logprob_tolog.T), self.global_step, dataformats="HWC",
                )
                self.log_train_images = False
        elif self.global_step < 10000 and self.global_step % 1000 == 0:
            attn_logprob_tolog = attn_logprob[0].data.cpu().numpy().squeeze()
            self.tb_logger.add_image(
                "train_attn", plot_alignment_to_numpy(attn_logprob_tolog.T), self.global_step, dataformats="HWC",
            )
            self.log_train_images = False
        if self.log_train_images:
            self.log_train_images = False
            output_dict["outputs"] = [spec, mel, attn_logprob]
        if self.loss2:
            loss2 = self.loss2(attn_logprob, tl, spec_len)
            total_loss = loss1 + loss2
            self.log(name="train_loss_2", value=loss2)
            output_dict["loss"] = total_loss
        if self.binarize_attention:
            loss3 = self.loss3(attn, attn2)
            total_loss += loss3
            self.log(name="train_loss_3", value=loss3)
            output_dict["loss"] = total_loss
        self.log(name="train_loss", value=output_dict["loss"])
        return output_dict

    def training_epoch_end(self, training_step_outputs):
        if "outputs" in training_step_outputs[0]:
            # if self.logger is not None and self.logger.experiment is not None:
            spec_target, spec_predict, attn = training_step_outputs[0]["outputs"]
            self.tb_logger.add_image(
                "train_mel_target",
                plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = spec_predict[0].data.cpu().numpy()
            self.tb_logger.add_image(
                "train_mel_predicted", plot_spectrogram_to_numpy(spec_predict.T), self.global_step, dataformats="HWC",
            )
            attn = attn[0].data.cpu().numpy().squeeze()
            self.tb_logger.add_image(
                "train_attn", plot_alignment_to_numpy(attn.T), self.global_step, dataformats="HWC",
            )

        # Switch to hard attention after 40% of training
        if not self.binarize_attention and self.current_epoch >= np.ceil(0.4 * self._trainer.max_epochs):
            logging.info(f"Using hard attentions at epoch: {self.current_epoch}")
            self.binarize_attention = True

        # Start training duration predictoru after 75% of training
        if not self.train_duration_predictor and self.current_epoch >= np.ceil(0.75 * self._trainer.max_epochs):
            logging.info(f"Starting training duration predictory at epoch: {self.current_epoch}")
            self.train_duration_predictor = True

    def validation_step(self, batch, batch_idx):
        f, fl, t, tl, _ = batch
        spec, spec_len = self.audio_to_melspec_precessor(f, fl)

        mel, _, _ = self(spec=spec, spec_len=spec_len, text=t, text_length=tl)

        # Loss
        loss = self.loss(spec_pred=mel, spec_target=spec, spec_target_len=spec_len, pad_value=-11.52)

        return {
            "val_loss": loss,
            "mel_target": spec,
            "mel_pred": mel,
        }

    def validation_epoch_end(self, outputs):
        if self.tb_logger is not None:
            _, spec_target, spec_predict = outputs[0].values()
            if not self.logged_real_samples:
                self.tb_logger.add_image(
                    "val_mel_target",
                    plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )
                self.logged_real_samples = True
            spec_predict = spec_predict[0].data.cpu().numpy()
            self.tb_logger.add_image(
                "val_mel_predicted", plot_spectrogram_to_numpy(spec_predict.T), self.global_step, dataformats="HWC",
            )
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()  # This reduces across batches, not workers!
        self.log('val_loss', avg_loss, sync_dist=True)

        self.log_train_images = True

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg.dataloader_params):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        dataset = instantiate(cfg.dataset, labels=self._cfg.labels, pad_id=len(self._cfg.labels))
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def list_available_models(self):
        pass

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

