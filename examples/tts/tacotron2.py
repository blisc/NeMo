# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
import torch
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import Tacotron2Model
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="tacotron2")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = Tacotron2Model(cfg=cfg.model, trainer=trainer)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


# def infer():
#     model = Tacotron2Model(cfg=cfg.model)
#     model = model.cuda()
#     model.load_from_checkpoint(
#         "/NeMo/examples/tts/experiments/1441344-Tacotron_O1_LJS_V1b/Tacotron2/2020-09-12_00-05-20/checkpoints/Tacotron2--last.ckpt"
#     )
#     model.eval()
#     with torch.no_grad():
#         with torch.cuda.amp.autocast():
#             input_ = model.parse(str_input="this is a test.")
#             spec = model.generate_spectrogram(tokens=input_)

#     from matplotlib import pyplot as plt

#     fig, (ax1, ax2) = plt.subplots(2, 1)
#     ax1.imshow(spec[0].cpu().numpy().astype(float), origin="lower")
#     plt.savefig("spec")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
