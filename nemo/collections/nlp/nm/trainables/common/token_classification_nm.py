__all__ = ['BertTokenClassifier', 'TokenClassifier']
from torch import nn as nn

from nemo.backends.pytorch import MultiLayerPerceptron, TrainableNM
from nemo.collections.nlp.nm.trainables.common.transformer.transformer_utils import gelu, transformer_weights_init
from nemo.core import AxisType, BatchTag, ChannelTag, NeuralType, TimeTag

ACT2FN = {"gelu": gelu, "relu": nn.functional.relu}


class BertTokenClassifier(TrainableNM):
    """
    Neural module which consists of MLP followed by softmax classifier for each
    token in the sequence.

    Args:
        hidden_size (int): hidden size (d_model) of the Transformer
        num_classes (int): number of classes in softmax classifier, e.g. size
            of the vocabulary in language modeling objective
        activation (str): activation function applied in classifier MLP layers
        log_softmax (bool): whether to apply log_softmax to MLP output
        dropout (float): dropout ratio applied to MLP
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        hidden_states:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)
        """
        return {"hidden_states": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)})}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)
        """
        return {"logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)})}

    def __init__(
        self,
        hidden_size,
        num_classes,
        activation='relu',
        log_softmax=True,
        dropout=0.0,
        use_transformer_pretrained=True,
    ):
        super().__init__()
        if activation not in ACT2FN:
            raise ValueError(f'activation "{activation}" not found')
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = ACT2FN[activation]
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlp = MultiLayerPerceptron(
            hidden_size, num_classes, self._device, num_layers=1, activation=activation, log_softmax=log_softmax
        )
        self.dropout = nn.Dropout(dropout)
        if use_transformer_pretrained:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))
        self.to(self._device)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        transform = self.norm(hidden_states)
        logits = self.mlp(transform)
        return logits


class TokenClassifier(TrainableNM):
    """
    Neural module which consists of MLP followed by softmax classifier for each
    token in the sequence.

    Args:
        hidden_size (int): hidden size (d_model) of the Transformer
        num_classes (int): number of classes in softmax classifier, e.g. size
            of the vocabulary in language modeling objective
        num_layers (int): number of layers in classifier MLP
        activation (str): activation function applied in classifier MLP layers
        log_softmax (bool): whether to apply log_softmax to MLP output
        dropout (float): dropout ratio applied to MLP
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        hidden_states:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)
        """
        return {"hidden_states": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)})}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)
        """
        return {"logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)})}

    def __init__(
        self,
        hidden_size,
        num_classes,
        name=None,
        num_layers=2,
        activation='relu',
        log_softmax=True,
        dropout=0.0,
        use_transformer_pretrained=True,
    ):
        super().__init__()

        self.name = name
        self.mlp = MultiLayerPerceptron(hidden_size, num_classes, self._device, num_layers, activation, log_softmax)
        self.dropout = nn.Dropout(dropout)
        if use_transformer_pretrained:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))
        # self.to(self._device) # sometimes this is necessary

    def __str__(self):
        name = TrainableNM.__str__(self)

        if self.name:
            name = self.name + name
        return name

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states)
        return logits
