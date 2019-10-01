#
#   MobileNet classification network
#   Copyright EAVISE
#

import functools
from collections import OrderedDict
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['MobileNetV2']


class MobileNetV2(lnn.module.Lightnet):
    """ MobileNet v2 classification network implementation :cite:`mobilenet_v2`.

    Args:
        num_classes (Number, optional): Number of classes; Default **1000**
        alpha (Number, optional): Number between [0-1] that controls the number of filters of the mobilenet convolutions; Default **1**
        input_channels (Number, optional): Number of input channels; Default **3**

    Attributes:
        self.stride: Subsampling factor of the network (input dimensions should be a multiple of this number)

    Note:
        The average pooling is implemented with an :class:`~torch.nn.AdaptiveAvgPool2d` layer. |br|
        For the base input dimension of 224x224, this is exactly the same as a 7x7 average pooling function,
        but the advantage of a adaptive average pooling is that this network can now handle multiple different input dimensions,
        as long as they are a multiple of the ``stride`` factor. |br|
        This is also how the implementation in `tensorflow <mobilenetv2tf_>`_ works.

    Warning:
        When changing the ``alpha`` value, you are changing the network architecture.
        This means you cannot use weights from this network with a different alpha value.

    .. _mobilenetv2tf: https://github.com/tensorflow/models/blob/505f554c6417931c96b59516f14d1ad65df6dbc5/research/slim/nets/mobilenet/mobilenet.py#L365
    """
    stride = 32

    def __init__(self, num_classes=1000, alpha=1, input_channels=3):
        super().__init__()

        # Parameters
        self.num_classes = num_classes
        self.alpha = alpha
        self.input_channels = input_channels

        # Network
        self.layers = nn.Sequential(
            # Base layers
            nn.Sequential(OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, int(alpha*32), 3, 2, 1, relu=functools.partial(nn.ReLU6, inplace=True))),
                ('2_bottleneck',    lnn.layer.Bottleneck(int(alpha*32), int(alpha*16), 3, 1, 1)),
                ('3_bottleneck',    lnn.layer.Bottleneck(int(alpha*16), int(alpha*24), 3, 2, 6)),
                ('4_bottleneck',    lnn.layer.Bottleneck(int(alpha*24), int(alpha*24), 3, 1, 6)),
                ('5_bottleneck',    lnn.layer.Bottleneck(int(alpha*24), int(alpha*32), 3, 2, 6)),
                ('6_bottleneck',    lnn.layer.Bottleneck(int(alpha*32), int(alpha*32), 3, 1, 6)),
                ('7_bottleneck',    lnn.layer.Bottleneck(int(alpha*32), int(alpha*32), 3, 1, 6)),
                ('8_bottleneck',    lnn.layer.Bottleneck(int(alpha*32), int(alpha*64), 3, 2, 6)),
                ('9_bottleneck',    lnn.layer.Bottleneck(int(alpha*64), int(alpha*64), 3, 1, 6)),
                ('10_bottleneck',   lnn.layer.Bottleneck(int(alpha*64), int(alpha*64), 3, 1, 6)),
                ('11_bottleneck',   lnn.layer.Bottleneck(int(alpha*64), int(alpha*64), 3, 1, 6)),
                ('12_bottleneck',   lnn.layer.Bottleneck(int(alpha*64), int(alpha*96), 3, 1, 6)),
                ('13_bottleneck',   lnn.layer.Bottleneck(int(alpha*96), int(alpha*96), 3, 1, 6)),
                ('14_bottleneck',   lnn.layer.Bottleneck(int(alpha*96), int(alpha*96), 3, 1, 6)),
                ('15_bottleneck',   lnn.layer.Bottleneck(int(alpha*96), int(alpha*160), 3, 2, 6)),
                ('16_bottleneck',   lnn.layer.Bottleneck(int(alpha*160), int(alpha*160), 3, 1, 6)),
                ('17_bottleneck',   lnn.layer.Bottleneck(int(alpha*160), int(alpha*160), 3, 1, 6)),
                ('18_bottleneck',   lnn.layer.Bottleneck(int(alpha*160), int(alpha*320), 3, 1, 6)),
                ('19_convbatch',    lnn.layer.Conv2dBatchReLU(int(alpha*320), int(alpha*1280),  1, 1, 0, relu=functools.partial(nn.ReLU6, inplace=True))),
            ])),

            # Classification specific layers
            nn.Sequential(OrderedDict([
                ('20_avgpool',      nn.AdaptiveAvgPool2d(1)),
                ('21_dropout',      nn.Dropout()),
                ('22_conv',         nn.Conv2d(int(alpha*1280), num_classes, 1, 1, 0)),
                ('23_flatten',      lnn.layer.Flatten()),
            ])),
        )
