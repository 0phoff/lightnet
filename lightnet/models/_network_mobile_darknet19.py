#
#   Mobile Darknet19 model
#   Copyright EAVISE
#

import functools
from collections import OrderedDict
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['MobileDarknet19']


class MobileDarknet19(lnn.module.Darknet):
    """ Darknet19 implementation with depthwise separable convolutions.

    Args:
        num_classes (Number, optional): Number of classes; Default **1000**
        input_channels (Number, optional): Number of input channels; Default **3**

    Attributes:
        self.stride: Subsampling factor of the network (input dimensions should be a multiple of this number)
    """
    stride = 32

    def __init__(self, num_classes=1000, input_channels=3):
        super().__init__()

        # Parameters
        self.num_classes = num_classes
        self.input_channels = input_channels

        # Network
        relu = functools.partial(nn.LeakyReLU, 0.1, inplace=True)
        self.layers = nn.Sequential(
            # Base layers
            nn.Sequential(OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 2, 1, relu=relu)),
                ('2_convdw',        lnn.layer.Conv2dDepthWise(32, 64, 3, 2, 1, relu=relu)),
                ('3_convdw',        lnn.layer.Conv2dDepthWise(64, 128, 3, 1, 1, relu=relu)),
                ('4_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0, relu=relu)),
                ('5_convdw',        lnn.layer.Conv2dDepthWise(64, 128, 3, 2, 1, relu=relu)),
                ('6_convdw',        lnn.layer.Conv2dDepthWise(128, 256, 3, 1, 1, relu=relu)),
                ('7_convbatch',     lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0, relu=relu)),
                ('8_convdw',        lnn.layer.Conv2dDepthWise(128, 256, 3, 2, 1, relu=relu)),
                ('9_convdw',        lnn.layer.Conv2dDepthWise(256, 512, 3, 1, 1, relu=relu)),
                ('10_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu)),
                ('11_convdw',       lnn.layer.Conv2dDepthWise(256, 512, 3, 1, 1, relu=relu)),
                ('12_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0, relu=relu)),
                ('13_convdw',       lnn.layer.Conv2dDepthWise(256, 512, 3, 1, 1, relu=relu)),
                ('14_max',          nn.MaxPool2d(2, 2)),
                ('15_convdw',       lnn.layer.Conv2dDepthWise(512, 1024, 3, 2, 1, relu=relu)),
                ('16_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0, relu=relu)),
                ('17_convdw',       lnn.layer.Conv2dDepthWise(512, 1024, 3, 1, 1, relu=relu)),
                ('18_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0, relu=relu)),
                ('19_convdw',       lnn.layer.Conv2dDepthWise(512, 1024, 3, 1, 1, relu=relu)),
            ])),

            # Classification specific layers
            nn.Sequential(OrderedDict([
                ('20_conv',         nn.Conv2d(1024, num_classes, 1, 1, 0)),
                ('21_avgpool',      nn.AdaptiveAvgPool2d(1)),
                ('22_flatten',      lnn.layer.Flatten()),
            ])),
        )
