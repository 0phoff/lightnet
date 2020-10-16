#
#   Darknet YOLOv2 model with upsample instead of Reorg
#   Copyright EAVISE
#

import logging
from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['YoloV2Upsample']
log = logging.getLogger('lightnet.models')


class YoloV2Upsample(lnn.module.Darknet):
    """ Yolo v2 implementation with an upsampling layer instead of a reorg layer.
        TODO : add more information
        TODO : Default anchors *2

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (list, optional): 2D list with anchor values; Default **Yolo v2 anchors**

    Attributes:
        self.stride: Subsampling factor of the network (input_dim / output_dim)
        self.inner_stride: Maximal internal subsampling factor of the network (input dimension should be a multiple of this)
        self.remap_darknet19: Remapping rules for weights from the :class:`~lightnet.models.Darknet19` model.
    """
    stride = 16
    inner_stride = 32
    remap_darknet19 = [
        (r'^layers.0.([1-9]_)',     r'layers.0.\1'),    # layers 1-9
        (r'^layers.0.(1[0-7]_)',    r'layers.0.\1'),    # layers 10-17
        (r'^layers.0.(\d{2}_)',     r'layers.1.\1'),    # remaining layers (18-23)
    ]

    def __init__(self, num_classes=20, input_channels=3, anchors=[(2.6442, 3.4629), (6.3855, 8.01888), (10.11174, 16.19784), (18.94224, 9.68106), (22.4728, 20.0142)]):
        super().__init__()
        if not isinstance(anchors, Iterable) and not isinstance(anchors[0], Iterable):
            raise TypeError('Anchors need to be a 2D list of numbers')

        # Parameters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.anchors = anchors

        # Network
        layer_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 1, 1)),
                ('2_max',           torch.nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
                ('4_max',           torch.nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('6_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0)),
                ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('8_max',           torch.nn.MaxPool2d(2, 2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ('10_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                ('11_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ('12_max',          torch.nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ('14_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                ('15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ('16_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                ('17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
            ]),

            # Sequence 1 : input = sequence0
            OrderedDict([
                ('18_max',          torch.nn.MaxPool2d(2, 2)),
                ('19_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ('20_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                ('21_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ('22_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                ('23_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ('24_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 1024, 3, 1, 1)),
                ('25_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 1024, 3, 1, 1)),
                ('26_upsample',     torch.nn.Upsample(scale_factor=2, mode='nearest'))
            ]),

            # Sequence 2 : input = sequence0
            OrderedDict([
                ('27_convbatch',    lnn.layer.Conv2dBatchReLU(512, 4*64, 1, 1, 0)),
            ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('28_convbatch',    lnn.layer.Conv2dBatchReLU((4*64)+1024, 1024, 3, 1, 1)),
                ('29_conv',         torch.nn.Conv2d(1024, len(self.anchors)*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = torch.nn.ModuleList([torch.nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        out0 = self.layers[0](x)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out0)
        out = self.layers[3](torch.cat((out2, out1), 1))

        return out
