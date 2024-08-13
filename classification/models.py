import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

from deform.DeformableBlock3D import DeformConv3d
"""
The structure of C3D-Light
"""
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class C3D_Light(nn.Module):

    def __init__(self,input_shape=(10, 181, 180), dropout=0.5,num_classes=2, n_filters=64, deform_idx=[], pretrained=False):
        super(C3D_Light, self).__init__()

        self.deform_idx = deform_idx

        self.model = nn.Sequential()

        self.model.add_module("conv3d_1",
                              nn.Conv3d(1, n_filters, kernel_size=(3,3,3), padding=(1, 1, 1)))  # n * (x/s) * (y/s) * (z/s)
        self.model.add_module("max_pool3d_1", nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

        self.model.add_module("conv3d_2", nn.Conv3d(n_filters, n_filters * 2, kernel_size=(3, 3, 3),
                                                    padding=(1, 1, 1)))  # n * (x/s) * (y/s) * (z/s)
        self.model.add_module("max_pool3d_2", nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        self.model.add_module("conv3d_3", nn.Conv3d(n_filters * 2, n_filters * 2, kernel_size=(3, 3, 3),
                                                    padding=(1, 1, 1)))  # n * (x/s) * (y/s) * (z/s)
        self.model.add_module("max_pool3d_3", nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))


        self.model.add_module("conv3d_4", nn.Conv3d(n_filters * 2, n_filters * 4, kernel_size=(3, 3, 3),
                                                        padding=(1, 1, 1)))  # n * (x/s) * (y/s) * (z/s)
        self.model.add_module("max_pool3d_4", nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))


        self.model.add_module("conv3d_5",
                                  nn.Conv3d(n_filters * 4, n_filters * 4, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
        self.model.add_module("max_pool3d_5",
                              nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1)))


        self.__init_weight()


        # n_flatten_units = 2 * n_filters * np.prod(np.array(input_shape) // (2 * stride))
        # print(n_flatten_units)
        self.model.add_module("flatten_1", Flatten())
        self.model.add_module("fully_conn_1", nn.Linear(256 * 1 * 6 * 6, 2048))
        self.model.add_module("activation_1", nn.ReLU(inplace=True))
        self.model.add_module("dropout_1", nn.Dropout(dropout))

        self.model.add_module("fully_conn_2", nn.Linear(2048, 2048))
        self.model.add_module("activation_2", nn.ReLU(inplace=True))
        self.model.add_module("dropout_2", nn.Dropout(dropout))

        self.model.add_module("fully_conn_3", nn.Linear(2048, num_classes))

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        return self.model(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



