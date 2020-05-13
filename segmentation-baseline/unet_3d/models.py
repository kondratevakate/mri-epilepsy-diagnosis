import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class VoxResNet(nn.Module):
    def __init__(self, input_shape=(128, 128, 128), num_classes=2, n_filters=32, stride=2, n_blocks=3,
                 n_flatten_units=None, dropout=0, n_fc_units=128):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential()

        self.model.add_module("conv3d_1", nn.Conv3d(1, n_filters, kernel_size=3, padding=1, stride=stride)) # n * (x/s) * (y/s) * (z/s)
        self.model.add_module("batch_norm_1", nn.BatchNorm3d(n_filters))
        self.model.add_module("activation_1", nn.ReLU(inplace=True))
        self.model.add_module("conv3d_2", nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1)) # n * (x/s) * (y/s) * (z/s)
        self.model.add_module("batch_norm_2", nn.BatchNorm3d(n_filters))
        self.model.add_module("activation_2", nn.ReLU(inplace=True))

#         1
        self.model.add_module("conv3d_3", nn.Conv3d(n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2)) # 2n * (x/2s) * (y/2s) * (z/2s)
        self.model.add_module("block_1", BasicBlock(2 * n_filters, 2 * n_filters))
        self.model.add_module("block_2", BasicBlock(2 * n_filters, 2 * n_filters))
        self.model.add_module("batch_norm_3", nn.BatchNorm3d(2 * n_filters))
        self.model.add_module("activation_3", nn.ReLU(inplace=True))

#         2
        if n_blocks >= 2:
            self.model.add_module("conv3d_4", nn.Conv3d(2 * n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2)) # 2n * (x/4s) * (y/4s) * (z/4s)
            self.model.add_module("block_3", BasicBlock(2 * n_filters, 2 * n_filters))
            self.model.add_module("block_4", BasicBlock(2 * n_filters, 2 * n_filters))
            self.model.add_module("batch_norm_4", nn.BatchNorm3d(2 * n_filters))
            self.model.add_module("activation_4", nn.ReLU(inplace=True))

#         3
        if n_blocks >= 3:
            self.model.add_module("conv3d_5", nn.Conv3d(2 * n_filters, 4 * n_filters, kernel_size=3, padding=1, stride=2)) # 4n * (x/8s) * (y/8s) * (z/8s)
            self.model.add_module("block_5", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("block_6", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("batch_norm_5", nn.BatchNorm3d(4 * n_filters))
            self.model.add_module("activation_5", nn.ReLU(inplace=True))

#         4
        if n_blocks >= 4:
            self.model.add_module("conv3d_6", nn.Conv3d(4 * n_filters, 4 * n_filters, kernel_size=3, padding=1, stride=2)) # 4n * (x/16s) * (y/16s) * (z/16s)
            self.model.add_module("block_7", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("block_8", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("batch_norm_6", nn.BatchNorm3d(4 * n_filters))
            self.model.add_module("activation_6", nn.ReLU(inplace=True))

#         self.model.add_module("max_pool3d_1", nn.MaxPool3d(kernel_size=3)) # (b/2)n * (x/(2^b)sk) * (y/(2^b)sk) * (z/(2^b)sk) ?

        if n_flatten_units is None:
            n_flatten_units = 4 * n_filters * np.prod(np.array(input_shape) // (2 ** n_blocks * stride))
        #         print(n_flatten_units)
        
        self.model.add_module("flatten_1", Flatten())
        self.model.add_module("fully_conn_1", nn.Linear(n_flatten_units, n_fc_units))
        self.model.add_module("activation_6", nn.ReLU(inplace=True))
        self.model.add_module("dropout_1", nn.Dropout(dropout))

        self.model.add_module("fully_conn_2", nn.Linear(n_fc_units, num_classes))

    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):
    def __init__(self, input_shape=(64, 76, 48), n_filters=16, n_blocks=3, stride=1, n_fc_units=128):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential()

        self.model.add_module("conv3d_1", nn.Conv3d(1, n_filters, kernel_size=3, stride=stride, padding=1))  # n * x * y
        self.model.add_module("batch_norm_1", nn.BatchNorm3d(n_filters))
        self.model.add_module("activation_1", nn.ReLU(inplace=True))
        self.model.add_module("conv3d_2", nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1))  # n * x * y
        self.model.add_module("batch_norm_2", nn.BatchNorm3d(n_filters))
        self.model.add_module("activation_2", nn.ReLU(inplace=True))
        self.model.add_module("max_pool3d_1", nn.MaxPool3d(kernel_size=2))  # n * (x/2) * (y/2)

        if n_blocks >= 2:
            self.model.add_module("conv3d_3",
                                  nn.Conv3d(n_filters, 2 * n_filters, kernel_size=3, padding=1))  # 2n * (x/2) * (y/2)
            self.model.add_module("batch_norm_3", nn.BatchNorm3d(2 * n_filters))
            self.model.add_module("activation_3", nn.ReLU(inplace=True))
            self.model.add_module("conv3d_4", nn.Conv3d(2 * n_filters, 2 * n_filters, kernel_size=3,
                                                        padding=1))  # 2n * (x/2) * (y/2)
            self.model.add_module("batch_norm_4", nn.BatchNorm3d(2 * n_filters))
            self.model.add_module("activation_4", nn.ReLU(inplace=True))
            self.model.add_module("max_pool3d_2", nn.MaxPool3d(kernel_size=2))  # 2n * (x/4) * (y/4)

        if n_blocks >= 3:
            self.model.add_module("conv3d_5", nn.Conv3d(2 * n_filters, 4 * n_filters, kernel_size=3,
                                                        padding=1))  # 4n * (x/4) * (y/4)
            self.model.add_module("batch_norm_5", nn.BatchNorm3d(4 * n_filters))
            self.model.add_module("activation_5", nn.ReLU(inplace=True))
            self.model.add_module("conv3d_6", nn.Conv3d(4 * n_filters, 4 * n_filters, kernel_size=3,
                                                        padding=1))  # 4n * (x/4) * (y/4)
            self.model.add_module("batch_norm_6", nn.BatchNorm3d(4 * n_filters))
            self.model.add_module("activation_6", nn.ReLU(inplace=True))
            self.model.add_module("max_pool3d_3", nn.MaxPool3d(kernel_size=2))  # 4n * (x/8) * (y/8)

        if n_blocks >= 4:
            self.model.add_module("conv3d_7", nn.Conv3d(4 * n_filters, 8 * n_filters, kernel_size=3,
                                                        padding=1))  # 8n * (x/8) * (y/8)
            self.model.add_module("batch_norm_7", nn.BatchNorm3d(8 * n_filters))
            self.model.add_module("activation_7", nn.ReLU(inplace=True))
            self.model.add_module("conv3d_8", nn.Conv3d(8 * n_filters, 8 * n_filters, kernel_size=3,
                                                        padding=1))  # 8n * (x/8) * (y/8)
            self.model.add_module("batch_norm_8", nn.BatchNorm3d(8 * n_filters))
            self.model.add_module("activation_8", nn.ReLU(inplace=True))
            self.model.add_module("max_pool3d_4", nn.MaxPool3d(kernel_size=2))  # 8n * (x/16) * (y/16)

        self.model.add_module("flatten_1", Flatten())

        if n_blocks == 1:
            self.model.add_module("fully_conn_1", nn.Linear(
                n_filters * (input_shape[0] // (2 * stride)) * (input_shape[1] // (2 * stride)) * (
                            input_shape[2] // (2 * stride)), n_fc_units))
        if n_blocks == 2:
            self.model.add_module("fully_conn_1", nn.Linear(
                2 * n_filters * (input_shape[0] // (4 * stride)) * (input_shape[1] // (4 * stride)) * (
                            input_shape[2] // (4 * stride)), n_fc_units))
        if n_blocks == 3:
            self.model.add_module("fully_conn_1", nn.Linear(
                4 * n_filters * (input_shape[0] // (8 * stride)) * (input_shape[1] // (8 * stride)) * (
                            input_shape[2] // (8 * stride)), n_fc_units))
        if n_blocks == 4:
            self.model.add_module("fully_conn_1", nn.Linear(
                8 * n_filters * (input_shape[0] // (16 * stride)) * (input_shape[1] // (16 * stride)) * (
                            input_shape[2] // (16 * stride)), n_fc_units))

        self.model.add_module("batch_norm_9", nn.BatchNorm1d(n_fc_units))
        self.model.add_module("activation_9", nn.ReLU(inplace=True))

    #         self.model.add_module("dropout_1", nn.Dropout(dropout))

    def forward(self, x):
        return self.model(x)


class ConvLSTM(nn.Module):
    def __init__(self, input_shape=(48, 64, 32), n_outputs=1,
                 hidden_size=128, n_layers=2, n_fc_units_rnn=128, dropout=0, stride=1,
                 n_filters=16, n_blocks=3, n_fc_units_cnn=128):
        super(self.__class__, self).__init__()
        self.model = CNN(input_shape, n_filters, n_blocks, stride, n_fc_units_cnn)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(n_fc_units_cnn, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, n_fc_units_rnn)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(n_fc_units_rnn, n_outputs)

    def forward(self, x):
        n_objects, seq_length = x.size()[0:2]
        x = x.contiguous().view([n_objects * seq_length] + list(x.size()[2:]))
        x = self.model(x)
        x = x.contiguous().view([n_objects, seq_length, -1])

        # Forward propagate RNN
        out, _ = self.lstm(x)

        # Decode hidden state of last time step
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out
