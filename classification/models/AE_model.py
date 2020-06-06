import torch
import torch.nn as nn
import torch.nn.functional as F
class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, skip=False, **kwargs):
        super(DownBlock, self).__init__()
        self.skip = skip
        self.block = nn.ModuleDict({
            '1_convx': nn.Conv3d(in_channels=c_in,
                              out_channels=c_out,
                              kernel_size=(kwargs['conv_k'], 1, 1),
                              stride=(kwargs['conv_s'],1,1),
                              padding=(kwargs['conv_pad'],0,0),
                          ),
            '2_convy': nn.Conv3d(in_channels=c_out,
                              out_channels=c_out,
                              kernel_size=(1, kwargs['conv_k'], 1),
                              stride=(1,kwargs['conv_s'],1),
                              padding=(0,kwargs['conv_pad'],0),
                          ),
            '3_convz': nn.Conv3d(in_channels=c_out,
                              out_channels=c_out,
                              kernel_size=(1, 1, kwargs['conv_k']),
                              stride=(1,1,kwargs['conv_s']),
                              padding=(0,0,kwargs['conv_pad']),
                          ),
            '4_pooling': nn.MaxPool3d(kernel_size=kwargs['maxpool_k'], stride=kwargs['maxpool_s'])
        })
        if kwargs['batch_norm']:
            self.block.update({'5_batch_norm': nn.BatchNorm3d(c_out)})
        if kwargs['act'] == 'l_relu':
            self.block.update({'6_act': nn.LeakyReLU()})
            self.init_gain = nn.init.calculate_gain('leaky_relu', 0.01)
        else:  ##elif
            self.block.update({'6_act': nn.ReLU()})
            self.init_gain = nn.init.calculate_gain('relu')
        self.init_weights()
        
    def init_weights(self):
        for _, m in self.block.items():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data, gain=self.init_gain)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
#         x_before_pool = None
        _,_,D,H,W = x.shape
        shape_before_pool = (D,H,W) #if not even size
        for key, module in sorted(self.block.items()):
            #             if key == 'poolig' and self.skip == True: # if use skip conection
            #                 x_before_pool = x
            x = module(x)
        return x, shape_before_pool  # ,x_before_pool


class UpBlock(nn.Module):
    def __init__(self, c_in, c_out, skip=False, **kwargs):
        super(UpBlock, self).__init__()
        self.skip = skip
        self.block = nn.ModuleDict()
        if kwargs['up'] == 'transpose_conv':
            self.block.update({'1_upsample': nn.ConvTranspose3d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=kwargs['scale'],
                stride=kwargs['scale'],
                padding=kwargs['t_conv_pad'],
            )})
        else:
            self.block.update({'1_upsample': nn.Upsample(
                scale_factor=kwargs['scale'],
                mode=kwargs['scale_mode'],
            )})
        self.block.update({'2_convx': nn.Conv3d(in_channels=c_in,
                                             out_channels=c_out,
                                             kernel_size=(kwargs['conv_k'],1,1),
                                             stride=(kwargs['conv_s'],1,1),
                                             padding=(kwargs['conv_pad'],0,0),
                                             ),
                           '3_convy': nn.Conv3d(in_channels=c_out,
                                             out_channels=c_out,
                                             kernel_size=(1,kwargs['conv_k'],1),
                                             stride=(1,kwargs['conv_s'],1),
                                             padding=(0,kwargs['conv_pad'],0),
                                             ),
                           '4_convz': nn.Conv3d(in_channels=c_out,
                                             out_channels=c_out,
                                             kernel_size=(1,1,kwargs['conv_k']),
                                             stride=(1,1,kwargs['conv_s']),
                                             padding=(0,0,kwargs['conv_pad']),
                                             ),
                          })
        if kwargs['batch_norm']:
            self.block.update({'5_batch_norm': nn.BatchNorm3d(c_out)})
        if kwargs['act'] == 'l_relu':
            self.block.update({'6_act': nn.LeakyReLU()})
            self.init_gain = nn.init.calculate_gain('leaky_relu', 0.01)
        else:  
            self.block.update({'6_act': nn.ReLU()})
            self.init_gain = nn.init.calculate_gain('relu')
        self.init_weights()
        
    def init_weights(self):
        for _, m in self.block.items():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data, gain=self.init_gain)
                nn.init.constant_(m.bias.data, 0)


    def forward(self, x, shape_before_pool=None, x_before_pool=None):
        for key, module in sorted(self.block.items()):
            #             if key == 'conv' and self.skip == True: # if use skip conection
            #                 assert x_before_pool == None, 'wrang skip_map'
            #                 x = torch.cat([x, x_before_pool], dim=1)
            x = module(x)
            if key =='1_upsample' and (shape_before_pool[0] > x.shape[2] or 
                               shape_before_pool[1] > x.shape[3] or 
                               shape_before_pool[2] > x.shape[4]): #not even size
                x = F.interpolate(x,(shape_before_pool[0],shape_before_pool[1], shape_before_pool[2]))
        return x


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.encode = nn.ModuleList()
        if kwargs['reduce_size']:
            self.encode.append(nn.Conv3d(1, 1, kernel_size=4, stride=4, padding=0))
        for i in range(kwargs['deapth']):
            self.encode.append(
                DownBlock(c_in=kwargs['chanels'][i],
                          c_out=kwargs['chanels'][i + 1],
                          skip=kwargs['skip_map'][i],
                          **kwargs['down_block_kwargs']
                          ))

    def forward(self, x):
#         skip_list = []
        size_list = []
        for module in self.encode:
            x, size = module(x)
            size_list.append(size)
#             skip_list.append(before_pooling)
        return x, size_list


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        self.decode = nn.ModuleList()
        for i in range(kwargs['deapth']):
            self.decode.append(
                UpBlock(c_in=kwargs['chanels'][i],
                        c_out=kwargs['chanels'][i + 1],
                        skip=kwargs['skip_map'][i],
                        **kwargs['up_block_kwargs']
                        ))
        if kwargs['reduce_size']:
            self.decode.append(nn.ConvTranspose3d(1, 1, kernel_size=4, stride=4, padding=0))
        self.vox = nn.Conv3d(in_channels=1, out_channels= 1,
                                             kernel_size= 3,
                                             stride= 1,
                                             padding=1,
                                             )
    def forward(self, x, size_list):
        size_list.reverse()
        for i, module in enumerate(self.decode):
            x = module(x, size_list[i])
        x = self.vox(x)
        return x


class AE(nn.Module):
    def __init__(self, **kwargs):
        super(AE, self).__init__()

        if kwargs['is_skip']:
            skip_map = kwargs['skip_map']
            assert len(skip_map) < kwargs['deapth'], 'skip map len shold mutch deapth'
        else:
            skip_map = [False for i in range(kwargs['deapth'])]

        chanels = [kwargs['c_in']]
        c = kwargs['c_base']
        for i in range(kwargs['deapth']):
            chanels.append(c)
            c = kwargs['inc_size'] * c

        encoder_kwargs = {
            'deapth': kwargs['deapth'],
            'chanels': chanels,
            'skip_map': skip_map,
            'reduce_size': kwargs['reduce_size'],
            'down_block_kwargs': kwargs['down_block_kwargs']
        }
        self.enc = Encoder(**encoder_kwargs)

        decoder_kwargs = {
            'deapth': kwargs['deapth'],
            'chanels': chanels[::-1],  # if use skip conection modify in_chanels for up block
            'skip_map': skip_map[::-1],
            'reduce_size': kwargs['reduce_size'],
            'up_block_kwargs': kwargs['up_block_kwargs']
        }
        self.dec = Decoder(**decoder_kwargs)

    def forward(self, x):
        x, size_list = self.enc(x)
        x = self.dec(x, size_list)
        return x


class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()

        self.disc = nn.ModuleDict({
            '1_convx': nn.Conv3d(in_channels=kwargs['c_in'],
                              out_channels=kwargs['c_out'],
                              kernel_size=(kwargs['conv_k'], 1, 1),
                              stride=(kwargs['conv_s'],1,1),
                              padding=(kwargs['conv_pad'],0,0),
                          ),
            '2_convy': nn.Conv3d(in_channels=kwargs['c_out'],
                              out_channels=kwargs['c_out'],
                              kernel_size=(1, kwargs['conv_k'], 1),
                              stride=(1,kwargs['conv_s'],1),
                              padding=(0,kwargs['conv_pad'],0),
                          ),
            '3_convz': nn.Conv3d(in_channels=kwargs['c_out'],
                              out_channels=kwargs['c_out'],
                              kernel_size=(1, 1, kwargs['conv_k']),
                              stride=(1,1,kwargs['conv_s']),
                              padding=(0,0,kwargs['conv_pad']),
                          ),
            '4_flat': nn.Flatten(),
            '5_l1': nn.Linear(kwargs['l_in'], kwargs['l_out'])
        })

        if kwargs['batch_norm']:
            self.disc.update({'6_batch_norm': nn.BatchNorm1d(kwargs['l_out'])})
        if kwargs['act'] == 'l_relu':
            self.disc.update({'7_act': nn.LeakyReLU()})
            self.init_gain = nn.init.calculate_gain('leaky_relu', 0.01)
        else:
            self.disc.update({'7_act': nn.ReLU()})
            self.init_gain = nn.init.calculate_gain('relu')
        self.disc.update({'8_drop': nn.Dropout(kwargs['p_drop'])})
        self.disc.update({'9_l_f': nn.Linear(kwargs['l_out'], kwargs['n_domains'])})
        
        self.init_weights()
        
    def init_weights(self):
        for key, m in self.disc.items():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data, gain=self.init_gain)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        for key, module in sorted(self.disc.items()):
            x = module(x)
        return x

class Classificator(nn.Module):
    def __init__(self, **kwargs):
        super(Classificator, self).__init__()
        self.clf = nn.ModuleDict({
            '1_convx': nn.Conv3d(in_channels=kwargs['c_in'],
                              out_channels=kwargs['c_out'],
                              kernel_size=(kwargs['conv_k'], 1, 1),
                              stride=(kwargs['conv_s'],1,1),
                              padding=(kwargs['conv_pad'],0,0),
                          ),
            '2_convy': nn.Conv3d(in_channels=kwargs['c_out'],
                              out_channels=kwargs['c_out'],
                              kernel_size=(1, kwargs['conv_k'], 1),
                              stride=(1,kwargs['conv_s'],1),
                              padding=(0,kwargs['conv_pad'],0),
                          ),
            '3_convz': nn.Conv3d(in_channels=kwargs['c_out'],
                              out_channels=kwargs['c_out'],
                              kernel_size=(1, 1, kwargs['conv_k']),
                              stride=(1,1,kwargs['conv_s']),
                              padding=(0,0,kwargs['conv_pad']),
                          ),
            '4_flat': nn.Flatten(),
            '5_l1': nn.Linear(kwargs['l_in'], kwargs['l_out'])
        })

        if kwargs['batch_norm']:
            self.clf.update({'6_batch_norm': nn.BatchNorm1d(kwargs['l_out'])})
        if kwargs['act'] == 'l_relu':
            self.clf.update({'7_act': nn.LeakyReLU()})
            self.init_gain = nn.init.calculate_gain('leaky_relu', 0.01)
        else:
            self.clf.update({'7_act': nn.ReLU()})
            self.init_gain = nn.init.calculate_gain('relu')
        self.clf.update({'8_drop': nn.Dropout(kwargs['p_drop'])})
        self.clf.update({'9_l_f': nn.Linear(kwargs['l_out'], kwargs['n_class'])})
        
        self.init_weights()
        
    def init_weights(self):
        for key, m in self.clf.items():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data, gain=self.init_gain)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        for key, module in sorted(self.clf.items()):
            x = module(x)
        return x
