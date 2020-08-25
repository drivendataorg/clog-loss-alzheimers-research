import math

# import efficientnet_pytorch
# from efficientnet_pytorch_3d import EfficientNet3D
# import resnest.torch
import torch
import torch.nn as nn
import torchvision

from models import resnet, resnext, densenet, resnet2p1d  # pre_act_resnet, wide_resnet


def get_encoder(model_name):
    if 'efficientnet' in model_name:
            encoder = efficientnet_pytorch.EfficientNet.from_pretrained(model_name, in_channels=1)
            n_f = encoder._fc.in_features
            encoder._fc = nn.Identity()
    elif 'resnest' in model_name:
        encoder = getattr(resnest.torch, model_name)(pretrained=True)
        encoder.conv1[0] = nn.Conv2d(1, encoder.conv1[0].out_channels,
                                     kernel_size=encoder.conv1[0].kernel_size,
                                     stride=encoder.conv1[0].stride,
                                     padding=encoder.conv1[0].padding,
                                     bias=False,
                                 )
        n_f = encoder.fc.in_features
        encoder.fc = nn.Identity()
    else:
        encoder = getattr(torchvision.models, model_name)(pretrained=True)
        encoder.conv1 = nn.Conv2d(1, encoder.conv1.out_channels,
                                  kernel_size=encoder.conv1.kernel_size,
                                  stride=encoder.conv1.stride,
                                  padding=encoder.conv1.padding,
                                  bias=False,
                                 )
        n_f = encoder.fc.in_features
        encoder.fc = nn.Identity()
        
    return encoder, n_f


class GlobalAvgPooling(nn.Module):
    def __init__(self, dim=0):
        super(self.__class__, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        # x: [seq, batch, features]
        return x.mean(dim=self.dim)
    
    
class GlobalMaxPooling(nn.Module):
    def __init__(self, dim=0):
        super(self.__class__, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        # x: [seq, batch, features]
        return x.max(dim=self.dim)[0]


class GlobalSoftMaxPooling(nn.Module):
    def __init__(self, dim=0):
        super(self.__class__, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        # x: [seq, batch, features]
        return torch.sum(x*torch.softmax(x, dim=self.dim), dim=self.dim)


class GlobalAttentivePooling(nn.Module):
    def __init__(self, n_units, n_hid, dim=0):
        super().__init__()

        self.dim = dim
        self.attn = nn.Sequential(
            nn.Linear(n_units, n_hid),
            nn.ReLU(inplace=True),
            nn.Linear(n_hid, 1),
        )
        
    def forward(self, x):
        # x: [seq, batch, features]
        attn = self.attn(x)  # [seq, batch, 1]
        attn = torch.softmax(attn, dim=self.dim)  # [seq, batch, 1]

        return torch.sum(attn*x, dim=self.dim)  # [batch, features]


class LSTM(nn.Module):
    def __init__(self, model_name='efficientnet-b3', n_classes=1):
        super().__init__()

        self.features, n_f = get_encoder(model_name)
        
        self.n_f = n_f
        n_f2 = 2*(n_f//2)
        self.lstm = nn.LSTM(n_f, n_f2//2, bidirectional=True)

        self.classifier = nn.Linear(2*n_f2, n_classes)

    def forward(self, x):
        # x: [batch, n_frames, h, w]

        lens = [len(_x) for _x in x]
        xs = torch.cat(x, dim=0).unsqueeze(1)  # [batch*n_frames, 1, h, w]
        xs = self.features(xs)  # [batch*n_frames, features]
        xs = torch.split_with_sizes(xs, lens, dim=0)  # [batch, n_frames, features]

        xs = torch.nn.utils.rnn.pack_sequence(xs, enforce_sorted=False)  # [n_frames, batch, features]
        x, _ = self.lstm(xs)  # [seq, batch, features]

        x, l = torch.nn.utils.rnn.pad_packed_sequence(x)
        l = l.cuda()
        mask = torch.arange(x.size(0)).cuda().unsqueeze(-1).unsqueeze(-1).expand(x.size())
        l_exp = l.unsqueeze(0).unsqueeze(-1).expand(x.size())
        mask = (mask < l_exp)

        x_sum = x.sum(0) / l[:, None]  # [batch, features]

        x[~mask] = float('-inf')
        x_max = x.max(0)[0]  # [batch, features]

        x = torch.cat([
            x_sum,
            x_max,
        ], dim=-1)

        return self.classifier(x)
    
    
class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.block(x)
    
    
class Conv1dResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        
        self.block = Conv1dBlock(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        return x + self.block(x)
    
    
class CNN(nn.Module):
    def __init__(self, model_name='efficientnet-b3', n_classes=1):
        super().__init__()

        self.features, n_f = get_encoder(model_name)
        
        self.n_f = n_f
        self.lstm = nn.Sequential(
            Conv1dBlock(self.n_f, self.n_f//2, 7, 0),
            nn.MaxPool1d(2, 2),
            
            Conv1dResBlock(self.n_f//2, 3),
#             nn.MaxPool1d(2, 2),
            
            Conv1dBlock(self.n_f//2, self.n_f//4, 3, 1),
            nn.MaxPool1d(2, 2),
            
            Conv1dResBlock(self.n_f//4, 3),
#             nn.MaxPool1d(2, 2),
            
            Conv1dBlock(self.n_f//4, self.n_f//8, 3, 1),
        )

        self.classifier = nn.Linear(2*(self.n_f//8), n_classes)

    def forward(self, x):
        # x: [batch, n_frames, h, w]

        lens = [len(_x) for _x in x]
        xs = torch.cat(x, dim=0).unsqueeze(1)
        xs = self.features(xs)  # [batch, n_frames, features]
        xs = torch.split_with_sizes(xs, lens, dim=0)

        xs = torch.nn.utils.rnn.pack_sequence(xs, enforce_sorted=False)
        x, l = torch.nn.utils.rnn.pad_packed_sequence(xs)
        
        x = x.permute(1, 2, 0)  # [batch, features, seq]
        x = self.lstm(x)  # [batch, features, seq]

        x = torch.cat([
            x.mean(-1),
            x.max(-1)[0],
        ], dim=-1)

        return self.classifier(x)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
    
    
class Transformer(nn.Module):
    def __init__(self, model_name='efficientnet-b3', n_classes=1):
        super().__init__()

        self.features, n_f = get_encoder(model_name)
        
        self.n_f = n_f
        
#         self.pos = PositionalEncoding(self.n_f)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_f, nhead=8)
        self.lstm = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Linear(2*self.n_f, n_classes)

    def forward(self, x):
        # x: [batch, n_frames, h, w]

        lens = [len(_x) for _x in x]
        xs = torch.cat(x, dim=0).unsqueeze(1)
        xs = self.features(xs)  # [batch, n_frames, features]
        xs = torch.split_with_sizes(xs, lens, dim=0)

        xs = torch.nn.utils.rnn.pack_sequence(xs, enforce_sorted=False)
        x, l = torch.nn.utils.rnn.pad_packed_sequence(xs)
        l = l.cuda()
        mask = torch.arange(x.size(0)).cuda().unsqueeze(-1).unsqueeze(-1).expand(x.size())
        l_exp = l.unsqueeze(0).unsqueeze(-1).expand(x.size())
        mask = (mask < l_exp)

#         attn_mask = _generate_square_subsequent_mask(len(x)).cuda()
#         x = x*math.sqrt(self.n_f)
#         x = self.pos(x)

        # TODO add src_key_padding_mask
        x = self.lstm(x)  #, attn_mask [seq, batch, features]

        x_sum = x.sum(0) / l[:, None]  # [batch, features]

        x[~mask] = float('-inf')
        x_max = x.max(0)[0]  # [batch, features]

        x = torch.cat([
            x_sum,
            x_max,
        ], dim=-1)

        return self.classifier(x)
    

def get_model(model_name, encoder, num_classes):
    if model_name == 'lstm':
        return LSTM(encoder, num_classes)
    elif model_name == 'cnn':
        return CNN(encoder, num_classes)
    elif model_name == 'transformer':
        return Transformer(encoder, num_classes)
    elif model_name == 'cnn3d':
        if 'efficientnet' in encoder:
            return EfficientNet3D.from_name(encoder, override_params={'num_classes': num_classes}, in_channels=1)
        elif 'resnet' in encoder:
            if encoder == 'resnet':
                encoder = 'resnet34'
            return resnet.generate_model(model_depth=int(encoder.split('resnet')[-1]),
                                         n_classes=num_classes,
                                         n_input_channels=1,
                                         shortcut_type='B',
                                         conv1_t_size=7,
                                         conv1_t_stride=1,
                                         no_max_pool=False,
                                         widen_factor=1.)
        elif 'resnext' in encoder:
            return resnext.generate_model(model_depth=int(encoder.split('resnext')[-1]),
                                          n_classes=num_classes,
                                          n_input_channels=1,
                                          cardinality=32,
                                          shortcut_type='B',
                                          conv1_t_size=7,
                                          conv1_t_stride=1,
                                          no_max_pool=False)
        elif 'resnet2p1d' in encoder:
            return resnet2p1d.generate_model(model_depth=int(encoder.split('resnet2p1d')[-1]),
                                             n_classes=num_classes,
                                             n_input_channels=1,
                                             shortcut_type='B',
                                             conv1_t_size=7,
                                             conv1_t_stride=1,
                                             no_max_pool=False,
                                             widen_factor=1.)
        elif 'densenet' in encoder:
            return densenet.generate_model(model_depth=int(encoder.split('densenet')[-1]),
                                           num_classes=num_classes,
                                           n_input_channels=1,
                                           conv1_t_size=7,
                                           conv1_t_stride=1,
                                           no_max_pool=False)
        else:
            print(encoder)
            raise
    else:
        print(model_name)
        raise
