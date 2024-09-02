# mstcn retrieved from https://github.com/tobiascz/TeCNO/blob/master/models/mstcn.py
import torch.nn.functional as F
import torch.nn as nn
import hparams_all as hp
import torch as t
import copy
import matplotlib.pyplot as plt

class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 causal_conv=False,
                 kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(dilation *
                                                   (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=dilation,
                                          dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)
    

class SingleStageModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 causal_conv=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2**i,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out_classes = self.conv_out_classes(out)
        return out_classes
    
    
class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, 
                    dim, num_classes, causal_conv):
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_f_maps = num_f_maps
        self.dim = dim
        self.num_classes = num_classes
        self.causal_conv = causal_conv

        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes,
                                       causal_conv=self.causal_conv)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes,
                                 self.num_classes,
                                 causal_conv=self.causal_conv))
            for s in range(self.num_stages - 1)
        ])
        self.smoothing = False

    def forward(self, x):
        out_classes = self.stage1(x)
        outputs_classes = out_classes.unsqueeze(0)
        for s in self.stages:
            out_classes = s(F.softmax(out_classes, dim=1))
            outputs_classes = t.cat(
                (outputs_classes, out_classes.unsqueeze(0)), dim=0)
        return outputs_classes
    

class ResBlock(nn.Module):
    def __init__(self, n_input):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels = n_input, 
            out_channels = n_input, padding = 1, kernel_size = 3)
        self.conv2 = nn.Conv1d(in_channels = n_input, 
            out_channels = n_input, padding = 1, kernel_size = 3)
        
        self.bn1 = nn.BatchNorm1d(n_input)
        self.bn2 = nn.BatchNorm1d(n_input)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + res
    

class AudioBlock(nn.Module):
    def __init__(self, num_resblocks):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels = hp.dim, 
            out_channels = hp.compute_dim, padding = 1, kernel_size = 3,
            stride = 2)
        self.bn_in = nn.BatchNorm1d(hp.compute_dim)

        self.layers = nn.ModuleList()
        for i in range(num_resblocks):
            self.layers.append(ResBlock(hp.compute_dim))

        self.conv_out = nn.Conv1d(in_channels = hp.compute_dim, 
            out_channels = hp.compute_dim, padding = 1, kernel_size = 3, 
            stride = 2)

        self.avg_pool = nn.AvgPool1d(hp.wav2vec_dim // 2 // 2)
        
        self.relu = nn.ReLU()
    
    def forward(self, x): # x shape (batch, time, feat)
        x = t.transpose(x, 1, 2) #(B, 1024, 149) (batch, feat, time)
        x = self.conv_in(x) #(B, dim, 75)
        x = self.bn_in(x)
        x = self.relu(x)
        for f in self.layers: x = f(x) #(B, compute_dim, 75)
        x = self.conv_out(x) #(B, compute_dim, 37)
        x = self.avg_pool(x) #(B, compute_dim, 1)
        return t.permute(x, (0, 2, 1))
    
class AudioBlock2(nn.Module):
    def __init__(self, num_resblocks):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels = hp.mfcc_dim2, 
            out_channels = hp.compute_dim, padding = 1, kernel_size = 3,
            stride = 2)
        self.bn_in = nn.BatchNorm1d(hp.compute_dim)

        self.layers = nn.ModuleList()
        for i in range(num_resblocks):
            self.layers.append(ResBlock(hp.compute_dim))

        self.conv_out = nn.Conv1d(in_channels = hp.compute_dim, 
            out_channels = hp.compute_dim, padding = 1, kernel_size = 3, 
            stride = 2)

        self.avg_pool = nn.AvgPool1d(hp.mfcc_dim // 2 // 2)
        
        self.relu = nn.ReLU()
    
    def forward(self, x): # x shape (batch, time, feat)
        x = t.transpose(x, 1, 2) #(B, 1024, 149) (batch, feat, time)
        x = self.conv_in(x) #(B, dim, 75)
        x = self.bn_in(x)
        x = self.relu(x)
        for f in self.layers: x = f(x) #(B, compute_dim, 75)
        x = self.conv_out(x) #(B, compute_dim, 37)
        x = self.avg_pool(x) #(B, compute_dim, 1)
        return t.permute(x, (0, 2, 1))
    

class TranscriptBlock(nn.Module): # Fully Connected layer (paper)
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(hp.tr_dim, hp.compute_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.tanh(x)
        x = x.unsqueeze(1) # shape 512, 1, 256
        return x


class GMU(nn.Module):
    def __init__(self):
        super().__init__()

        self.Wz1 = nn.Linear(hp.compute_dim, hp.compute_dim, bias = False)
        self.Wt1 = nn.Linear(hp.compute_dim, hp.compute_dim, bias = False)

        self.Wz2 = nn.Linear(2 * hp.compute_dim, hp.compute_dim, bias = False)
        self.Wt2 = nn.Linear(hp.compute_dim, hp.compute_dim, bias = False)

        self.Wz3 = nn.Linear(3 * hp.compute_dim, hp.compute_dim, bias = False)
        self.Wt3 = nn.Linear(hp.compute_dim, hp.compute_dim, bias = False)

        self.tanh = nn.Tanh()
        self.sigma = nn.Sigmoid()

    def forward(self, c1, c2, c3):
        z1 = self.sigma(self.Wz1(c1))
        h1 = self.tanh(self.Wt1(c1))

        z2 = self.sigma(self.Wz2(t.cat((c1, c2), dim = 2)))
        h2 = self.tanh(self.Wt2(c2))

        z3 = self.sigma(self.Wz3(t.cat((c1, c2, c3), dim = 2)))
        h3 = self.tanh(self.Wt3(c3))
        
        return (h1 * z1) + (h2 * z2) + (h3 * z3)

class M3X1(nn.Module):
    def __init__(self):
        super().__init__()

        self.audio = AudioBlock(hp.num_resblocks)

        self.mfcc = AudioBlock2(hp.num_resblocks)

        self.gmu = GMU()

        self.mstcn = MultiStageModel(hp.num_stages,
            hp.num_layers_speech, hp.num_f_maps_speech, hp.compute_dim_speech, 
            hp.num_classes_speech, hp.causal_conv)
            
        self.logsoftmax = t.nn.LogSoftmax(dim = 1)

        # self.relu = nn.LeakyReLU()

    def forward(self, c1, c2, c3, h):
        
        c1 = self.audio(c1) #(B, 1, 128)  --> (B, 1, 256)
        c2 = self.audio(c2)
        c3 = self.mfcc(c3)

        h = h.repeat(c1.shape[0], 1) # (B, 128)
        h = h.unsqueeze(1) # (B, 1, 128)

        f = self.gmu(c1, c2, c3) #(B, 1, 128)    # t1, t2
        f += h
        f = t.permute(f, (1, 2, 0)) #(1, 128, B)
        
        m = self.mstcn(f) #(n_stage, 1, n_class, B)
        m = m.squeeze(dim = 1) #(n_stage, n_class, B)
        m = t.permute(m, (2,1,0)) #(B, n_class, n_stage)
        
        return self.logsoftmax(m) #(B, n_class, n_stages)