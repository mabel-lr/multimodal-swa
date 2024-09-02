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
    


class M3X1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.xray_in = nn.Conv1d(in_channels = hp.dim, 
            out_channels = hp.compute_dim_x, kernel_size = 1)

        self.mstcn = MultiStageModel(hp.num_stages,
            hp.num_layers_x, hp.num_f_maps_x, hp.compute_dim_x, 
            hp.num_classes_x, hp.causal_conv)
            
        self.logsoftmax = t.nn.LogSoftmax(dim = 1)
        self.relu = t.nn.LeakyReLU()

    def forward(self, x, log, h):
        l = t.repeat_interleave(log[:,1:], repeats=64, dim=1)
        concat = t.cat((x,l), dim = 1 )
        x = concat

        x = x.unsqueeze(2) #(B, 1024, 1)
        x = self.xray_in(x) #(B, 128, 1)  --> (B, 256, 1)
        x = self.relu(x)
        x = t.permute(x, (0, 2, 1)) #(B, 1, 128)        

        h = h.repeat(x.shape[0], 1) # (B, 128)
        h = h.unsqueeze(1) # (B, 1, 128)

        x += h
        x = t.permute(x, (1, 2, 0)) #(1, 128, B)
        
        m = self.mstcn(x) #(n_stage, 1, n_class, B)
        m = m.squeeze(dim = 1) #(n_stage, n_class, B)
        m = t.permute(m, (2,1,0)) #(B, n_class, n_stage)
        
        return self.logsoftmax(m) #(B, n_class, n_stages)