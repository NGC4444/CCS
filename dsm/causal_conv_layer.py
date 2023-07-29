import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

if __name__=='__main__':
    x = torch.randint(0,100,(100,16,25)).type(torch.float)
    kernel_size = 3
    in_channels=16
    out_channels=16
    dilation = 1
    convlayer = CausalConv1d(in_channels,out_channels,kernel_size,dilation=dilation)
    print(convlayer(x).shape)
    print(convlayer(x))