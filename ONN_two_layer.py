import torch
import torch.fft as fft
import torch.nn as nn
import numpy as np
from ONN_one_layer import *

class ONN2(nn.Module):
    def __init__(self, config,
                 phase1: None | torch.Tensor = None,
                 phase2: None | torch.Tensor = None,
                 out_function = sum_func): 
        super(ONN2, self).__init__()
        
        self.array_size = config.array_size
        self.pixel_size = config.pixel_size
        self.wavelength = config.wavelength
        self.K = config.K # VolnovoiVector
        self.aperture_size = config.aperture_size
        self.image_size = config.image_size
        self.distance = config.distance
        
        border = np.pi * self.array_size / self.aperture_size
        arr = torch.linspace(-border, border, self.array_size+1)[:self.array_size]
        xv, yv = torch.meshgrid(arr, arr, indexing='ij')
        xx = xv**2 + yv**2
        U = torch.roll(xx, (int(self.array_size/2), int(self.array_size/2)), dims = (0, 1))
        self.p = torch.sqrt(-U+self.K**2)

        border = np.pi * self.array_size*2 / self.aperture_size
        arr = torch.linspace(-border, border, self.array_size*2+1)[:self.array_size*2]
        xv, yv = torch.meshgrid(arr, arr, indexing='ij')
        xx = xv**2 + yv**2
        U = torch.roll(xx, (int(self.array_size), int(self.array_size)), dims = (0, 1))
        self.p_extended = torch.sqrt(-U+self.K**2)

        coords = config.coords
        l = torch.linspace(-config.array_size/100,config.array_size/100,config.array_size)
        Y, X = torch.meshgrid(l, l, indexing='ij')
        
        self.mask = torch.stack([(X > coords[x][0]-self.image_size/200) * (X < coords[x][0]+self.image_size/200) * (Y > coords[x][1]-self.image_size/200) * (Y < coords[x][1]+self.image_size/200) for x in range(10)])

        self.maxpool = nn.MaxPool2d(kernel_size = self.array_size)
        self.dropout = nn.Dropout(0.5)
        self.phase1: torch.Tensor
        self.phase2: torch.Tensor
        if(phase1 != None):
            self.phase1 = phase1
        else:
            self.phase1 = nn.Parameter(torch.rand(self.array_size, self.array_size, dtype=torch.float))
        if(phase2 != None):
            self.phase2 = phase2
        else:
            self.phase2 = nn.Parameter(torch.rand(self.array_size, self.array_size, dtype=torch.float))
        self.zero = nn.ZeroPad2d(int((self.array_size - self.image_size)/2))
        self.zero_add = nn.ZeroPad2d(int(self.array_size/2))
        self.softmax = nn.Softmax(dim=1)
        self.one = torch.ones((512, 512))
        self.function = out_function
        
    def propagation(self, field, z, p):
        eta = torch.exp(1j*z*p)
        res = fft.ifft2(fft.fft2(field) * eta)
        #res = res * self.dropout(self.one)
        return res
    
    def DOE(self, i):
        if 1==i:
            return torch.exp(1j*self.phase1)
        return torch.exp(1j*self.phase2)
    
    def forward(self, x):
         x = x/(torch.sum(x**2, dim = (1, 2, 3))[:, None, None, None]**0.5)*np.sqrt(50000)
         x = self.zero(x)
         x = self.propagation(x, self.distance, self.p)
         x = x * self.DOE(1)
         x = self.zero_add(x)
         x = self.propagation(x, self.distance, self.p_extended)
         x = x[:,:,256:768,256:768]
         X = x * self.DOE(2)
         X = self.propagation(X, self.distance, self.p)
         res = X * self.mask
         res = torch.abs(res)**2
         #res = self.dropout(res)
         res=self.function(res, self.array_size)
         return X, res