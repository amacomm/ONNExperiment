import torch
import torch.fft as fft
import torch.nn as nn
import numpy as np

class CONFIG():
    def __init__(self, array_size: int = 512,
                 pixel_size: float = 18e-6,
                 wavelength: float = 532e-9,
                 image_size: int = 56,
                 distance: float = 0.3):
        self.array_size: int = array_size
        self.pixel_size: float = pixel_size
        self.wavelength: float = wavelength
        self.K: float = 2*np.pi/self.wavelength # VolnovoiVector
        self.aperture_size: float = self.array_size * self.pixel_size
        self.image_size: int = image_size
        self.distance: float = distance

        size = self.image_size/100
        aa = size*4
        self.coords = torch.tensor([[-aa, aa],
                                [0,	aa],
                                [aa,	aa],
                                [-1.5*aa,	0],
                                [-0.5*aa,	0],
                                [0.5*aa,	0],
                                [1.5*aa,	0],
                                [-aa,	-aa],
                                [0,	-aa],
                                [aa,	-aa]])

def sum_func(image: torch.tensor, k = 0) -> torch.Tensor:
    return torch.sum(image, dim = (2, 3))

def max_func(image: torch.tensor, k_size: int) -> torch.Tensor:
    return nn.MaxPool2d(kernel_size = k_size)(image)

class ONN(nn.Module):
    def __init__(self, config,
                 phase: None | torch.Tensor = None,
                 out_function = sum_func): 
        super(ONN, self).__init__()
        
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
        self.U = torch.roll(xx, (int(self.array_size/2), int(self.array_size/2)), dims = (0, 1))
        self.p = torch.sqrt(-self.U+self.K**2)

        coords = config.coords
        l = torch.linspace(-config.array_size/100,config.array_size/100,config.array_size)
        Y, X = torch.meshgrid(l, l, indexing='ij')
        
        self.mask = torch.stack([(X > coords[x][0]-self.image_size/200) * (X < coords[x][0]+self.image_size/200) * (Y > coords[x][1]-self.image_size/200) * (Y < coords[x][1]+self.image_size/200) for x in range(10)])

        self.maxpool = nn.MaxPool2d(kernel_size = self.array_size)
        self.maxpool2 = nn.MaxPool1d(kernel_size = 10)
        self.dropout = nn.Dropout(0.5)
        self.phase = None
        if(phase != None):
            self.phase = phase
        else:
            self.phase = nn.Parameter(torch.rand(self.array_size, self.array_size, dtype=torch.float))
        self.phase2 = nn.Parameter(torch.rand(self.array_size, self.array_size, dtype=torch.float))
        self.zero = nn.ZeroPad2d(int((self.array_size - self.image_size)/2))
        self.softmax = nn.Softmax(dim=1)
        self.one = torch.ones((512, 512))
        self.function = out_function
        
    def propagation(self, field, z):
        eta = torch.exp(1j*z*self.p)
        res = fft.ifft2(fft.fft2(field) * eta)
        res = res * self.dropout(self.one)
        return res
    
    def DOE(self, i):
        if 1==i:
            return torch.exp(1j*self.phase)
        return torch.exp(1j*self.phase2)
    
    def forward(self, x):
         x = x/(torch.sum(x**2, dim = (1, 2, 3))[:, None, None, None]**0.5)*np.sqrt(500)
         x = self.zero(x)
         x = self.propagation(x, self.distance)
         X = x * self.DOE(1)
         X = self.propagation(X, self.distance)
         X = X * self.DOE(2)
         X = self.propagation(X, self.distance)
         res = X * self.mask
         res = torch.abs(res)
         res = (res**2)
         #res = self.dropout(res)
         res=self.function(res, self.array_size)
         return X, res