import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import torch.nn as nn
import os
from PIL import Image

class PlacesCNN(nn.Module):
    def __init__(self, arch = 'resnet50'):
        super(PlacesCNN, self).__init__()
        model_file = '%s_places365.pth.tar' % arch
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        modules = list(model.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, images, features='pool'):
        out = self.resnet(images)
        if features == 'pool':
            out = self.adaptive_pool(out)
            out = torch.reshape(out, (out.size(0),out.size(1)))
            return out
        elif features == 'spatial':
            return out