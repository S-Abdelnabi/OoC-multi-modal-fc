from torchvision.models import resnet152, resnet101
import torch.nn as nn
import torch 

class EncoderCNN(nn.Module):
    def __init__(self, resnet_arch = 'resnet101'):
        super(EncoderCNN, self).__init__()
        if resnet_arch == 'resnet101':
            resnet = resnet101(pretrained=True)
        elif resnet_arch == 'resnet152':
            resnet = resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, images, features='pool'):
        out = self.resnet(images)
        if features == 'pool':
            out = self.adaptive_pool(out)
            out = torch.reshape(out, (out.size(0),out.size(1)))
        return out
