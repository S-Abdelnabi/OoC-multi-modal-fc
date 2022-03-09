from torchvision.models import resnet152, resnet101
import torch.nn as nn
import torch 
import clip
import torch.nn.functional as F

class ClipClassifier(nn.Module):
    def __init__(self, settings, clip_model=None):
        super(ClipClassifier, self).__init__()
        self.clip = clip_model
        self.pdrop = settings['pdrop']
        self.classifier = nn.Linear(512,1).half()
        
    def forward(self,qimage_clip_processed, qtext_clip_tokenized):
        encoded_img = self.clip.encode_image(qimage_clip_processed)   
        encoded_text = self.clip.encode_text(qtext_clip_tokenized)  
        
        encoded_img = encoded_img / encoded_img.norm(dim=-1, keepdim=True) 
        encoded_text = encoded_text / encoded_text.norm(dim=-1, keepdim=True)

        joint_features = encoded_img*encoded_text
        joint_features = F.dropout(joint_features, p=self.pdrop)
        consis_out = self.classifier(joint_features)
        return consis_out 
        