import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import json
from urllib.parse import urlparse
from PIL import Image
import os 
import clip 

class NewsClipDataset(Dataset):
    def __init__(self, visual_news_root_dir, news_clip_root_dir, split, transform):
        self.visual_news_root_dir = visual_news_root_dir
        self.news_clip_root_dir = news_clip_root_dir
        self.transform = transform
        self.visual_news_data_dict = json.load(open(os.path.join(self.visual_news_root_dir+"data.json")))
        self.visual_news_data_mapping = {ann["id"]: ann for ann in self.visual_news_data_dict}
        self.news_clip_data_dict = json.load(open(os.path.join(self.news_clip_root_dir,split+".json")))["annotations"]
        
    def __len__(self):
        return len(self.news_clip_data_dict)   

    def load_img_pil(self,image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
                       
    def load_queries(self,visual_news_caption_item, visual_news_image_item):
        caption = visual_news_caption_item['caption']
        caption_tokenized = clip.tokenize(caption) 
        image_path = os.path.join(self.visual_news_root_dir, visual_news_image_item['image_path'])
        pil_img = self.load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        return transform_img, caption_tokenized

    def __getitem__(self, idx):      
        if torch.is_tensor(idx):
            idx = idx.tolist()    
        label = torch.as_tensor(1) if self.news_clip_data_dict[idx]['falsified'] else torch.as_tensor(0)
   
        visual_news_caption_item = self.visual_news_data_mapping[self.news_clip_data_dict[idx]["id"]]
        visual_news_image_item = self.visual_news_data_mapping[self.news_clip_data_dict[idx]["image_id"]]
        qImg, qCap = self.load_queries(visual_news_caption_item, visual_news_image_item)

        #sample = {'label': label,'qImg': qImg, 'qCap': qCap}   
        return label, qImg, qCap
        
## test 
#import json 
#import os 
#import torch
#import torchvision
#import dataset_mismatch
#transform = torchvision.transforms.Compose([
#    torchvision.transforms.Resize(256),
#    torchvision.transforms.CenterCrop(224),
#    torchvision.transforms.ToTensor(),
#    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#])
#visual_news_root_dir = "../visual_news/origin/"
#news_clip_root_dir = "../news_clippings/data/merged_balanced/"
#split = 'val'

#val_dataset = dataset_mismatch.NewsClipDataset(visual_news_root_dir, news_clip_root_dir, split, transform)
#from torch.utils.data import DataLoader
#def custom_collate_mismatch(batch): 
#    labels = torch.stack(batch[0],dim=0)
#    imgs = torch.stack(batch[1], dim=0)
#    captions_batch = batch[2]
#    return labels, imgs, captions_batch 
    
#val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, collate_fn = custom_collate_mismatch, num_workers=0)
#labels, imgs, captions_batch  = next(iter(val_dataloader))
