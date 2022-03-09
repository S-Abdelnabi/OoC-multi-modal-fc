import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import json
from urllib.parse import urlparse
from PIL import Image
import os 
import cv2 
import imghdr
from unidecode import unidecode

def process_string(input_str):
    input_str = input_str.replace('&#39;', ' ')
    input_str = input_str.replace('<b>','')
    input_str = input_str.replace('</b>','')
    input_str = unidecode(input_str)  
    return input_str
        
class NewsContextDataset(Dataset):
    def __init__(self, context_data_items_dict, visual_news_root_dir, queries_root_dir, news_clip_root_dir, domain_to_idx_dict_path, split, transform, clip_transform=None):
        self.context_data_items_dict = context_data_items_dict
        self.visual_news_root_dir = visual_news_root_dir
        self.queries_root_dir = queries_root_dir
        self.news_clip_root_dir = news_clip_root_dir
        self.transform = transform
        self.idx_to_keys = list(context_data_items_dict.keys())
        with open(domain_to_idx_dict_path) as json_file:
            self.domain_to_idx_dict = json.load(json_file)
        self.visual_news_data_dict = json.load(open(os.path.join(self.visual_news_root_dir+"data.json")))
        self.visual_news_data_mapping = {ann["id"]: ann for ann in self.visual_news_data_dict}
        self.news_clip_data_dict = json.load(open(os.path.join(self.news_clip_root_dir,split+".json")))["annotations"]
        self.clip_transform = clip_transform 
        
    def __len__(self):
        return len(self.context_data_items_dict)   

    def load_img_pil(self,image_path):
        if imghdr.what(image_path) == 'gif': 
            try:
                with open(image_path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('RGB')
            except:
                return None 
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def load_entities(self,inv_dict):
        entities = []
        if 'entities' in inv_dict.keys():
            entities = inv_dict['entities']
        if 'best_guess_lbl' in inv_dict.keys():
            entities = entities + inv_dict['best_guess_lbl']
        for i in range(0,len(entities)):
            entities[i] = process_string(entities[i])
                    
        return entities 
    
    def get_domain_idx(self,page):
        if 'page_link' in page.keys():
            domain = urlparse(page['page_link']).netloc
            if domain in self.domain_to_idx_dict.keys(): 
                domain_idx = self.domain_to_idx_dict[domain]
            else:
                domain_idx = self.domain_to_idx_dict['UNK']
        else:
            domain_idx = self.domain_to_idx_dict['UNK']
        return domain_idx
            
    def load_imgs_direct_search(self,item_folder_path,direct_dict):   
        list_imgs_tensors = []
        count = 0   
        metadata = {}              
        keys_to_check = ['images_with_captions','images_with_no_captions','images_with_caption_matched_tags']
        for key1 in keys_to_check:
            if key1 in direct_dict.keys():
                for page in direct_dict[key1]:
                    image_path = os.path.join(item_folder_path,page['image_path'].split('/')[-1])
                    pil_img = self.load_img_pil(image_path)
                    if pil_img == None: continue
                    transform_img = self.transform(pil_img)
                    metadata[count] = {'domain': urlparse(page['page_link']).netloc, 'name': page['image_path'].split('/')[-1]}
                    count = count + 1 
                    list_imgs_tensors.append(transform_img)
        stacked_tensors = torch.stack(list_imgs_tensors, dim=0)
        return stacked_tensors, metadata

    def load_queries(self,visual_news_caption_item, visual_news_image_item):
        caption = visual_news_caption_item['caption']
        image_path = os.path.join(self.visual_news_root_dir, visual_news_image_item['image_path'])
        pil_img = self.load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        if self.clip_transform is not None:        
            clip_transform_img = self.clip_transform(pil_img)
            return transform_img, clip_transform_img, caption
        return transform_img, caption
        
    def load_captions(self,inv_dict):
        captions = []
        domains = {}   
        pages_with_captions_keys = ['all_fully_matched_captions','all_partially_matched_captions']
        count = 0 
        for key1 in pages_with_captions_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        item = page['title']
                        item = process_string(item)
                        captions.append(item)
                        domains[count] = urlparse(page['page_link']).netloc
                        count = count+1 
                    
                    if 'caption' in page.keys():
                        sub_captions_list = []
                        unfiltered_captions = []
                        page_domain = urlparse(page['page_link']).netloc
                        for key2 in page['caption']:
                            sub_caption = page['caption'][key2]
                            sub_caption_filter = process_string(sub_caption)
                            if sub_caption in unfiltered_captions: continue 
                            sub_captions_list.append(sub_caption_filter) 
                            unfiltered_captions.append(sub_caption) 

                            domains[count] = urlparse(page['page_link']).netloc
                            count = count+1
                        captions = captions + sub_captions_list 
                    
        pages_with_title_only_keys = ['partially_matched_no_text','fully_matched_no_text']
        for key1 in pages_with_title_only_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        title = process_string(page['title'])
                        captions.append(title)
                        domains[count] = urlparse(page['page_link']).netloc
                        count = count+1 
        return captions, domains  

    def __getitem__(self, idx):      
        if torch.is_tensor(idx):
            idx = idx.tolist()    
        key = self.idx_to_keys[idx]
        label = torch.as_tensor(1) if self.news_clip_data_dict[int(key)]['falsified'] else torch.as_tensor(0)
        direct_path_item = os.path.join(self.queries_root_dir,self.context_data_items_dict[key]['direct_path'])
        direct_ann_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json')))
        inverse_path_item = os.path.join(self.queries_root_dir,self.context_data_items_dict[key]['inv_path'])
        inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json')))
        entities = self.load_entities(inv_ann_dict)
        captions,captions_domains = self.load_captions(inv_ann_dict)    
        imgs,imgs_data = self.load_imgs_direct_search(direct_path_item,direct_ann_dict)     
        visual_news_caption_item = self.visual_news_data_mapping[self.news_clip_data_dict[int(key)]["id"]]
        #print(visual_news_caption_item)
        visual_news_image_item = self.visual_news_data_mapping[self.news_clip_data_dict[int(key)]["image_id"]]
        #print(visual_news_image_item)
        if self.clip_transform is not None:
            qImg, qImg_clip, qCap =  self.load_queries(visual_news_caption_item, visual_news_image_item)
        else:
            qImg, qCap = self.load_queries(visual_news_caption_item, visual_news_image_item)

        sample = {'label': label, 'entities':entities, 'caption': captions, 'caption_domains': captions_domains, 'imgs': imgs, 'imgs_data': imgs_data, 'qImg': qImg, 'qCap': qCap}  
        if self.clip_transform is not None:      
            sample['qImg_clip': qImg_clip]         
        return sample

    def __getitemNoimgs__(self, idx):      
        if torch.is_tensor(idx):
            idx = idx.tolist()    
        key = self.idx_to_keys[idx]
        label = torch.as_tensor(1) if self.news_clip_data_dict[int(key)]['falsified'] else torch.as_tensor(0)
        direct_path_item = os.path.join(self.queries_root_dir,self.context_data_items_dict[key]['direct_path'])
        direct_ann_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json')))
        inverse_path_item = os.path.join(self.queries_root_dir,self.context_data_items_dict[key]['inv_path'])
        inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json')))
        entities = self.load_entities(inv_ann_dict)
        captions,captions_domains = self.load_captions(inv_ann_dict)            
        visual_news_caption_item = self.visual_news_data_mapping[self.news_clip_data_dict[int(key)]["id"]]
        visual_news_image_item = self.visual_news_data_mapping[self.news_clip_data_dict[int(key)]["image_id"]]
        if self.clip_transform is not None:
            qImg, qImg_clip, qCap =  self.load_queries(visual_news_caption_item, visual_news_image_item)
        else:
            qImg, qCap = self.load_queries(visual_news_caption_item, visual_news_image_item)

        sample = {'label': label, 'entities':entities, 'caption': captions, 'caption_domains': captions_domains, 'qImg': qImg, 'qCap': qCap}  
        if self.clip_transform is not None:      
            sample['qImg_clip': qImg_clip]         
        return sample
