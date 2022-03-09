import numpy as np
import json
from urllib.parse import urlparse
from PIL import Image
import torch
import torch.nn as nn
import dataset_compute_emb 
import torchvision 
import os 
import argparse
import io
import spacy
NER = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser(description='Precompute embeddings')
parser.add_argument('--queries_dataset_root', type=str, default='../queries_dataset/merged_balanced/',
                    help='location to the root folder of the query dataset')
                    
parser.add_argument('--visual_news_root', type=str, default='../visual_news/origin/',
                    help='location to the root folder of the visualnews dataset')

parser.add_argument('--news_clip_root', type=str, default='../news_clippings/data/merged_balanced/',
                    help='location to the root folder of the clip dataset')
                    
parser.add_argument('--dataset_items_file', type=str, default='dataset_items_train.json',
                    help='location to the dataset items file')

parser.add_argument('--domains_file', type=str, default='domain_to_idx_dict.json',
                    help='location to the domains to idx file')
                    
parser.add_argument('--split', type=str, default='train',
                    help='which split to compute the embeddings for')

parser.add_argument('--resnet_arch', type=str, default='resnet152',
                    help='which resnet arch to use')
  
parser.add_argument('--start_idx', type=int, default=-1,
                    help='where to start, if not specified will be start from 0')
                    
parser.add_argument('--delete_html', action='store_true', help='whether to delete existing html files')

def compare_entities(text1,text2):
    for word in text1.ents:
        for word2 in text2.ents:
            if word.text.lower() == word2.text.lower() and word.label_ == word2.label_:
                return 1
    return 0 

def find_entities_overlap(text_list, query_ner):
    overlap = []
    for item in text_list:
        item_ner = NER(item)
        overlap.append(compare_entities(item_ner, query_ner))
    return np.asarray(overlap)

args = parser.parse_args()
print("Precomputing binary features for: "+args.split)
print("Reading items from: "+args.dataset_items_file)

    
def save_features(features, full_file_path):
    if torch.is_tensor(features):
        np.savez_compressed(full_file_path,features.detach().cpu().numpy())
    else:
        np.savez_compressed(full_file_path,features)
        
#load items file 
context_data_items_dict = json.load(open(args.dataset_items_file))

#transform 
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#dataset file 
dataset = dataset_compute_emb.NewsContextDataset(context_data_items_dict, args.visual_news_root, args.queries_dataset_root, args.news_clip_root,args.domains_file, args.split, transform)

if args.start_idx != -1:
    start_idx = args.start_idx 
else:
    start_idx = 0 
    
for i in range(start_idx, len(dataset)):
    key = list(context_data_items_dict.keys())[i]
    print("item number: " + str(i)+ ", key: " + str(key)) 
    sample = dataset.__getitemNoimgs__(i)
    keys_as_int = [int(x) for x in context_data_items_dict.keys()]    

    query_text = sample['qCap']
    query_ner = NER(query_text)
    inv_path_item = os.path.join(args.queries_dataset_root,context_data_items_dict[key]['inv_path'])
        
    if sample['entities']: 
        entities_overlap = find_entities_overlap(sample['entities'], query_ner)
        #processed text 
        save_features(entities_overlap, os.path.join(inv_path_item, 'entities_binary_feature2'))  
    
    if sample['caption']:   
        cap_overlap = find_entities_overlap(sample['caption'], query_ner)
        #processed text 
        save_features(cap_overlap, os.path.join(inv_path_item, 'captions_binary_feature2'))  
        
 

