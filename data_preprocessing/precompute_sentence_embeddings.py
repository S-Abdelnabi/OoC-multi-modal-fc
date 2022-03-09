import numpy as np
import json
from urllib.parse import urlparse
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import dataset_compute_emb 
import torchvision 
import os 
import argparse
import io

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

parser.add_argument('--end_idx', type=int, default=-1,
                    help='where to end, if not specified will do all')
                    
parser.add_argument('--delete_html', action='store_true', help='whether to delete existing html files')


args = parser.parse_args()
print("Precomputing features for: "+args.split)
print("Reading items from: "+args.dataset_items_file)

sbert_model = SentenceTransformer('paraphrase-mpnet-base-v2')
sbert_model.eval()
sbert_model.cuda()

def get_sent_bert_emb(text, convert_to_tensor=False):
    with torch.no_grad():
        encoded_sentences = sbert_model.encode(text,convert_to_tensor=convert_to_tensor)
    return encoded_sentences 
    
def save_features(features, full_file_path):
    if torch.is_tensor(features):
        np.savez_compressed(full_file_path,features.detach().cpu().numpy())
    else:
        np.savez_compressed(full_file_path,features)
        
def save_dict(dict_data, full_file_path):
    with io.open(full_file_path, 'w') as db_file:
        json.dump(dict_data, db_file)

def delete_textfiles(folder_path):
    files_in_directory = os.listdir(folder_path)
    filtered_files = [file for file in files_in_directory if file.endswith(".txt")]
    for file in filtered_files:
        path_to_file = os.path.join(folder_path, file)
        os.remove(path_to_file)

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

if args.end_idx != -1:
    end_idx = args.end_idx 
else:
    end_idx = len(context_data_items_dict)
    
print('Start: '+str(start_idx))
print('End: '+str(end_idx))
    
for i in range(start_idx, end_idx):
    key = list(context_data_items_dict.keys())[i]
    print("item number: " + str(i)+ ", key: " + str(key)) 
    sample = dataset.__getitemNoimgs__(i)
    keys_as_int = [int(x) for x in context_data_items_dict.keys()]    
                
    inv_path_item = os.path.join(args.queries_dataset_root,context_data_items_dict[key]['inv_path'])

    if sample['entities']: 
        #print(sample['entities'])
        entities_features = get_sent_bert_emb(sample['entities'], convert_to_tensor=False)
        #these are processed features according to the processing i dataset_compute_emb
        save_features(entities_features, os.path.join(inv_path_item, 'entities_features2'))  
    
    if sample['caption']:   
        #print(sample['caption'])
        caption_features = get_sent_bert_emb(sample['caption'], convert_to_tensor=False)
        save_features(caption_features, os.path.join(inv_path_item, 'caption_features2'))
        
    qCap_features = get_sent_bert_emb([sample['qCap']], convert_to_tensor=False)
    save_features(qCap_features,os.path.join(inv_path_item,'qCap_sbert_features'))   
    
 

