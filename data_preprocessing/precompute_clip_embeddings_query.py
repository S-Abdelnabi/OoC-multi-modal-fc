import numpy as np
import json
from urllib.parse import urlparse
from PIL import Image
import torch
import torch.nn as nn
import dataset_mismatch
import torchvision 
import os 
import argparse
import io
import clip 
import clip_classifier

parser = argparse.ArgumentParser(description='Precompute embeddings')
parser.add_argument('--queries_dataset_root', type=str, default='../queries_dataset/merged_balanced/',
                    help='location to the root folder of the query dataset')
                    
parser.add_argument('--visual_news_root', type=str, default='../visual_news/origin/',
                    help='location to the root folder of the visualnews dataset')

parser.add_argument('--news_clip_root', type=str, default='../news_clippings/data/merged_balanced/',
                    help='location to the root folder of the clip dataset')
                    
parser.add_argument('--dataset_items_file', type=str, default='dataset_items_train.json',
                    help='location to the dataset items file')
                    
parser.add_argument('--clip_path', type=str, default='clip_model/best_model_acc.pth.tar',
                    help='location to the clip model')

parser.add_argument('--domains_file', type=str, default='domain_to_idx_dict.json',
                    help='location to the domains to idx file')
                    
parser.add_argument('--split', type=str, default='train',
                    help='which split to compute the embeddings for')

parser.add_argument('--resnet_arch', type=str, default='resnet152',
                    help='which resnet arch to use')
  
parser.add_argument('--start_idx', type=int, default=-1,
                    help='where to start, if not specified will be start from 0')
                    
parser.add_argument('--delete_html', action='store_true', help='whether to delete existing html files')


args = parser.parse_args()
print("Precomputing features for: "+args.split)
print("Reading items from: "+args.dataset_items_file)

#### settings of the model ####
model_settings = {'pdrop':0.5}
base_clip, preprocess = clip.load("ViT-B/32", device="cuda")
classifier_clip = clip_classifier.ClipClassifier(model_settings,base_clip)
classifier_clip.cuda()

#### load Datasets ####
dataset = dataset_mismatch.NewsClipDataset(args.visual_news_root, args.news_clip_root, args.split, preprocess)

if os.path.isfile(args.clip_path):
    print("=> loading checkpoint '{}'".format(args.clip_path))
    checkpoint = torch.load(args.clip_path)
    classifier_clip.load_state_dict(checkpoint['state_dict'])
    clip.model.convert_weights(classifier_clip)
    print("=> loaded checkpoint '{}' (epoch {})".format(args.clip_path, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.clip_path))
    
def save_features(features, full_file_path):
    if torch.is_tensor(features):
        np.savez_compressed(full_file_path,features.detach().cpu().numpy())
    else:
        np.savez_compressed(full_file_path,features)

#load items file 
context_data_items_dict = json.load(open(args.dataset_items_file))


if args.start_idx != -1:
    start_idx = args.start_idx 
else:
    start_idx = 0 

classifier_clip.eval()
    
for i in range(start_idx, len(context_data_items_dict)):
    key = list(context_data_items_dict.keys())[i]
    print("item number: " + str(i)+ ", key: " + str(key)) 
    label, qImg, qCap = dataset.__getitem__(int(key))
          
    inv_path_item = os.path.join(args.queries_dataset_root,context_data_items_dict[key]['inv_path'])

    qImg = qImg.cuda() 
    with torch.no_grad():
        qImg_clip_features = classifier_clip.clip.encode_image(torch.unsqueeze(qImg,dim=0))
    save_features(qImg_clip_features,os.path.join(inv_path_item,'qImg_clip_features'))   
    
    qCap = qCap.cuda()
    with torch.no_grad():
        qCap_features = classifier_clip.clip.encode_text(qCap) 
    save_features(qCap_features,os.path.join(inv_path_item,'qCap_clip_features'))   

