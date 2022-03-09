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
import sys 

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

parser.add_argument('--resnet_arch', type=str, default='resnet50',
                    help='which resnet arch to use')
  
parser.add_argument('--start_idx', type=int, default=-1,
                    help='where to start, if not specified will be start from 0')

parser.add_argument('--end_idx', type=int, default=-1,
                    help='where to end, if not specified will do all')
                    
parser.add_argument('--delete_html', action='store_true', help='whether to delete existing html files')


args = parser.parse_args()
print("Precomputing features for: "+args.split)
print("Reading items from: "+args.dataset_items_file)


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
visual_news_data_dict = json.load(open(os.path.join(args.visual_news_root+"data.json")))
visual_news_data_mapping = {ann["id"]: ann for ann in visual_news_data_dict}
news_clip_data_dict = json.load(open(os.path.join(args.news_clip_root,args.split+".json")))["annotations"]

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
    keys_as_int = [int(x) for x in context_data_items_dict.keys()]    
    direct_path_item = os.path.join(args.queries_dataset_root,context_data_items_dict[key]['direct_path'])
    #load labels. 
    res_labels = json.load(open(os.path.join(direct_path_item,'detected_labels')))
    #img features 
    all_imgs_features = json.load(open(os.path.join(direct_path_item,'metadata_of_features')))
    if len(res_labels) != len(all_imgs_features):
        print('mismatch')
        sys.exit()

    inv_path_item = os.path.join(args.queries_dataset_root,context_data_items_dict[key]['inv_path'])
    query_labels = json.load(open(os.path.join(inv_path_item,'query_detected_labels')))['labels']
    if len(query_labels) == 0:
        print('no label for query')
        intersection = np.zeros((len(res_labels),))
    else:
        intersection = []
        for one_res_img in res_labels.keys():
            labels_per_res = res_labels[one_res_img]['labels']
            intersection_per_img = 0
            if len(labels_per_res) == 0: 
                intersection.append(intersection_per_img) 
                print('no labels for result')
                continue 
            for query_label in query_labels:
                if query_label in labels_per_res: 
                    intersection_per_img = intersection_per_img + 1
            intersection_per_img = intersection_per_img/len(query_labels)
            intersection.append(intersection_per_img)         
    if len(res_labels) != len(intersection):
        print('mismatch in overlap')
        sys.exit()    
    save_features(np.asarray(intersection), os.path.join(inv_path_item, 'labels_overlap_percentage'))  




