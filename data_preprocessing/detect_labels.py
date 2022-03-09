import numpy as np
import json
from urllib.parse import urlparse
import torch
import torch.nn as nn
import resnet_places
import dataset_compute_emb 
import torchvision 
import os 
import argparse
import io
import sys 

parser = argparse.ArgumentParser(description='Precompute embeddings')
parser.add_argument('--queries_dataset_root', type=str, default='../queries_dataset/merged_balanced/',
                    help='location to the root folder of the query dataset')
parser.add_argument('--google_cred_json', type=str, default='my_file.json',
                    help='json file for credentials')
                    
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
  
parser.add_argument('--start_idx', type=int, default=-1,
                    help='where to start, if not specified will be start from 0')

parser.add_argument('--end_idx', type=int, default=-1,
                    help='where to end, if not specified will do all')
                    
parser.add_argument('--delete_html', action='store_true', help='whether to delete existing html files')


args = parser.parse_args()
print("Precomputing features for: "+args.split)
print("Reading items from: "+args.dataset_items_file)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_cred_json
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
        


def detect_labels(path):
    """Detects labels in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)

    labels = response.label_annotations
    labels_description = []
    labels_scores = []

    for label in labels:
        if label.description:
            labels_description.append(label.description)
            labels_scores.append(label.score)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
        #sys.exit()
    return labels_description, labels_scores


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
    if int(key)%2==0 or (int(key)%2!=0 and int(key)-1 not in keys_as_int) :
        direct_path_item = os.path.join(args.queries_dataset_root,context_data_items_dict[key]['direct_path'])
        #load metadata. 
        labels = {}
        imgs_metadata = json.load(open(os.path.join(direct_path_item,'metadata_of_features')))
        #loop through imgs.
        for img_idx in imgs_metadata.keys():     
            file_path = os.path.join(direct_path_item,imgs_metadata[img_idx]['name'])
            try:
                labels_description, labels_score = detect_labels(file_path) 
            except:               
                labels_description = []
                labels_score = []                
            labels[img_idx] = {'labels': labels_description, 'scores': labels_score}         
        save_dict(labels, os.path.join(direct_path_item,'detected_labels'))

    inv_path_item = os.path.join(args.queries_dataset_root,context_data_items_dict[key]['inv_path'])
    visual_news_image_item = visual_news_data_mapping[news_clip_data_dict[int(key)]["image_id"]]
    image_path = os.path.join(args.visual_news_root, visual_news_image_item['image_path'])
    try:
        labels_description, labels_score = detect_labels(image_path)
    except:
        labels_description = []
        labels_score = []
    save_dict({'labels': labels_description, 'scores': labels_score}, os.path.join(inv_path_item,'query_detected_labels'))


