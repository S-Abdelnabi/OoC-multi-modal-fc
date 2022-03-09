#!/usr/bin/env python
# coding: utf-8

# In[27]:

import argparse
import requests
import os
import PIL
import shutil
from PIL import Image
import imghdr
from bs4 import BeautifulSoup
import bs4
import time
from googleapiclient.discovery import build
from google.cloud import vision
import io
import os
from bs4 import NavigableString
import json
from utils import get_captions_from_page, save_html
import time

parser = argparse.ArgumentParser(description='Download dataset for direct search queries')
parser.add_argument('--save_folder_path', type=str, default='queries_dataset',
                    help='location where to download data')
parser.add_argument('--google_cred_json', type=str, default='my_file.json',
                    help='json file for credentials')
parser.add_argument('--google_api_key', type=str, default='',
                    help='api_key for the custom search engine')
                    
parser.add_argument('--google_cse_id', type=str, default='',
                    help='custom search engine id')                    
                    
parser.add_argument('--split_type', type=str, default='merged_balanced',
                    help='which split to use in the NewsCLIP dataset')
parser.add_argument('--sub_split', type=str, default='val',
                    help='which split to use from train,val,test splits')
                    
parser.add_argument('--how_many_queries', type=int, default=1,
                    help='how many query to issue for each item - each query is 10 images')
parser.add_argument('--continue_download', type=int, default=0,
                    help='whether to continue download or start from 0 - should be 0 or 1')

parser.add_argument('--how_many', type=int, default=-1,
                    help='how many items to query and download, 0 means download untill the end')
parser.add_argument('--end_idx', type=int, default=-1,
                    help='where to end, if not specified, will be inferred from how_many')    
parser.add_argument('--start_idx', type=int, default=-1,
                    help='where to start, if not specified will be inferred from the current saved json or 0 otherwise')


parser.add_argument('--hashing_cutoff', type=int, default=15,
                    help='threshold used in hashing')
args = parser.parse_args()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_cred_json
my_api_key = args.google_api_key
my_cse_id =  args.google_cse_id

full_save_path = os.path.join(args.save_folder_path,args.split_type,'direct_search',args.sub_split)
if not os.path.exists(full_save_path):
    os.makedirs(full_save_path)
    
##file for saving errors in saving..
if os.path.isfile(os.path.join(full_save_path,'unsaved.txt')) and args.continue_download:
    saved_errors_file= open(os.path.join(full_save_path,'unsaved.txt'), "a")
else:
    saved_errors_file= open(os.path.join(full_save_path,'unsaved.txt'), "w")

##file for keys with no annotations..
if os.path.isfile(os.path.join(full_save_path,'no_annotations.txt')) and args.continue_download:
    no_annotations_file= open(os.path.join(full_save_path,'no_annotations.txt'), "a")
else:
    no_annotations_file= open(os.path.join(full_save_path,'no_annotations.txt'), "w")

# json file containing the index and path of all downloaded items. 
json_download_file_name = os.path.join(full_save_path,args.sub_split+'.json') 

#continue using the current saved json file -- don't start a new file -- load the saved dict 
if os.path.isfile(json_download_file_name) and os.access(json_download_file_name, os.R_OK) and args.continue_download:
    with open(json_download_file_name, 'r') as fp:
        all_direct_annotations_idx = json.load(fp)
#start a new file -- start from an empty dict 
else:
    with io.open(json_download_file_name, 'w') as db_file:
        db_file.write(json.dumps({}))
    with io.open(json_download_file_name, 'r') as db_file:
        all_direct_annotations_idx = json.load(db_file)

### load visual_news and CLIP datasets ####
visual_news_data = json.load(open("visual_news/origin/data.json"))
visual_news_data_mapping = {ann["id"]: ann for ann in visual_news_data}
clip_data = json.load(open("news_clippings/data/"+args.split_type+"/"+args.sub_split+".json"))
clip_data_annotations = clip_data["annotations"]

### find where to start from the clip_data ###
# if continue_download, get the last idx of the last downloaded item. otherwise, start from 0 
### find where to start from the clip_data ###
if args.start_idx != -1:
    start_counter = args.start_idx
else:
    if all_direct_annotations_idx:
        start_counter = int(list(all_direct_annotations_idx.keys())[-1])+2
    else:
        start_counter = 0

#set the end_counter using the how_many argument
#multiply by 2 since the textual queries loop skips 2 (repeated queries) 
#if no valid how_many queries, set the end_counter to the end of the annotations list 
if args.end_idx > 0:
    end_counter = args.end_idx
else:   
    if args.how_many>0: 
        end_counter = start_counter + 2*args.how_many  
    else:
        end_counter = len(clip_data_annotations) 
        
print("==========")
print("subset to download is: "+args.split_type+" - "+args.sub_split)
print("Starting from index: %5d"%start_counter)
print("Ending at index: %5d"%end_counter)
if args.continue_download == 1:
    print("Continue download on file: "+json_download_file_name)
else:
    print("Start download from: " + str(start_counter) +" with creating a new file: "+json_download_file_name)
print("==========") 

def google_search(search_term, api_key, cse_id, how_many_queries, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res_list = []
    for i in range(0,how_many_queries):
        start = i*10 + 1
        res = service.cse().list(q=search_term, searchType='image', lr='lang_en', num = 10, start=start, cx=cse_id, **kwargs).execute()
        res_list.append(res)
    return res_list

def download_and_save_image(image_url, save_folder_path, file_name):
    try:
        response = requests.get(image_url,stream = True,timeout=(60,60))
        if response.status_code == 200:
            response.raw.decode_content = True
            image_path = os.path.join(save_folder_path,file_name+'.jpg')
            with open(image_path,'wb') as f:
                shutil.copyfileobj(response.raw, f)
            if imghdr.what(image_path).lower() == 'png':
                img_fix = Image.open(image_path)
                img_fix.convert('RGB').save(image_path)
            return 1 
        else:
            return 0
    except:
        return 0 

def get_direct_search_annotation(search_results_lists,save_folder_path):
    image_save_counter = 0 
    
    direct_annotation = {}
    images_with_captions = []
    images_with_captions_matched_tags = []
    images_with_no_captions = []
    
    for one_result_list in search_results_lists:
        if 'items' in one_result_list.keys():
            for item in one_result_list['items']:
                image={}
                caption = {}
                if 'link' in item.keys():
                    image['img_link'] = item['link']
                if 'contextLink' in item['image'].keys():          
                    image['page_link'] = item['image']['contextLink'] 
                if 'displayLink' in item.keys():            
                    image['domain'] = item['displayLink']
                if 'snippet' in item.keys():
                    image['snippet'] = item['snippet']
                #download and save images
                download_status = download_and_save_image(item['link'], save_folder_path, str(image_save_counter))
                #if the image cannot be downloaded, skip..
                if download_status == 0:
                    continue                
                image['image_path'] = os.path.join(save_folder_path,str(image_save_counter)+'.jpg')
            
                try:
                    caption,title,code,req = get_captions_from_page(item['link'],item['image']['contextLink'])
                except: 
                    print('Error happened in getting captions')
                    continue 
                saved_html_flag = save_html(req, os.path.join(save_folder_path,str(image_save_counter)+'.txt'))     
                if saved_html_flag:            
                    image['html_path'] = os.path.join(save_folder_path,str(image_save_counter)+'.txt')
                
                if len(code)>0 and (code[0] == '5' or code[0] == '4'):
                    image['is_request_error'] = True 
                if 'title' in item.keys():
                    if item['title'] is None: item['title'] = ''
                else:
                    item['title'] = ''            
                if title is None: title = ''       
                if len(title) > len( item['title'].lstrip().rstrip()):
                    image['page_title'] = title
                else:
                    image['page_title'] = item['title']

                if caption:
                    image['caption'] = caption
                    images_with_captions.append(image)
                else:
                    try:
                        caption,title,code,req = get_captions_from_page(item['link'],item['image']['contextLink'],req,args.hashing_cutoff)
                    except: 
                        print('Error happened in getting captions')
                        continue 
                    if caption: 
                        image['caption'] = caption          
                        images_with_captions_matched_tags.append(image)
                    else:
                        images_with_no_captions.append(image)
                image_save_counter = image_save_counter + 1
    
    
    if len(images_with_captions) == 0 and len(images_with_no_captions) == 0 and len(images_with_captions_matched_tags) == 0:
        direct_annotation = {}
    else:
        direct_annotation['images_with_captions'] = images_with_captions
        direct_annotation['images_with_no_captions'] = images_with_no_captions
        direct_annotation['images_with_caption_matched_tags'] = images_with_captions_matched_tags
    return direct_annotation    

def search_and_save_one_query(text_query, id_in_clip, image_id_in_visualNews, text_id_in_visualNews, new_folder_path):
    global all_direct_annotations_idx, no_annotations_file
    
    result = google_search(text_query, my_api_key, my_cse_id,how_many_queries=args.how_many_queries)
    direct_search_results = get_direct_search_annotation(result,new_folder_path)
    new_json_file_path = os.path.join(new_folder_path,'direct_annotation.json')
    if direct_search_results:
        new_entry = {id_in_clip: {'image_id_in_visualNews':image_id_in_visualNews, 'text_id_in_visualNews': text_id_in_visualNews, 'folder_path':new_folder_path}}
        all_direct_annotations_idx.update(new_entry)
        save_json_file(json_download_file_name, all_direct_annotations_idx, id_in_clip, saving_idx_file=True)
        save_json_file(new_json_file_path, direct_search_results, id_in_clip)
    else:
        #save that there is no annotations for that key... 
        no_annotations_file.write(str(id_in_clip)+'\n')  
        no_annotations_file.flush()        

def save_json_file(file_path, dict_file, cur_id_in_clip, saving_idx_file=False):
    global all_direct_annotations_idx, saved_errors_file
    #load the previous saved file 
    if saving_idx_file:
        with open(file_path, 'r') as fp:
            old_idx_file = json.load(fp)  
    try:
        with io.open(file_path, 'w') as db_file:
            json.dump(dict_file, db_file)
    except:
        saved_errors_file.write(str(cur_id_in_clip)+'\n')
        saved_errors_file.flush()
        if saving_idx_file:
            all_direct_annotations_idx = old_idx_file 
            with io.open(file_path, 'w') as db_file:
                json.dump(old_idx_file, db_file)
      
### Loop ####
for i in range(start_counter,end_counter,2):
    print("Item number: %6d"%i)
    start_time = time.time()
    ann = clip_data_annotations[i]
    text_query = visual_news_data_mapping[ann["id"]]["caption"]
    id_in_clip = i 
    image_id_in_visualNews = ann["image_id"]
    text_id_in_visualNews = ann["id"]
    new_folder_path = os.path.join(full_save_path,str(i))
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    search_and_save_one_query(text_query, id_in_clip, image_id_in_visualNews, text_id_in_visualNews, new_folder_path) 
    end_time = time.time()   
    print("--- Time elapsed for 1 query: %s seconds ---" % (end_time - start_time))

