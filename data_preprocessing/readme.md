### Overview 

- We share our data preprocessing and embeddings computing. These scripts are used to compute the embeddings and save the files described in [curated_dataset](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/curated_dataset).

- If you are using our curated preprocessed dataset with the precomputed embeddings, you can skip this step. 

- - -

### Enviroment 
- python=3.7.6
- spacy=2.3.4
- transformers=4.8.1
- pytorch=1.7.1
- torchvision=0.8.2
- imagehash=4.2.0
- - -
### Requirments
- You will need to have ```dataset_items_<split>.json``` files (list of items, their keys in NewsCLIPpings, and paths to direct and inverse search.
- As always, you need to have access to both NewsCLIPpings and VisualNews.
- - -
### Compute ResNet (ImageNet) embeddings 
```
precompute_embeddings_resnet_ImageNet.py --split <split> --dataset_items_file dataset_items_<split>.json 
```
### Compute ResNet places embeddings
```
python precompute_places_embeddings.py --split <split> --dataset_items_file dataset_items_<split>.json 
```
- You can specify *start_idx* and *end_idx* if needed. 
- This will precompute and save the places embeddings under the 'direct_search' paths for the visual evidence, and under 'inverse_search' for the query images. 
- We used the pytorch ResNet50 checkpoint from: https://github.com/CSAILVision/places365 
- - -
### Detect labels 
- Run [detect labels API](https://cloud.google.com/vision/docs/labels) on all images.
- You need to configure the Google cloud platform and enable billing for this API and have some authentication form, we use 'API key' authentication. Change the argument *google_cred_json* to include the path to your file.
```
python detect_labels.py --split <split> --dataset_items_file dataset_items_<split>.json
```
- You can specify *start_idx* and *end_idx* if needed. 

- Then, find the overlap between the query and visual evidence images: 
```
python compute_labels_overlap.py --split <split> --dataset_items_file dataset_items_<split>.json 
```
- - -
### Detect named entities overlap (binary feature)
- Whether there is an overlap in named entities between query and textual evidence. 
- This uses the spacy NER to find named entities.
```
python precompute_binary_features.py --split <split> --dataset_items_file dataset_items_<split>.json 
```
- - -

### Sentence embeddings for textual evidence ###
```
python precompute_sentence_embeddings.py --split <split> --dataset_items_file dataset_items_<split>.json
```
- - -
### CLIP embeddings ###

- Refer to [CLIP repository](https://github.com/openai/CLIP) for the requirements  
- You can find our CLIP checkpoint here, fine-tuned on the task of classifying pairs as pristine or falsified.
```
python precompute_clip_embeddings_query.py --split <split> --dataset_items_file dataset_items_<split>.json
``` 
- - -

### Detect overlap to queries ###
- For pristine examples, we filter out the evidence that had an exact match to the query (same text after processing, same perceptual hash, and coming from the same website).
- This script will save the 'imgs_to_keep_idx' and 'captions_to_keep_idx', as described in [curated_dataset](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/curated_dataset).
 ```
 python find_duplicates_to_query.py --split <split> --dataset_items_file dataset_items_<split>.json 
 ```
