## Dataset Access 
If you like to have access to our collected evidence, please fill out this [form](https://forms.gle/HZeUK1EEveGF9yEV9). 

We share the raw Google search results, in addition to our preprocessing, and precomputed embeddings. 

## Dataset description:

- The query images and captions are found in the NewsCLIPpings datasets (we use the merged balanced dataset).
- For each instance, we collect:
	- direct search: search for images given captions 
	- inverse search: search for captions given images 
	- we compute embeddings for: visual evidence, textual evidence, and the query.

- Direct search (visual evidence):
	- Each even-indexed instance has its own directory, under which you can find:
		- The images search results
		- *direct_annotations.json*, it contains the raw results from the search API (during the data collection). You can find there the images links, pages links, and the captions of the found images (note that we don't use these captions in our model). 
		- In addition to the raw results, you can find the precomputed embeddings of the search results, as follows: 
			- precomputed resnet embeddings (*resnet_features.npz*)
			- precomputed resnet embeddings trained on places dataset (*places_resnet_features.npz*)
			- precomputed clip embeddings (*imgs_clip_features.npz*), note that we don't use these during training
			- *metadata_of_features*: the mapping from file names to the indices of embeddings, as follows: ```{'<index in features arrays>':{'domain':<website_name>, 'name':<img_file_name>}}```
			- *imgs_to_keep_idx*: indices of images that are not exact matches of the query. For pristine examples, we exclude images that have the same perceptual hashes as the query. We use this array during training to keep only the non-exact matches. For falsified examples, we consider all visual evidence images without filtering.
			- *detected_labels*: the labels detected in each visual evidence images using label detection APIs. Organized as follows: ```{'<index in features arrays>': {'labels':[found labels],'scores': [scores of labels]}```
	- Note that odd-indexed instances (i) have the same captions of the previous even instances (i-1). Therefore, we only have the results saved once. During training, we assign the odd instances visual evidence to their even counterparts. 
	- The validation and test splits are in val, and test directories, respectively. The training data is split into 4 directories. 

- Inverse search (textual evidence):
	- Each instance has its own directory, under which you can find the textual evidence and its embeddings, and the query embeddings, as follows:
		- Textual evidence:
			- *inverse_annotation.json*: raw search results as per the dataset collection pipeline, contains entities, the pages found by the inverse search API, and the crawled captions if found.
			- *caption_features2.npz*: sentence embeddings features of captions (text is preprocessed to remove some non-english letters or html artifacts, 'caption_features.npz' text is not preprocessed). 
			- *captions_clip_features.npz*: sentence embeddings of clip, we don't use that in our final model.
			- *entities_features2.npz*: sentence embeddings features of entities (text is preprocessed)
			- *entities_clip_features.npz*: sentence embeddings of clip, we don't use that in our final model.
			- *entities_binary_feature2.npz*: binary features of whether there is a named entity overlap between each entity and the query caption.
			- *captions_binary_feature2.npz*: binary features of whether there is a named entity overlap between each caption and the query caption.
			- *captions_info*: the captions and their domains. organized as: {'captions': [captions], 'domains:{'index':domain_name}}. The order of captions/domains are the same as the order in the feature arrays.
			- *captions_to_keep_idx*: indices of captions that are not an exact match to the query. For pristine examples, we exclude exact matches. For falsified examples, we keep all.
			- Note that if no captions were found, all caption-related files will not exist in the directory. 
		- In addition to textual evidence, there are other query-related files:
			- *qImg_resnet_features.npz*: resnet features of the query image
			- *qImg_places_resnet_features.npz* places resnet features of the query image
			- *qImg_clip_features*: clip features of the query image
			- *query_detected_labels*: detected labels in the query image
			- *labels_overlap.npz*: absolute count of overlapping labels between the query image and each visual evidence image
			- *labels_overlap_percentage*: percentage of overlapping labels between the query image and each visual evidence image.
			- *qCap_clip_features.npz*: clip features of query caption
			- *qCap_sbert_features.npz*: sentence embeddings of the query caption
	- The validation and test splits are in val, and test directories, respectively. The training data is split into 6 directories. 

- We have a ```dataset_items_<split>.json``` file, that lists the dataset items summarizes the paths for each instance. Organized as follows: ```{<index_in_newsclippings>: {'direct_path': <path to the instance directory of direct search>, 'inv_path':  <path to the instance directory of inverse search>}}```, this already takes care of having different directories for the training data, and for assigning direct search directories to odd-indexed instances.
- We have a *domain_to_idx_dict.json* file that maps the domains of the websites to #ids.
- For more information on how to read the raw images and captions, please refer to our custom data loader for embeddings precomputing in [data_preprocessing](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/data_preprocessing). 
- For more information on how to read the embeddings during training, please refer to our custom data loader and collate functions for training. 
