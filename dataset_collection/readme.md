# Dataset collection

- This directory contains our code for the dataset collection pipeline.

- This might be helpful if you need to crawl the evidence dataset from scratch, or if you need to crawl evidence for a different subset of the [NewsCLIPpings Dataset](https://github.com/g-luo/news_clippings).

## Main Requirments 
You will need to have the [NewsCLIPpings Dataset](https://github.com/g-luo/news_clippings) and the [VisualNews Dataset](https://github.com/FuxiaoLiu/VisualNews-Repository).
Your file structure should look like this: 
```javascript
news_clippings
│
└── data/
└── embeddings/

visual_news
│
└── origin/
│    └── data.json
│        ...
└── ...
```

In addition, you need the following dependencies: 

- beautifulsoup4=4.9.3
- google-cloud-vision
- google-api-python-client==2.3.0
- fasttext (you will need the language [identification model](https://fasttext.cc/docs/en/language-identification.html))
- pillow=7.1.2
- requests
- imagehash

We use Google Cloud Platform: [Custom Search](https://developers.google.com/custom-search/v1/introduction) to search for images given captions, and [Google vision](https://cloud.google.com/vision/docs/detecting-web) to detect web entities and perform inverse image search. You will need to enable these two APIs and have a way of credential authentication. For the [Custom Search](https://developers.google.com/custom-search/v1/introduction) you need to provide an API key and a custom search engine id. 

## Collection 
Our crawlers to extract captions and snippets from pages can be found in 'utils'.

### Direct search 
- Search for images given captions. 
- In our implementation, we also crawl the captions for these found images (however, we didn't use them in our model).
- To perform the search for images, run:

```javascript
python download_direct_annotations_dirs.py --hashing_cutoff <cutoff> --sub_split <split> --start_idx <start> --how_many <samples_num> --save_folder_path <folder_path>
```
- where:
  - *hashing_cutoff* is the cutoff of image hash when search with the content of the image to find the image tag in the html (if not found by the image url).
  - *sub_split* is which subsplit of the newsclippings to collect (train, val, test)
  - *start idx* the index from the newsclippings to start crawling from. 
  - *how many* how many samples to collect, you can instead specify an end index via '--end_idx'
  - *save_folder_path* save directory, this script will create sub-directories under it for each sample with the index name.
  - Specifying *start_idx* and *end_idx* (or *how_many*) can be helpful if you want to parallelize the collection 
 
- In case you want to resume the download, you can specify *--continue_download 1*, this will read the last downloaded index and will continue from it. 
- Output:
  - the script will create a directory for each even index in the newsclippings dataset (odd indices are repeated captions). Your file structure should look like: ```<args.save_folder_path>/<args.split_type>/direct_search/<args.sub_split>```, where *split_type* is the split type from newsclippings, *merged_balanced* in our case. Under which, a ```<split>.json``` file would be created that contains the already finished indicies. You will also find a sub-directory for each index.
  - Under each sub-directory, it saves the images, and a 'direct_annotation.json' file. It contains lists of: images with captions, images without captions, images with captions found by matching the content of images. Each sub-list contains: 'image link', 'page link', 'domain', 'snippet' (snippet from the html), 'image_path' (saved image path), 'page_title', 'captions' if found. 
  - This script also saved the html files of the containing pages. 

### Inverse search 
- Inverse search an image.
- Get the web entities. 
- Get pages containing the images. 
- Crawl captions from these pages.

- To perform the inverse search for images, run:
```javascript
python download_inverse_annotations_dirs.py --hashing_cutoff <cutoff> --sub_split <split> --start_idx <start> --how_many <samples_num> --save_folder_path <folder_path>
```
- parameters are similar to the direct search. 
- Output:
  - the script will create a directory for each index in the newsclippings dataset. Your file structure should look like: ```<args.save_folder_path>/<args.split_type>/inverse_search/<args.sub_split>```. 
  - For each sub-directory, you will find a *inverse_annotation.json*, that contains: *page_link*, *image_link*, *title*, *caption* if found.
