## Search results 

We provide the links to the search results we got when searching with items (images and captions) from the NewsCLIPpings dataset. You can find them [here](https://drive.google.com/drive/folders/1xUylNApvY4I2aaVB2e3IbRR4_zNQvTmb?usp=sharing).

### Description
- There are three files for the 3 sub-splits (train,val,test) from the 'merged_balanced' split of the NewsCLIPpings dataset.
- Each file has the followings:
  - The json file keys are the index of the sample in the NewsCLIPpings dataset (note that there might be some few indices missing, these are the indicies where we did not have any search results).
  - Each sample has: 
      - *label*: 1 for falsified, 0 for pristine
      - *entities*: List of entities retured by the [Google inverse search API](https://cloud.google.com/vision/docs/detecting-web).
      - *links_direct_search*: the images we got when searching with the caption. This is a list of the search results. Each item in the last has the following: image link, containing page link.
      - *links_inv_search*: the images we got when performing the inverse image search. This is a list of the search results. Each item in the last has the following: image link, containing page link. These are the 'fully matched images' and the 'partially matched images', as returned by the API, note that we don't include 'visually similar images' that were returned by the API because they might belong to different events. Note that this list might be empty if the image was not found by the API.
- These results are the raw search results we got. Later after processing, we removed some of these results if they had an exact match with the query (for pristine examples).
- If you need to crawl captions/titles for the images, you can refer to our crawler in the [dataset collection pipeline](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/dataset_collection)

