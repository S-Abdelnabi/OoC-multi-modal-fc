## [Open-Domain, Content-based, Multi-modal Fact-checking of Out-of-Context Images via Online Resources](https://arxiv.org/pdf/2112.00061.pdf) 

- Authors: [Sahar Abdelnabi](https://scholar.google.de/citations?user=QEiYbDYAAAAJ&hl=en), [Rakibul Hasan](https://rakib062.github.io/), [Mario Fritz](https://cispa.saarland/group/fritz/)
- CVPR'22
- This repository contains code to reproduce our dataset collection and training for our paper. Detailed instructions can be found in each subdirectory. 
- - -
### Abstract ###
*Misinformation is now a major problem due to its potential high risks to our core democratic and societal values and orders. Out-of-context misinformation is one of the easiest and effective ways used by adversaries to spread viral false stories. In this threat, a real image is re-purposed to support other narratives by misrepresenting its context and/or elements. The internet is being used as the go-to way to verify information using different sources and modalities. Our goal is an inspectable method that automates this time-consuming and reasoning-intensive process by fact-checking the image-caption pairing using Web evidence. To integrate evidence and cues from both modalities, we introduce the concept of 'multi-modal cycle-consistency check'; starting from the image/caption, we gather textual/visual evidence, which will be compared against the other paired caption/image, respectively. Moreover, we propose a novel architecture, Consistency-Checking Network (CCN), that mimics the layered human reasoning across the same and different modalities: the caption vs. textual evidence, the image vs. visual evidence, and the image vs. caption. Our work offers the first step and benchmark for open-domain, content-based, multi-modal fact-checking, and significantly outperforms previous baselines that did not leverage external evidence.*

<p align="center">
<img src="https://github.com/S-Abdelnabi/OoC-multi-modal-fc/blob/gh-pages/teaser.PNG" width="850">
</p>

- - -
### Dataset collection ###
We [here](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/dataset_collection) share our dataset collection pipeline that you can use to download the dataset from scratch or to download other subsets from the NewsCLIPpings dataset.
- - -

### Evidence links ###
We share the links that resulted from the Google search we performed using query images and captions. You can find [here](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/evidence_links) more details about how to get them and their format. You can adapt the [crawler pipeline](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/dataset_collection) to extract and download the evidence from these links. 
- - -
### Dataset access and description ###
If you would like to access our already-collected evidence (along with the preprocessing and precomputed embeddings), please find more details under [curated_dataset](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/curated_dataset).
- - -
### Dataset preprocessing and embeddings computation ###
You can find our pipeline for preprocessing the data and computing the embeddings [data_preprocessing](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/data_preprocessing). If you are using our collected evidence dataset, you can skip this step. 
- - -

### Training and evaluation of *CCN*
We share our [training and evaluation code](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/blob/main/training_and_evaluation) for two setups: 1) Training using sentence embeddings. 2) Training using BERT+LSTM. 

- - -
### Fine-tuning CLIP
You can find our code to finetune CLIP in [finetuning_clip](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/finetuning_clip).
- - -

### Checkpoints 

Checkpoints can be found [here](https://drive.google.com/drive/folders/1x266t1uHutc5iZIE02hOCrVefwqZ2qfm?usp=sharing).
- - -
### Citation ###

- If you find this code or dataset helpful, please cite our paper:
```javascript
@inproceedings{abdelnabi2021open,
    title = {Open-Domain, Content-based, Multi-modal Fact-checking of Out-of-Context Images via Online Resources},
    author = {Sahar Abdelnabi, Rakibul Hasan, Mario Fritz},
    booktitle = {CVPR},
    year = {2022}
}
```
