{:refdef: style="text-align: center;"}
![teaser](teaser.PNG){:width="120%"}
{: refdef}

### Abstract
*Misinformation is now a major problem due to its potential high risks to our core democratic and societal values and orders. Out-of-context misinformation is one of the easiest and effective ways used by adversaries to spread viral false stories. In this threat, a real image is re-purposed to support other narratives by misrepresenting its context and/or elements. The internet is being used as the go-to way to verify information using different sources and modalities. Our goal is an inspectable method that automates this time-consuming and reasoning-intensive process by fact-checking the image-caption pairing using Web evidence. To integrate evidence and cues from both modalities, we introduce the concept of 'multi-modal cycle-consistency check'; starting from the image/caption, we gather textual/visual evidence, which will be compared against the other paired caption/image, respectively. Moreover, we propose a novel architecture, Consistency-Checking Network (CCN), that mimics the layered human reasoning across the same and different modalities: the caption vs. textual evidence, the image vs. visual evidence, and the image vs. caption. Our work offers the first step and benchmark for open-domain, content-based, multi-modal fact-checking, and significantly outperforms previous baselines that did not leverage external evidence.*

### Dataset
- We base our evidence dataset on the [NewsCLIPpings Dataset](https://github.com/g-luo/news_clippings), that is based on the [VisualNews Dataset](https://github.com/FuxiaoLiu/VisualNews-Repository).
- Request/download both datasets to start. 
- We offer our evidence dataset in two formats: [links of evidence items](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/evidence_links), and the already-crawled evidence. 
- If you want to re-crawl our evidence from the links, or crawl evidence for other subsets of the [NewsCLIPpings Dataset](https://github.com/g-luo/news_clippings), we share our [dataset collection pipeline](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/dataset_collection), where you can find our custom crawler. 
- If you want to access our already-crawled evidence dataset, please requrest access via this [form](https://forms.gle/HZeUK1EEveGF9yEV9).

### Questions 
For any questions, please contact: sahar.abdelnabi@cispa.de

### Citation ###
If you find our code or dataset helpful, please cite our paper:
```javascript
@inproceedings{abdelnabi2021open,
    title = {Open-Domain, Content-based, Multi-modal Fact-checking of Out-of-Context Images via Online Resources},
    author = {Sahar Abdelnabi, Rakibul Hasan, Mario Fritz},
    booktitle = {CVPR},
    year = {2022}
}
```



