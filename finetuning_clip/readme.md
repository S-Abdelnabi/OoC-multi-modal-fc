### Overview

We here share our code for finetuning CLIP on the [NewsCLIPpings](https://arxiv.org/pdf/2104.05893.pdf) dataset (on pairs only, without considering evidence, a similar setup to the NewsCLIPpings paper).

### Requirements
- Follow the instructions in [here](https://github.com/g-luo/news_clippings) to set up the dataset, and [here] to set up [CLIP](https://github.com/openai/CLIP)
- We use pytorch=1.7.1, and python=3.7.6

### Training 
- To train, run:
```
python main_ft_clip.py --pdrop 0.05
```
### Evaluation 
- To evaluate, run:
```
python evaluate_clip.py --seed 1111
```
### Checkpoints
Our checkpoint can be found [here](https://drive.google.com/drive/folders/1x266t1uHutc5iZIE02hOCrVefwqZ2qfm?usp=sharing). 
