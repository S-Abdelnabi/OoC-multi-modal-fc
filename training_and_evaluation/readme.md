### Requirements
- pytorch=1.7.1
- transformers=4.8.1

### Training

- Our *CCN* training framwork assumes that we have the precomputed embeddings. You can find them in our collected dataset, or as done in the [preprocessing step](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/data_preprocessing)
- We have two variants of the model: 1) using sentence embeddings, 2) using BERT+LSTM.


#### [Using sentence embeddings](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/training/sent_emb)
```
python main_sent_emb.py --use_src --use_cap_memory --use_img_memory  --use_ent_memory --use_places_memory 
--inp_pdrop 0.05 --pdrop_mem 0.1 --consistency clip --pdrop 0 --nlayers 2 --domains_dim 20 --emb_pdrop 0.25 
--lr_sched cycle --lr 0.000009 --lr_max 0.00006 --epochs 60 
--filter_dup --binary_ner_ent --binary_ner_cap --labels_overlap
```
- You will need the *domain_to_idx_dict.json* file, in addition to ```dataset_items_<split>.json``` files (dict of dataset items, keys are NewsCLIPPings indices, each key has *direct_path* and *inv_path*). Should be placed in the same directory. The datasets are placed as: *../queries_dataset/*, *../visual_news/*, and *../news_clippings/*
- You can adjust the paths to the datasets and these files via the arguments.

#### [Using BERT+LSTM](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/training/bert_lstm)
```
python main_bert_lstm.py --use_src --use_cap_memory --use_img_memory  --use_ent_memory --use_places_memory 
--inp_pdrop 0.05 --pdrop_mem 0.2 --consistency clip --pdrop 0 --nlayers 2 --domains_dim 20 --emb_pdrop 0.25 
--lr_sched cycle --lr 0.000009 --lr_max 0.00006 --epochs 30 
--filter_dup --binary_ner_ent --binary_ner_cap --batch_size 32 --labels_overlap --lstm_maxlen 150
```
- You will need the *domain_to_idx_dict.json* file, in addition to ```dataset_items_<split>.json``` files (dict of dataset items, keys are NewsCLIPPings indices, each key has *direct_path* and *inv_path*). Should be placed in the same directory. The datasets are placed as: *../queries_dataset/*, *../visual_news/*, and *../news_clippings/*
- You can adjust the paths to the datasets and these files via the arguments.
- To evaluate independently of training, run:

```
python evaluate_bert_lstm.py --use_src --use_cap_memory --use_img_memory  --use_ent_memory --use_places_memory 
--inp_pdrop 0.05 --pdrop_mem 0.2 --consistency clip --pdrop 0 --nlayers 2 --domains_dim 20 --emb_pdrop 0.25 
--lr_sched cycle --lr 0.000009 --lr_max 0.00006 --epochs 30 
--filter_dup --binary_ner_ent --binary_ner_cap --batch_size 32 --labels_overlap --lstm_maxlen 150
```
  - This will also save under './attn_weights' the attention vectors for each memory type for each minibatch, which you can use to inspect the highest attention evidence items.


