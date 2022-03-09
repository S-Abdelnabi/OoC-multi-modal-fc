### Requirements
- pytorch=1.7.1
- transformers=4.8.1

### Training

- Our training framwork assumes that we have the precomputed embeddings. You can find them in our collected dataset, or as done in the [preprocessing step](https://github.com/S-Abdelnabi/OoC-multi-modal-fc/tree/main/data_preprocessing)


#### Training using the precomputed embeddings (including sentence embeddings)
```
python main_sent_emb.py --use_src --use_cap_memory --use_img_memory  --use_ent_memory --use_places_memory 
--inp_pdrop 0.05 --pdrop_mem 0.1 --consistency clip --pdrop 0 --nlayers 2 --domains_dim 20 --emb_pdrop 0.25 
--lr_sched cycle --lr 0.000009 --lr_max 0.00006 --epochs 60 
--filter_dup --binary_ner_ent --binary_ner_cap --labels_overlap
```
- You will need the *domain_to_idx_dict.json* file, in addition to ```dataset_items_<split>.json``` files (dict of dataset items, keys are NewsCLIPPings indices, each key has *'direct_path* and *inv_path*). 
- You can adjust the paths to the datasets and these files via the arguments.


