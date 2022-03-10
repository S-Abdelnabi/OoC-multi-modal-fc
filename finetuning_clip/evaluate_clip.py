from custom_collate import collate_mismatch
import dataset_mismatch  
import json 
import os 
import torch
import torchvision
import torch.nn as nn
import numpy as np 
from torch.utils.data import DataLoader
import argparse
import io
import clip_classifier
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import time 
import clip 
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='Training using the precomputed embeddings')
##### locations #####  
parser.add_argument('--visual_news_root', type=str, default='../visual_news/origin/',
                    help='location to the root folder of the visualnews dataset')
parser.add_argument('--news_clip_root', type=str, default='../news_clippings/data/merged_balanced/',
                    help='location to the root folder of the clip dataset')               
parser.add_argument('--exp_folder', type=str, default='./exp/',
                    help='path to the folder to log the output and save the models')
                    
###### model details ########                    
parser.add_argument('--pdrop', type=float, default=0.5,
                    help='dropout probability')


##### Training details #####
parser.add_argument('--batch_size', type=int, default=64,
                    help='dimension of domains embeddings') 
parser.add_argument('--num_workers', type=int, default=6,
                    help='number of data loaders workers') 
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--log_interval', type=int, default=200,
                    help='how many batches')
parser.add_argument('--resume', type=str, default = '', help='path to model')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='which optimizer to use')
parser.add_argument('--lr_clip', type=float, default= 5e-7,
                    help='learning rate of the clip model')
parser.add_argument('--lr_classifier', type=float, default=5e-5,
                    help='learning rate of the clip model')                    
parser.add_argument('--sgd_momentum', type=float, default=0.9,
                    help='momentum when using sgd')                      
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--wdecay', default=1.2e-6, type=float,
                        help='weight decay pow (default: -5)')
                    
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#### load input files ####



#### settings of the model ####
model_settings = {'pdrop': args.pdrop}
base_clip, preprocess = clip.load("ViT-B/32", device="cuda")
classifier_clip = clip_classifier.ClipClassifier(model_settings,base_clip)
classifier_clip.cuda()

#### load Datasets and DataLoader ####
val_dataset = dataset_mismatch.NewsClipDataset(args.visual_news_root, args.news_clip_root, 'val', preprocess)
test_dataset = dataset_mismatch.NewsClipDataset(args.visual_news_root, args.news_clip_root, 'test', preprocess)

val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = collate_mismatch, num_workers=args.num_workers,  pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = collate_mismatch, num_workers=args.num_workers,  pin_memory=True)


#resume training
stored_loss = 100000000   
stored_acc = 0

classifier_list = ['classifier.weight', 'classifier.bias']
classifier_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in classifier_list, classifier_clip.named_parameters()))))
base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in classifier_list, classifier_clip.named_parameters()))))

 
#define loss function
criterion = nn.BCEWithLogitsLoss().cuda()

params = list(classifier_clip.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


def evaluate(loader):
    true_label = []
    prediction = []
    total_loss = 0
    correct = 0
    correct_falsified = 0
    correct_match = 0  
    
    total_num = 0
    total_num_falsified = 0
    total_num_match = 0
    
    classifier_clip.eval()
    for (idx, batch) in enumerate(loader):
        labels = batch[0].cuda()
        imgs = batch[1].cuda() 
        captions = batch[2].cuda()
        with torch.no_grad():
            #captions_tokenized = base_clip.tokenize(captions_text).cuda()
            #forward 
            output = classifier_clip(imgs,captions) 
            #compute loss     
            loss = criterion(output, torch.unsqueeze(labels.float(), 1))
            total_loss += loss.item()
            #compute correct predictions 
            pred = torch.sigmoid(output) >= 0.5
            truth = torch.unsqueeze(labels,1)>=0.5
            #print(1 if pred else 0)
            true_label.append(1 if truth else 0)
            prediction.append(1 if pred else 0)
            correct += pred.eq(truth).sum().item()
            total_num += labels.size(0) 
            index_ones = ((labels == 1).nonzero(as_tuple=True)[0])
            correct_falsified += pred[index_ones,:].eq(truth[index_ones,:]).sum().item()  
            total_num_falsified += index_ones.size(0)

            index_zeros = ((labels == 0).nonzero(as_tuple=True)[0])
            correct_match += pred[index_zeros,:].eq(truth[index_zeros,:]).sum().item()   
            total_num_match += index_zeros.size(0)
            
    avg_loss = total_loss/len(loader)    
    acc = (correct/total_num)*100
    acc_falsified = (correct_falsified/total_num_falsified)*100
    acc_match = (correct_match/total_num_match)*100
    f1 = f1_score(true_label,prediction)
    return avg_loss, acc, acc_falsified, acc_match,f1 
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None: p.grad.data = p.grad.data.float() 
    
    
# Load the best saved model if exists and run for the test data.
if os.path.isfile(os.path.join(args.exp_folder, 'best_model_loss.pth.tar')):
    checkpoint = torch.load(os.path.join(args.exp_folder, 'best_model_loss.pth.tar'))
    classifier_clip.load_state_dict(checkpoint['state_dict'])
    clip.model.convert_weights(classifier_clip)
    print("=> loaded checkpoint: '{}')".format(os.path.join(args.exp_folder, 'best_model_loss.pth.tar')))
    # Run on test data.
    test_loss, test_acc, acc_false, acc_match, f1 = evaluate(test_dataloader)
    print('=' * 89)
    print('| End of training | best loss | test loss {:5.2f} | test acc {:8.3f} | false acc {:8.3f} | match acc {:8.3f} | F1 {:8.3f}'.format(
    test_loss, test_acc, acc_false, acc_match, f1))
    print('=' * 89)

# Load the best saved model if exists and run for the test data.
if os.path.isfile(os.path.join(args.exp_folder, 'best_model_acc.pth.tar')):
    checkpoint = torch.load(os.path.join(args.exp_folder, 'best_model_acc.pth.tar'))
    classifier_clip.load_state_dict(checkpoint['state_dict'])
    clip.model.convert_weights(classifier_clip)
    print("=> loaded checkpoint: '{}')".format(os.path.join(args.exp_folder, 'best_model_acc.pth.tar')))
    # Run on test data.
    test_loss, test_acc, acc_false, acc_match, f1 = evaluate(test_dataloader)
    print('=' * 89)
    print('| End of training | best loss | test loss {:5.2f} | test acc {:8.3f} | false acc {:8.3f} | match acc {:8.3f} | F1 {:8.3f}'.format(
    test_loss, test_acc, acc_false, acc_match, f1))
    print('=' * 89)