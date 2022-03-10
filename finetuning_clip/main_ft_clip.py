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
train_dataset = dataset_mismatch.NewsClipDataset(args.visual_news_root, args.news_clip_root, 'train', preprocess)
val_dataset = dataset_mismatch.NewsClipDataset(args.visual_news_root, args.news_clip_root, 'val', preprocess)
test_dataset = dataset_mismatch.NewsClipDataset(args.visual_news_root, args.news_clip_root, 'test', preprocess)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn = collate_mismatch, num_workers=args.num_workers, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = collate_mismatch, num_workers=args.num_workers,  pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = collate_mismatch, num_workers=args.num_workers,  pin_memory=True)

#create exp folder if it has not been created 
if not os.path.isdir(args.exp_folder):
    os.makedirs(args.exp_folder)

#resume training
stored_loss = 100000000   
stored_acc = 0

classifier_list = ['classifier.weight', 'classifier.bias']
classifier_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in classifier_list, classifier_clip.named_parameters()))))
base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in classifier_list, classifier_clip.named_parameters()))))

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD([{'params': base_params}, {'params': classifier_params, 'lr': args.lr_classifier}], lr=args.lr_clip, weight_decay=args.wdecay, momentum=args.sgd_momentum)    
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam([{'params': base_params}, {'params': classifier_params, 'lr': args.lr_classifier}], lr=args.lr_clip, weight_decay=args.wdecay) 

if args.resume:
    if os.path.isfile(args.resume):
        log_file_val_loss = open(os.path.join(args.exp_folder,'log_file_val.txt'),'a') 
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        if 'best_val_loss' in checkpoint: stored_loss = checkpoint['best_val_loss']
        if 'best_val_acc' in checkpoint: stored_acc = checkpoint['best_val_acc']
        classifier_clip.load_state_dict(checkpoint['state_dict'])
        clip.model.convert_weights(classifier_clip)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    log_file_val_loss = open(os.path.join(args.exp_folder,'log_file_val.txt'),'w') 
#define loss function
criterion = nn.BCEWithLogitsLoss().cuda()



params = list(classifier_clip.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


def evaluate(loader):
    total_loss = 0
    correct = 0
    total_num = 0
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
            correct += pred.eq(truth).sum().item()
            total_num += labels.size(0) 
    avg_loss = total_loss/len(loader)    
    acc = (correct/total_num)*100
    return avg_loss, acc 
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None: p.grad.data = p.grad.data.float() 
    
def train():
    total_loss = 0
    start_time = time.time()
    classifier_clip.train()
    for (idx, batch) in enumerate(train_dataloader):
        #prepare batches 
        labels = batch[0].cuda()
        imgs = batch[1].cuda() 
        captions = batch[2].cuda()
        #captions_tokenized = base_clip.tokenize(captions_text).cuda()
        #forward 
        output = classifier_clip(imgs,captions) 
        #compute loss     
        loss = criterion(output, torch.unsqueeze(labels.float(), 1))
        total_loss += loss.item()
    
        #backward and optimizer step 
        optimizer.zero_grad()
        loss.backward()
        convert_models_to_fp32(classifier_clip)
        optimizer.step()
        clip.model.convert_weights(classifier_clip)
        #log    
        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.8f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                    epoch, idx, len(train_dataloader) , optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
            
     
try:
#you can exit from training by: ctrl+c
    for epoch in range(args.start_epoch, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss, val_acc = evaluate(val_dataloader)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.2f} | '
                'val acc {:8.2f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss , val_acc))
        print('-' * 89)
        log_file_val_loss.write(str(val_loss) + ', '+ str(val_acc)+ '\n')
        log_file_val_loss.flush()
        if val_loss < stored_loss:
            print('New best model loss')
            stored_loss = val_loss
            torch.save({'epoch': epoch,
                        'state_dict': classifier_clip.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'best_val_loss': stored_loss,
                        'best_val_acc': stored_acc},
                        os.path.join(args.exp_folder, 'best_model_loss.pth.tar'))
        if val_acc > stored_acc:
            print('New best model acc')
            stored_acc = val_acc
            torch.save({'epoch': epoch,
                        'state_dict': classifier_clip.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'best_val_loss': stored_loss,
                        'best_val_acc': stored_acc},
                        os.path.join(args.exp_folder, 'best_model_acc.pth.tar'))
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    
# Load the best saved model if exists and run for the test data.
if os.path.isfile(os.path.join(args.exp_folder, 'best_model_loss.pth.tar')):
    checkpoint = torch.load(os.path.join(args.exp_folder, 'best_model_loss.pth.tar'))
    classifier_clip.load_state_dict(checkpoint['state_dict'])
    clip.model.convert_weights(classifier_clip)
    print("=> loaded checkpoint: '{}')".format(os.path.join(args.exp_folder, 'best_model_loss.pth.tar')))
    # Run on test data.
    test_loss, test_acc = evaluate(test_dataloader)
    print('=' * 89)
    print('| End of training | best loss | test loss {:5.2f} | test acc {:8.3f}'.format(
    test_loss, test_acc))
    print('=' * 89)

# Load the best saved model if exists and run for the test data.
if os.path.isfile(os.path.join(args.exp_folder, 'best_model_acc.pth.tar')):
    checkpoint = torch.load(os.path.join(args.exp_folder, 'best_model_acc.pth.tar'))
    classifier_clip.load_state_dict(checkpoint['state_dict'])
    clip.model.convert_weights(classifier_clip)
    print("=> loaded checkpoint: '{}')".format(os.path.join(args.exp_folder, 'best_model_acc.pth.tar')))
    # Run on test data.
    test_loss, test_acc = evaluate(test_dataloader)
    print('=' * 89)
    print('| End of training |  best acc | test loss {:5.2f} | test acc {:8.3f}'.format(
    test_loss, test_acc))
    print('=' * 89)