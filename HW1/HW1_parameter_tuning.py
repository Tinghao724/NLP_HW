# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import numpy as np
from collections import Counter
import pickle as pkl


import numpy as np
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_vocab(all_tokens,max_vocab_size,PAD_IDX=0,UNK_IDX=1):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token


def token2index_dataset(tokens_data,UNK_IDX=1):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data

class NewsGroupDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, target_list):
        """
        @param data_list: list of newsgroup tokens 
        @param target_list: list of newsgroup targets 

        """
        self.data_list = data_list
        self.target_list = target_list
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        
        token_idx = self.data_list[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]

def newsgroup_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[1])), 
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
    return [torch.from_numpy(np.array(data_list)), torch.LongTensor(length_list), torch.LongTensor(label_list)]

# First import torch related libraries

class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, emb_dim):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(BagOfWords, self).__init__()
        # pay attention to padding_idx 
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim,20)
    
    def forward(self, data, length):
        """
        
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0],1).expand_as(out).float()
     
        # return logits
        out = self.linear(out.float())
        return out


# Function for testing the model
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = data, lengths, labels
        outputs = F.softmax(model(data_batch, length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)

# Load Test data
train_targets = pd.read_csv("train_targets.csv").iloc[:,0].tolist()
val_targets = pd.read_csv("val_targets.csv").iloc[:,0].tolist()
test_targets = pd.read_csv("test_targets.csv").iloc[:,0].tolist()



# Then, load preprocessed train, val and test datasets
train_data_tokens = pkl.load(open("train_data_tokens_1.p", "rb"))
all_train_tokens = pkl.load(open("all_train_tokens_1.p", "rb"))

val_data_tokens = pkl.load(open("val_data_tokens_1.p", "rb"))
test_data_tokens = pkl.load(open("test_data_tokens_1.p", "rb"))


## Tuning the optimal values for max_vocab_size and MAX_SENTENCE_LENGTH

#max_vocab_sizes = [5000,10000, 15000, 20000, 30000]
#MAX_SENTENCE_LENGTHs =[100, 200, 300]


max_vocab_sizes = [5000,6000,7000,8000,9000,10000]
MAX_SENTENCE_LENGTHs =[200]


val_accs=[]

for mv in max_vocab_sizes:
    for lm in MAX_SENTENCE_LENGTHs: 
        
        MAX_SENTENCE_LENGTH = lm
        
        token2id, id2token = build_vocab(all_train_tokens,mv)
        
        train_data_indices = token2index_dataset(train_data_tokens)
        val_data_indices = token2index_dataset(val_data_tokens)
        test_data_indices = token2index_dataset(test_data_tokens)
        
        
        
        
        
        BATCH_SIZE = 32
        train_dataset = NewsGroupDataset(train_data_indices, train_targets)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   collate_fn=newsgroup_collate_func,
                                                   shuffle=True)
        
        val_dataset = NewsGroupDataset(val_data_indices, val_targets)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   collate_fn=newsgroup_collate_func,
                                                   shuffle=True)
        
        test_dataset = NewsGroupDataset(test_data_indices, test_targets)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   collate_fn=newsgroup_collate_func,
                                                   shuffle=False)
        
        
        
        
        emb_dim = 100
        model = BagOfWords(len(id2token), emb_dim)
        
        
        
        learning_rate = 0.01
        num_epochs = 5 # number epoch to train
        
        # Criterion and Optimizer
        criterion = torch.nn.CrossEntropyLoss()  
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        
        
        for epoch in range(num_epochs):
            for i, (data, lengths, labels) in enumerate(train_loader):
                model.train()
                data_batch, length_batch, label_batch = data, lengths, labels
                optimizer.zero_grad()
                outputs = model(data_batch, length_batch)
                loss = criterion(outputs, label_batch)
                loss.backward()
                optimizer.step()
                # validate every 100 iterations
                if i > 0 and i % 100 == 0:
                    # validate
                    val_acc = test_model(val_loader, model)
                    print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format( 
                               epoch+1, num_epochs, i+1, len(train_loader), val_acc))
    
        
        val_temp = test_model(val_loader, model)

        val_accs.append(val_temp)

        print ("Val Acc {}".format(val_temp))
        

### Tuning the parameters for batch size
BATCH_SIZEs = [70,75,80,85,90,95,100]

val_accs=[]

for b in BATCH_SIZEs:
    mv = 7000
    MAX_SENTENCE_LENGTH = 200
    
    token2id, id2token = build_vocab(all_train_tokens,mv)
    
    train_data_indices = token2index_dataset(train_data_tokens)
    val_data_indices = token2index_dataset(val_data_tokens)
    test_data_indices = token2index_dataset(test_data_tokens)
    
    
    
    
    
    BATCH_SIZE = b
    train_dataset = NewsGroupDataset(train_data_indices, train_targets)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=newsgroup_collate_func,
                                               shuffle=True)
    
    val_dataset = NewsGroupDataset(val_data_indices, val_targets)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=newsgroup_collate_func,
                                               shuffle=True)
    
    test_dataset = NewsGroupDataset(test_data_indices, test_targets)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=newsgroup_collate_func,
                                               shuffle=False)
    
    
    
    
    emb_dim = 100
    model = BagOfWords(len(id2token), emb_dim)
    
    
    
    learning_rate = 0.01
    num_epochs = 5 # number epoch to train
    
    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    
    for epoch in range(num_epochs):
        for i, (data, lengths, labels) in enumerate(train_loader):
            model.train()
            data_batch, length_batch, label_batch = data, lengths, labels
            optimizer.zero_grad()
            outputs = model(data_batch, length_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            # validate every 100 iterations
            if i > 0 and i % 100 == 0:
                # validate
                val_acc = test_model(val_loader, model)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format( 
                           epoch+1, num_epochs, i+1, len(train_loader), val_acc))

    
    val_temp = test_model(val_loader, model)

    val_accs.append(val_temp)

    print ("Val Acc {}".format(val_temp))

### Tuning the parameters for embedding size
emb_dims =[10,15,20,25,30,90]


val_accs=[]

for e in emb_dims:
    mv = 7000
    MAX_SENTENCE_LENGTH = 200
    
    token2id, id2token = build_vocab(all_train_tokens,mv)
    
    train_data_indices = token2index_dataset(train_data_tokens)
    val_data_indices = token2index_dataset(val_data_tokens)
    test_data_indices = token2index_dataset(test_data_tokens)
    
    
    
    BATCH_SIZE =  85
    train_dataset = NewsGroupDataset(train_data_indices, train_targets)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=newsgroup_collate_func,
                                               shuffle=True)
    
    val_dataset = NewsGroupDataset(val_data_indices, val_targets)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=newsgroup_collate_func,
                                               shuffle=True)
    
    test_dataset = NewsGroupDataset(test_data_indices, test_targets)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=newsgroup_collate_func,
                                               shuffle=False)
    
    
    
    
    emb_dim = e
    model = BagOfWords(len(id2token), emb_dim)
    
    
    
    learning_rate = 0.01
    num_epochs = 5 # number epoch to train
    
    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    
    for epoch in range(num_epochs):
        for i, (data, lengths, labels) in enumerate(train_loader):
            model.train()
            data_batch, length_batch, label_batch = data, lengths, labels
            optimizer.zero_grad()
            outputs = model(data_batch, length_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            # validate every 100 iterations
            if i > 0 and i % 100 == 0:
                # validate
                val_acc = test_model(val_loader, model)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format( 
                           epoch+1, num_epochs, i+1, len(train_loader), val_acc))

    
    val_temp = test_model(val_loader, model)

    val_accs.append(val_temp)

    print ("Val Acc {}".format(val_temp))


## Choose optimization methods


mv = 7000
MAX_SENTENCE_LENGTH = 200

token2id, id2token = build_vocab(all_train_tokens,mv)

train_data_indices = token2index_dataset(train_data_tokens)
val_data_indices = token2index_dataset(val_data_tokens)
test_data_indices = token2index_dataset(test_data_tokens)





BATCH_SIZE = 85 
train_dataset = NewsGroupDataset(train_data_indices, train_targets)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=newsgroup_collate_func,
                                           shuffle=True)

val_dataset = NewsGroupDataset(val_data_indices, val_targets)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=newsgroup_collate_func,
                                           shuffle=True)

test_dataset = NewsGroupDataset(test_data_indices, test_targets)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=newsgroup_collate_func,
                                           shuffle=False)




emb_dim = 20
model = BagOfWords(len(id2token), emb_dim)



learning_rate = 0.01
num_epochs = 25 # number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Rprep(model.parameters(), lr=learning_rate)



for epoch in range(num_epochs):
    for i, (data, lengths, labels) in enumerate(train_loader):
        model.train()
        data_batch, length_batch, label_batch = data, lengths, labels
        optimizer.zero_grad()
        outputs = model(data_batch, length_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()
        # validate every 100 iterations
        if i > 0 and i % 100 == 0:
            # validate
            val_acc = test_model(val_loader, model)
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format( 
                       epoch+1, num_epochs, i+1, len(train_loader), val_acc))


print ("Val Acc {}".format(test_model(val_loader, model)))
 
    




## Find tje optimal learning rate with Adam


learning_rates =[0.002]


val_accs=[]

for lrs in learning_rates:
    mv = 7000
    MAX_SENTENCE_LENGTH = 200
    
    token2id, id2token = build_vocab(all_train_tokens,mv)
    
    train_data_indices = token2index_dataset(train_data_tokens)
    val_data_indices = token2index_dataset(val_data_tokens)
    test_data_indices = token2index_dataset(test_data_tokens)
    
    
    
    BATCH_SIZE =  85
    train_dataset = NewsGroupDataset(train_data_indices, train_targets)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=newsgroup_collate_func,
                                               shuffle=True)
    
    val_dataset = NewsGroupDataset(val_data_indices, val_targets)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=newsgroup_collate_func,
                                               shuffle=True)
    
    test_dataset = NewsGroupDataset(test_data_indices, test_targets)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=newsgroup_collate_func,
                                               shuffle=False)
    
    
    
    
    emb_dim = 20
    model = BagOfWords(len(id2token), emb_dim)
    
    
    
    learning_rate = lrs
    num_epochs = 20 # number epoch to train
    
    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    
    for epoch in range(num_epochs):
        for i, (data, lengths, labels) in enumerate(train_loader):
            model.train()
            data_batch, length_batch, label_batch = data, lengths, labels
            optimizer.zero_grad()
            outputs = model(data_batch, length_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            # validate every 100 iterations
            if i > 0 and i % 100 == 0:
                # validate
                val_acc = test_model(val_loader, model)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format( 
                           epoch+1, num_epochs, i+1, len(train_loader), val_acc))

    
    val_temp = test_model(val_loader, model)

    val_accs.append(val_temp)

    print ("Val Acc {}".format(val_temp))


## Use adaptive learning with Adam


mv = 7000
MAX_SENTENCE_LENGTH = 200

token2id, id2token = build_vocab(all_train_tokens,mv)

train_data_indices = token2index_dataset(train_data_tokens)
val_data_indices = token2index_dataset(val_data_tokens)
test_data_indices = token2index_dataset(test_data_tokens)





BATCH_SIZE = 85 
train_dataset = NewsGroupDataset(train_data_indices, train_targets)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=newsgroup_collate_func,
                                           shuffle=True)

val_dataset = NewsGroupDataset(val_data_indices, val_targets)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=newsgroup_collate_func,
                                           shuffle=True)

test_dataset = NewsGroupDataset(test_data_indices, test_targets)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=newsgroup_collate_func,
                                           shuffle=False)




emb_dim = 20
model = BagOfWords(len(id2token), emb_dim)



learning_rate = 0.01
num_epochs = 15 # number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)

for epoch in range(num_epochs):
    scheduler.step()
    for i, (data, lengths, labels) in enumerate(train_loader):
        model.train()
        data_batch, length_batch, label_batch = data, lengths, labels
        optimizer.zero_grad()
        outputs = model(data_batch, length_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()
        # validate every 100 iterations
        if i > 0 and i % 100 == 0:
            # validate
            val_acc = test_model(val_loader, model)
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format( 
                       epoch+1, num_epochs, i+1, len(train_loader), val_acc))


print ("Val Acc {}".format(test_model(val_loader, model)))
 
    






