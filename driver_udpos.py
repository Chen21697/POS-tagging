# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torchtext import data
from torchtext import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
TEXT = data.Field(lower = True)
UD_TAGS = data.Field(unk_token = None)

fields = (("text", TEXT), ("udtags", UD_TAGS))
train_data, valid_data, test_data = datasets.UDPOS.splits(fields)

#%%
sentences = [["The", "old", "man", "the", "boat", "."],
             ["The", "complex", "houses", "married", "and", "single", "soldiers", "and", "their", "families", "."],
             ["The", "man", "who", "hunts", "ducks", "out", "on", "weekends", "."]]

#%%
# build the vocabuurary
TEXT.build_vocab(train_data, vectors = "glove.6B.100d")
UD_TAGS.build_vocab(train_data)

#%%
# build the batch iteratros
BATCH_SIZE = 128

train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), 
                                                               batch_size = BATCH_SIZE,
                                                               device = device)
#%%
# model structure
class Bi_LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layer, output_dim, pad_idx) :
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        self.bi_lstm = nn.LSTM(input_size = embedding_dim,
                               hidden_size = hidden_dim,
                               num_layers = num_layer,
                               bidirectional = True)
        self.hidden_1 = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text):
        embedding_out = self.embedding(text)
        lstm_out, _ = self.bi_lstm(embedding_out)
        hidden_1_out = self.hidden_1(lstm_out)
        return hidden_1_out

#%%
def build_model(text, tags):
    
    input_dim = len(text.vocab)
    output_dim = len(tags.vocab)
    # embedding dimension depends on GloVe
    embedding_dim = 100
    hidden_dim = 64
    num_layer = 2
    pad_idx = text.vocab.stoi[text.pad_token]
    
    model = Bi_LSTM(input_dim,
                    embedding_dim,
                    hidden_dim,
                    num_layer,
                    output_dim,
                    pad_idx)
    
    pretrained_embeddings = text.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim).to(device)
    return model
def tag_percentage(tag_counts):
    total_count = sum([count for tag, count in tag_counts])
    tag_counts_percentages = [(tag, count, count/total_count) for tag, count in tag_counts]
    return tag_counts_percentages

def training(model, train_iter, optimizer, crit, tag_pad_idx):
    model.train()
    epoch_train_loss = 0
    epoch_train_acc = 0

    for t_batch in train_iter:
        
        t_b_text = t_batch.text
        t_b_tags = t_batch.udtags
        
        t_pred = model(t_b_text)
        
        # transform format
        t_pred = t_pred.reshape(-1, t_pred.shape[-1])
        t_b_tags = t_b_tags.reshape(-1)

        t_loss = crit(t_pred, t_b_tags)
        t_loss.backward()
        optimizer.step()
        
        epoch_train_loss += t_loss.item()
        t_acc = accuracy(t_pred, t_b_tags, 0)
        
        epoch_train_acc += t_acc.item()
        
    return epoch_train_loss/len(train_iter), epoch_train_acc/len(train_iter)

def validation_testing_evaluation(model, valid_iter, crit, tag_pad_idx):
    model.eval()
    epoch_valid_loss = 0
    epoch_valid_acc = 0

    for v_batch in valid_iter:
        
        v_b_text = v_batch.text
        v_b_tags = v_batch.udtags
        
        v_pred = model(v_b_text)
        
        # transform format
        v_pred = v_pred.reshape(-1, v_pred.shape[-1])
        v_b_tags = v_b_tags.reshape(-1)
        
        v_loss = crit(v_pred, v_b_tags)
        
        epoch_valid_loss += v_loss.item()
        v_acc = accuracy(v_pred, v_b_tags, 0)
        
        epoch_valid_acc += v_acc.item()
        
    return epoch_valid_loss/len(valid_iter), epoch_valid_acc/len(valid_iter)


def accuracy(y, y_head, tag_pad_idx):
    y = y.argmax(dim = 1, keepdim = True)
    non_pad_elements = (y_head != tag_pad_idx).nonzero()
    y_head_2 = y_head[non_pad_elements]
    correct_count = (y[non_pad_elements].squeeze(1) == y_head_2).sum()
    
    return correct_count / torch.FloatTensor([y_head[non_pad_elements].shape[0]]).to(device)

def plot_train_valid_loss(t_loss, v_loss, epochs):
    fig = plt.figure()
    plt.plot(range(1, epochs+1), t_loss, c='b', label='training loss')
    plt.plot(range(1, epochs+1), v_loss, c='g', label='validation loss')
    plt.xticks(range(1, epochs+1))
    plt.yticks(np.arange(0, 2.5, 0.25))
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(linestyle = '--')
    plt.legend()
    plt.show()
    
def plot_train_valid_acc(t_acc, v_acc, epochs):
    fig = plt.figure()
    plt.plot(range(1, epochs+1), t_acc, c='b', label='training acc')
    plt.plot(range(1, epochs+1), v_acc, c='g', label='validation acc')
    plt.xticks(range(1, epochs+1))
    #plt.yticks(np.arange(0.2, 1, 0.))
    
    plt.xlabel('epoch')
    plt.ylabel('acc %')
    plt.grid(linestyle = '--')
    plt.legend()
    plt.show()
    
def tag_sentence(sentence, model, text, tags):
    model.eval()
    numericalized = [text.vocab.stoi[x] for x in sentence]
    token_tensor = torch.LongTensor(numericalized).unsqueeze(-1).to(device)
    
    y = model(token_tensor)
    y = y.argmax(-1)
    y_head = [tags.vocab.itos[x.item()] for x in y]
    return y_head
def plot_percentage(tags):
    x = []
    y = []
    z = []
    for tag, count, percent in tags:
        x.append(tag)
        y.append(count)
        z.append(percent*100)

    objects = (x)
    y_pos = np.arange(len(x))
    performance = z
    
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.show()
#%%
if __name__== "__main__":
    
    epochs = 10
    #lr = 0.003
    
    model = build_model(TEXT, UD_TAGS).to(device)
    
    tag_pad_idx = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
    crit = nn.CrossEntropyLoss(ignore_index = tag_pad_idx).to(device)
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params = parameters)
    t_loss_history = []
    v_loss_history = []
    
    t_acc_history = []
    v_acc_history = []
    
    best_valid_loss = float('inf')
    # start training
    for i in range(epochs):
        t_loss, t_acc = training(model, train_iter, optimizer, crit, tag_pad_idx)
        v_loss, v_acc = validation_testing_evaluation(model, valid_iter, crit, tag_pad_idx)
        
        t_loss_history.append(t_loss)
        v_loss_history.append(v_loss)
        
        t_acc_history.append(t_acc)
        v_acc_history.append(v_acc)
        
        print("epoch:", i+1, "t_loss", t_loss,  "v_loss", v_loss, "t_acc", t_acc, "v_acc", v_acc)
        
        # earily stopping
        if v_loss < best_valid_loss:
            best_valid_loss = v_loss
            torch.save(model.state_dict(), "task3_best_tuned")
    
    # plotting the loss graph
    plot_train_valid_loss(t_loss_history, v_loss_history, epochs)
    
    # plotting the acc graph
    plot_train_valid_acc(t_acc_history, v_acc_history, epochs)
    
    # evaluate the model based on testing set
    model.load_state_dict(torch.load("task3_best_tuned"))
    ts_loss, ts_acc = validation_testing_evaluation(model, test_iter, crit, tag_pad_idx)
    print("testing loss:", ts_loss, "testing accuracy", ts_acc )
    
    # task 3.3
    for s in sentences:
        word_tag = tag_sentence(s, model, TEXT, UD_TAGS)
        print("token:", s)
        print("pred:", word_tag)
    
    # task 3.1
    plot_percentage(tag_percentage(UD_TAGS.vocab.freqs.most_common()))
