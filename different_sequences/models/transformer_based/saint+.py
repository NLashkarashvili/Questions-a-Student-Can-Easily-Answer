import psutil
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gc
import os
import warnings
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

#####################################################
# apart from data preprocessing provided in data_preparation file
# here we also provide code for prev_time_elapsed and time_lag
# variables that are included in the original saint+ model 
# NOTE: previous question response also goes in the decoder part.

data['prev_time_elapsed'] = None
data['time_lag'] = None
data['time_lag'] = data['time_lag'].astype(np.float)
data['prev_time_elapsed'] = data['prev_time_elapsed'].astype(np.float)unique_chapts = data['chapter_id'].unique()
cnt = 0
for user in tqdm(data['user_id.x'].unique()):
        for chapter in unique_chapts:
            tmp_user = data[(data['user_id.x']==user) & (data['chapter_id']==chapter)]
            if len(tmp_user) < 1:
                continue
            tmp_time_elapsed = tmp_user.end_practice - tmp_user.start_practice
            tmp_time_elapsed = tmp_time_elapsed / np.timedelta64(1, 's')
            #shifting time elapsed by one
            #so that time_elapsed row for each question
            #would refer to the time that user took to answer
            #previous question
            tmp_time_elapsed = np.insert(np.array(tmp_time_elapsed[:-1]), 0, 0., axis=0)
            tmp_time_elapsed = np.cumsum(tmp_time_elapsed)
            indices = tmp_user.index
            start_row = indices[0]
            data['time_lag'].iloc[start_row] = 0
            for_mean = np.arange(len(tmp_user))
            for_mean[0] = 1
            time_substrahend = tmp_user.start_practice.iloc[:-1]
            time_substrahend = time_substrahend.apply(lambda a: a.timestamp())
            time_substrahend = np.array(time_substrahend)
            
            time_minuend = tmp_user.start_practice.iloc[1:]
            time_minuend = time_minuend.apply(lambda a: a.timestamp())
            time_minuend = np.array(time_minuend)

            data['prev_time_elapsed'].iloc[indices] = tmp_time_elapsed/for_mean
            data['time_lag'].iloc[indices[1:]] = time_minuend - time_substrahend


data.drop(columns=['end_practice'], inplace=True)
data = data.sort_values(['start_practice'], ascending=True).reset_index(drop=True)
data['answered_correctly'] = data['q']
data.drop(columns='q', inplace=True)

#IN OUR EXPERIMENT WE SET MAX_SEQ TO 
#FOLLOWING VALUES 2, 5-15, 100, 200, 300, 400
MAX_SEQ = 100
n_part = data['sub_chapter_id'].nunique() + 1
D_MODEL = 128
N_LAYER = 2
DROPOUT = 0.2

class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAINTModel(nn.Module):
    def __init__(self, n_skill, n_part, max_seq=MAX_SEQ, embed_dim= D_MODEL, elapsed_time_cat_flag = False):
        super(SAINTModel, self).__init__()

        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.n_chapter= 39
        self.n_sub_chapter = n_part
        self.elapsed_time_cat_flag = elapsed_time_cat_flag

        self.q_embedding = nn.Embedding(self.n_skill+1, embed_dim) ## exercise
        self.c_embedding = nn.Embedding(self.n_chapter+1, embed_dim) ## category
        self.sc_embedding = nn.Embedding(self.n_sub_chapter, embed_dim) ## category
        self.pos_embedding = nn.Embedding(max_seq+1, embed_dim) ## position
        self.res_embedding = nn.Embedding(2+1, embed_dim) ## response
        self.feat_embedding = nn.Linear(2, embed_dim)
    



        self.transformer = nn.Transformer(nhead=8, d_model = embed_dim, num_encoder_layers= N_LAYER, num_decoder_layers= N_LAYER, dropout = DROPOUT)

        self.dropout = nn.Dropout(DROPOUT)
        self.layer_normal = nn.LayerNorm(embed_dim) 
        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
    
    def forward(self, question, chapter, schapter, response, user_features):

        device = question.device  
        ## embedding layer
        question = self.q_embedding(question)
        chapter = self.c_embedding(chapter)
        schapter = self.sc_embedding(schapter)
        pos_id = torch.arange(question.size(1)).unsqueeze(0).to(device)
        pos_id = self.pos_embedding(pos_id)
        res = self.res_embedding(response)
        user_features = self.feat_embedding(user_features)
        

        enc = pos_id + question + chapter + schapter 
        dec = pos_id + res + enc + user_features
        enc = enc.permute(1, 0, 2) 
        dec = dec.permute(1, 0, 2)
        mask = future_mask(enc.size(0)).to(device)
        att_output = self.transformer(enc, dec, src_mask=mask, tgt_mask=mask, memory_mask = mask)
        att_output = self.layer_normal(att_output)
        att_output = att_output.permute(1, 0, 2) 
        
        
        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1)


class PRACTICE_DATASET(Dataset):
    def __init__(self, data, maxlength=MAX_SEQ, test=False):
        super(PRACTICE_DATASET, self).__init__()
        self.maxlength = maxlength
        self.data = data
        self.test = test
        self.users = list()
        for user in data.index:
            self.users.append(user)
            
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, ix):
        user = self.users[ix]
        question_id, chapter, schapter, term, temp_feats, labels = self.data[user]
        question_id = np.array(question_id, np.int16)
        chapter = np.array(chapter, np.int16)
        schapter = np.array(schapter, np.int16)
        tem_feats = np.array(temp_feats, np.float)
        labels = np.array(labels, np.int8)   
            

        
        if n > self.maxlength:
            question_id = question_id[-self.maxlength : ]
            chapter = chapter[-self.maxlength :]
            schapter = schapter[-self.maxlength: ]
            temp_feats = temp_feats[-self.maxlength :, :]
            labels = labels[-self.maxlength: ]
            responses = np.append(2, labels[:-1])
        else:
            question_id = np.pad(question_id, (self.maxlength - n, 0))
            chapter = np.pad(chapter, (self.maxlength - n, 0))
            schapter = np.pad(schapter, (self.maxlength - n, 0))
            temp_feats = [[0]*len(temp_feats[0])]*(self.maxlength  - n)+list(temp_feats[:])
            temp_feats = np.array(temp_feats, np.float)
            responses = np.append(2, labels[:-1])
            labels = np.pad(labels, (self.maxlength - n, 0))
            responses = np.pad(responses, (self.maxlength - n, 0), mode='constant', constant_values = 2)
        

        
        return question_id, chapter, schapter, responses, np.array(temp_feats), labels 





#group data based on user id
group = data.groupby(['user_id.x']).apply(lambda r: (
                r['question_id'],
                r['chapter_id'],
                r['sub_chapter_id'],
                r['term'],
                np.array([ r['prev_time_elapsed'],
                           r['time_lag']
              ]).transpose(),
                r['answered_correctly'],
                ))




X = np.array(group.keys())
kfold = KFold(n_splits=5, shuffle=True)
train_losses = list()
train_aucs = list()
train_accs = list()
val_losses = list()
val_aucs = list()
val_accs = list()
test_losses = list()
test_aucs = list()
test_accs = list()
for train, test in kfold.split(X):
    users_train, users_test =  X[train], X[test]
    n = len(users_test)//2
    users_test, users_val = users_test[:n], users_test[n: ]
    train = PRACTICE_DATASET(group[users_train])
    valid = PRACTICE_DATASET(group[users_val])
    test = PRACTICE_DATASET(group[users_test])
    train_dataloader = DataLoader(train, batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(valid, batch_size=32, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test, batch_size=32, shuffle=True, num_workers=8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saint = SAINTModel(n_skill, n_part)
    epochs = 100
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(saint.parameters(), betas=(0.9, 0.999), lr = 0.0005, eps=1e-8)
    saint.to(device)
    criterion.to(device)
    
    def train_epoch(model=saint, train_iterator=train_dataloader, optim=optimizer, criterion=criterion, device=device):
        model.train()

        train_loss = []
        num_corrects = 0
        num_total = 0
        labels = []
        outs = []
        tbar = tqdm(train_iterator)
        for item in tbar:
            question_id = item[0].to(device).long()
            chapter = item[1].to(device).long()
            schapter = item[2].to(device).long()
            responses = item[3].to(device).long()
            temp_feats = item[4].to(device).float()
            label = item[5].to(device).float()            
            target_mask = (question_id!=0)
            optim.zero_grad()
            output = model(question_id, chapter, schapter, responses, temp_feats)
            output = torch.reshape(output, label.shape)

            output = torch.masked_select(output, target_mask)
            label = torch.masked_select(label, target_mask)

            loss = criterion(output, label)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            pred = (torch.sigmoid(output) >= 0.5).long()

            num_corrects += (pred == label).sum().item()
            num_total += len(label)

            labels.extend(label.view(-1).data.cpu().numpy())
            outs.extend(output.view(-1).data.cpu().numpy())

            tbar.set_description('loss - {:.4f}'.format(loss))
        acc = num_corrects / num_total
        auc = roc_auc_score(labels, outs)
        loss = np.mean(train_loss)

        return loss, acc, auc
   

    def val_epoch(model=saint, val_iterator=test_dataloader, 
              criterion=criterion, device=device):
        model.eval()

        train_loss = []
        num_corrects = 0
        num_total = 0
        labels = []
        outs = []
        tbar = tqdm(val_iterator)
        for item in tbar:
            question_id = item[0].to(device).long()
            chapter = item[1].to(device).long()
            schapter = item[2].to(device).long()
            responses = item[3].to(device).long()
            temp_feats = item[4].to(device).float()
            label = item[5].to(device).float()            
            target_mask = (question_id!=0)
            with torch.no_grad():
                output = model(question_id, chapter, schapter, responses, temp_feats)

            output = torch.reshape(output, label.shape)
            output = torch.masked_select(output, target_mask)
            label = torch.masked_select(label, target_mask)

            loss = criterion(output, label)
            train_loss.append(loss.item())

            pred = (torch.sigmoid(output) >= 0.5).long()
            num_corrects += (pred == label).sum().item()
            num_total += len(label)

            labels.extend(label.view(-1).data.cpu().numpy())
            outs.extend(output.view(-1).data.cpu().numpy())

            tbar.set_description('valid loss - {:.4f}'.format(loss))

        acc = num_corrects / num_total
        auc = roc_auc_score(labels, outs)
        loss = np.average(train_loss)

        return loss, acc, auc
    
    MIN_VAL = 1000000000
    count = 0
    print('----------------------------------------------------------------------------')
    for epoch in range(epochs):
        train_loss, train_acc, train_auc = train_epoch(model=saint, device=device)
        print("epoch - {} train_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, train_loss, train_acc, train_auc))
        val_loss, val_acc, val_auc = val_epoch(model=saint, val_iterator= val_dataloader, device=device)
        print("epoch - {} val_loss - {:.2f} val acc - {:.3f} val auc - {:.3f}".format(epoch, val_loss, val_acc, val_auc))
        if val_loss < MIN_VAL:
            count = 0
            MIN_VAL = val_loss
        else:
            count += 1

        if count == patience:
            print('Val Loss does not improve for {} consecutive epochs'.format(patience))
            break
    test_loss, test_acc, test_auc = val_epoch(model=saint, device=device)
    print("epoch - {} test_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, test_loss, test_acc, test_auc))
    test_losses.append(test_loss)
    test_aucs.append(test_auc)
    test_accs.append(test_acc)
    train_aucs.append(train_auc)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
#display test loss/acc/auc
print("test avg loss: ", np.mean(test_losses), np.std(test_losses) )
print("test avg acc: ", np.mean(test_accs), np.std(test_accs))
print("test avg auc: ", np.mean(test_aucs), np.std(test_aucs))

#display train loss/acc/auc
print("train avg loss: ", np.mean(train_losses), np.std(train_losses) )
print("train avg acc: ", np.mean(train_accs), np.std(train_accs))
print("train avg auc: ", np.mean(train_aucs), np.std(train_aucs))