import psutil
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import gc
import os
import warnings
warnings.filterwarnings("ignore")


# original SAKT doesn't include exercise category features
# therefore here when we observe how model performance 
# changes over different sequences, we don't use chapter and sub_chapter


#create dataset class
#to prepare it for train and valid sets
#here only original features are included
#that were present in SAKT: questions and answers
class PRACTICE_DATASET(Dataset):
    def __init__(self, group, n_skill=data['question_id'].nunique() + 1, min_samples=1, max_seq=200):
        super(PRACTICE_DATASET, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill        
        self.user_ids = group.keys()
        self.data = group
    
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_ = self.data[user_id]
        q_, qa_ = np.array(q_), np.array(qa_)
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        #if more the user interactions are more than self.maxlength 
        #only the last self.maxlength interactions will be included
        #if the number of user interactions would be less than self.maxlength
        #the padding would be utilized
        if seq_len >= self.max_seq:
            q[-self.max_seq:] = q_[-self.max_seq:]
            qa[-self.max_seq:] = qa_[-self.max_seq:]
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_

        target_id = q[1:]
        label = qa[1:]

        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_skill

        return x, target_id, label
    

#USE ONLY FEATURES PRESENT IN SAKT
group = data.groupby(['user_id.x']).apply(lambda r: (
                r['question_id'].values,
                r['answered_correctly'].values
                ))


#feed forward network
class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)
    
#define mask that would be used in multi head attention layer
def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)

#define the SAKT model
class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=200, embed_dim=128, dropout_rate=0.2):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim

        #embeddings
        self.embedding = nn.Embedding(2*n_skill+1, embed_dim) 
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_normal = nn.LayerNorm(embed_dim) 

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
    
    def forward(self, x, question_ids):
        device = x.device        
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)

        pos_x = self.pos_embedding(pos_id)
        x = x + pos_x

        e = self.e_embedding(question_ids)

        x = x.permute(1, 0, 2) 
        e = e.permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        att_output = self.layer_normal(att_output + e)
        att_output = att_output.permute(1, 0, 2) 

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1)
    
#user KFold
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
    train_dataloader = DataLoader(train, batch_size=64, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(valid, batch_size=64, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test, batch_size=64, shuffle=True, num_workers=8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saint = SAKTModel(n_skill)
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
            x = item[0].to(device).long()
            target_id = item[1].to(device).long()
            label = item[2].to(device).float()            
            target_mask = (target_id!=0)
            optim.zero_grad()
            output = model(x, target_id)
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
            x = item[0].to(device).long()
            target_id = item[1].to(device).long()
            label = item[2].to(device).float()                
            target_mask = (target_id!=0)
            with torch.no_grad():
                output = model(x, target_id)

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
    train_loss, train_acc, train_auc = val_epoch(model=saint, val_iterator=train_dataloader, device=device)

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

    
    

    
    
    