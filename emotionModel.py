import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTImageProcessor, ViTForImageClassification, ViTFeatureExtractor
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import tqdm

class EmotionDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, index):
        path = self.data.iloc[index]['path']
        label = self.data.iloc[index]['label']

        img = cv2.imread(path)

        if img is None:
            print(path)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, torch.tensor(label)
    
dataset_root = "C:/Users/chace/code/_datasets/fef"

train_csv = pd.read_csv(f'{dataset_root}/train.csv')
test_csv = pd.read_csv(f'{dataset_root}/test.csv')

train_csv['path'] = train_csv['path'].str.replace('\\','/')
test_csv['path'] = test_csv['path'].str.replace('\\','/')

train_csv['path'] = f'{dataset_root}/' + train_csv['path']
test_csv['path'] = f'{dataset_root}/' + test_csv['path']

train_data, val_data = train_test_split(train_csv, test_size = 0.2, random_state = 101)

model_path = 'google/vit-base-patch16-224' #'WinKawaks/vit-tiny-patch16-224'
processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path,num_labels=2,ignore_mismatched_sizes=True).cuda()
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path) 
image_mean, image_std = processor.image_mean, processor.image_std

# turn off most of model layers
model.vit.embeddings.requires_grad_(False)
model.vit.encoder.layer.requires_grad_(False)

model.vit.encoder.layer[-2].requires_grad_(True)
model.vit.encoder.layer[-1].requires_grad_(True)

transform = Compose([
    Resize((224,224)),
    ToTensor(),
    Normalize(mean=image_mean,std=image_std)
])

train_dataset = EmotionDataset(train_data, transform=transform)
test_dataset = EmotionDataset(test_csv, transform=transform)
val_dataset = EmotionDataset(val_data,transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

##### Training Loop
optimizer = Adam(model.parameters(),1e-5)
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 5
epoch_number = 0
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

def epoch_train():
    running_loss = 0
    
    for i,data in (pbar:=tqdm.tqdm(enumerate(train_loader), total=len(train_loader.dataset)//train_loader.batch_size)):
        input, labels = data
        input = input.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        output = model(input).logits
        loss = loss_fn(output, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({ "loss": running_loss / (i + 1) })

    return running_loss / (i + 1)

def run_validation():
    running_val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, val in tqdm.tqdm(enumerate(val_loader), total=len(val_loader.dataset)//val_loader.batch_size):
            val_inputs, val_labels = val
            val_inputs = val_inputs.cuda()
            val_labels = val_labels.cuda()
            val_outputs = model(val_inputs).logits
            val_loss = loss_fn(val_outputs, val_labels)
            running_val_loss += val_loss.item()
    avg_val_loss = running_val_loss / (i+1)
    return avg_val_loss

def run_test():
    with torch.no_grad():
        total_test_loss = 0
        for i, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader.dataset)//test_loader.batch_size):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs).logits
            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / (i+1)
    return avg_test_loss

for epoch in range(epochs):
    print('Epoch {}:'.format(epoch + 1))

    model.train(True)
    avg_loss = epoch_train()
    running_val_loss = 0.0
    model.eval()
    
    avg_val_loss = run_validation()
    avg_test_loss = run_test()

    print('Loss train {} validation {} test {}'.format(avg_loss, avg_val_loss, avg_test_loss))

torch.save(model, "./ckpt_google_vit_base_2.bin")