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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, torch.tensor(label)
    

train_csv = pd.read_csv('/Users/sfarahmand/Desktop/HackGT/fatigue/train.csv')
test_csv = pd.read_csv('/Users/sfarahmand/Desktop/HackGT/fatigue/test.csv')

train_csv['path'] = train_csv['path'].str.replace('\\','/')
test_csv['path'] = test_csv['path'].str.replace('\\','/')

train_csv['path'] = '/Users/sfarahmand/Desktop/HackGT/fatigue/' + train_csv['path']
test_csv['path'] = '/Users/sfarahmand/Desktop/HackGT/fatigue/' + test_csv['path']

train_data, val_data = train_test_split(train_csv, test_size = 0.2, random_state = 101)


model_path = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path,num_labels=2,ignore_mismatched_sizes=True).cuda()
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path) 
image_mean, image_std = processor.image_mean, processor.image_std

transform = Compose([

    Resize((224,224)),
    ToTensor(),
    Normalize(mean=image_mean,std=image_std)

])

train_dataset = EmotionDataset(train_data, transform=transform)
test_dataset = EmotionDataset(test_csv, transform=transform)
val_dataset = EmotionDataset(val_data,transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)



##### Training Loop
optimizer = Adam(model.parameters(),1e-5)
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 5
epoch_number = 0
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))


def epoch_train(epoch_number, writer):
    running_loss = 0
    last_loss = 0
    for i,data in enumerate(train_loader):
        input, labels = data
        input = input.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        output = model(input).logits
        loss = loss_fn(output, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss/1000
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0

    return last_loss

best_val_loss = 10000000
for epoch in range(epochs):
    print('Epoch {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = epoch_train(epoch_number, writer)
    running_val_loss = 0.0
    model.eval()

    with torch.no_grad():
        for i, val in enumerate(validation_loader):
            val_inputs, val_labels = data
            val_outputs = model(val_inputs)
            val_loss = loss_fin(val_outputs, val_labels)
            running_val_loss += val_loss
    avg_val_loss = running_val_loss / (i+1)
    print('Loss train {} validation {}'.format(avg_loss, avg_val_loss))

    writer.add_scalars('Training vs Validation Loss', {
        'Training':avg_loss,
        'Validation':a
    },
    epoch_number + 1
    )
    writer.flush()

    if avg_val_loss < best_val_loss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(),model_path)
    epoch_number += 1
