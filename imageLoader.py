
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

from transformers import ViTImageProcessor
from torchvision.transforms import Compose, ToTensor

def load_image(image_path):
    
    #Load Image
    img = cv2.imread('/Users/sfarahmand/Desktop/HackGT/fatigue/'+image_path)
    #Convert to grayscale, equalize histogram, normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    normalized = img / 255.0
    
    #Plot Images
    fig = plt.imshow(normalized)  
    plt.axis('off')
    size = fig.get_size()
    plt.show()
    return img
    


def process_image(path_list):

    #Replace backslashes with forward slashes in file path for pathname to be readable by opencv
    path_list = [filepath.replace('\\', '/') for filepath in path_list]
    for image_path in path_list:
        a = load_image(image_path)
        

  
#List of filepaths and corresponding label
train_file = pd.read_csv('/Users/sfarahmand/Desktop/HackGT/fatigue/train.csv')
test_file =  pd.read_csv('/Users/sfarahmand/Desktop/HackGT/fatigue/test.csv')
image_path = '/Users/sfarahmand/Desktop/HackGT/fatigue/images/train/fatigue/733.jpg'


#Visual Transformer from HuggingFace
model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

#Process image function call
process_image(train_file['path'])

#print(train_file.head())
#print(train_file.shape)
#print(test_file.head())
#print(test_file.shape)



