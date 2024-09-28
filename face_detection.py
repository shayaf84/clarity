import cv2
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTImageProcessor, ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model_path = 'xacer/vit-base-patch16-224-fatigue'
cap = cv2.VideoCapture(0)

def crop_bounding_box(image, x, y, w, h):
    return image[y:y+h,x:x+w]


processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path,num_labels=2,ignore_mismatched_sizes=True)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path) 
image_mean, image_std = processor.image_mean, processor.image_std

transform = Compose([

    Resize((224,224)),
    ToTensor(),
    Normalize(mean=image_mean,std=image_std)

])


transform = Compose([

    Resize((224,224)),
    ToTensor(),
    Normalize(mean=image_mean,std=image_std)

])

class_value = None


while True:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cropped_image = crop_bounding_box(frame,x,y,w,h)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
        cropped_pil = Image.fromarray(cropped_image)
        transformed_pil = transform(cropped_pil)

        input_tensor = transformed_pil.unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
        predicted_class = torch.argmax(outputs.logits).item()
        if predicted_class == 1:
            class_value = "Active"
        elif predicted_class == 0:
            class_value = "Fatigued"
        

        print("Predicted Class:",predicted_class)
        cv2.putText(frame,f'{class_value}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('cropped_image',frame)
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

    
