import torch
import torchvision
from torchvision import transforms
import os

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

base_dir = "/Users/paif_iris/Desktop/PCOS_CNN/Dataset/train_test_split"
train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "val")
test_dir  = os.path.join(base_dir, "test")

img_size = (224, 224)
batch_size = 32

#data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,          
    rotation_range=15,       
    width_shift_range=0.1,   
    height_shift_range=0.1, 
    shear_range=0.1,         
    zoom_range=0.1,         
    horizontal_flip=True,  
    fill_mode="nearest"    
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


#creating batches of data from directories
train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=True, seed=42
)

val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False, seed=42
)

test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False, seed=42
)
