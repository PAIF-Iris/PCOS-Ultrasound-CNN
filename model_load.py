from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
loaded_model = load_model("resnet50_pcos_trainable_true.keras")
loaded_model.summary()

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
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False, seed=42
)

loss, accuracy= loaded_model.evaluate(test_gen)    
