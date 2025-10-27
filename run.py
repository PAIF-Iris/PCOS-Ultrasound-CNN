from functions import remove_duplicate
import splitfolders
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

##remove duplicates and moved all images to no_duplicates folder
#remove_duplicate.remove_even_images("/Users/paif_iris/Desktop/PCOS_CNN/Dataset/Infected")
#remove_duplicate.remove_even_images("/Users/paif_iris/Desktop/PCOS_CNN/Dataset/NonInfected")

##Split dataset and move into train_test_split folder
# input_folder = "/Users/paif_iris/Desktop/PCOS_CNN/Dataset/no_duplicates"
# output_folder = "/Users/paif_iris/Desktop/PCOS_CNN/Dataset/train_test_split"

# splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.8, 0.1, 0.1))


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


#define model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))   #transfer learning: using the pretrained imagenet
#does not change the weights of ResNet50
for layer in base_model.layers:
    layer.trainable = False

#adding custom layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=preds)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#silently monitors training progress, prevents overfitting by stopping early, and saves weights of the next best model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_resnet50_pcos.h5', monitor='val_accuracy', save_best_only=True)
]

#starts training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)

loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy: {acc:.4f}")

model.save("resnet50_pcos_new_duplicate_remove_strategy.h5")