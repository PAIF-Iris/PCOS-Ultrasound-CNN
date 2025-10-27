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

# remove_duplicate.find_similar_images(["/Users/paif_iris/Desktop/PCOS_CNN/Dataset/Infected"], 
#                     "/Users/paif_iris/Desktop/PCOS_CNN/Dataset/Infected_no_duplicates")

# remove_duplicate.find_similar_images(["/Users/paif_iris/Desktop/PCOS_CNN/Dataset/NonInfected"], 
#                     "/Users/paif_iris/Desktop/PCOS_CNN/Dataset/NonInfected_no_duplicates")


# input_folder = "/Users/paif_iris/Desktop/PCOS_CNN/Dataset/no_duplicates"
# output_folder = "/Users/paif_iris/Desktop/PCOS_CNN/Dataset/train_test_split"

# splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.8, 0.1, 0.1))

base_dir = "/Users/paif_iris/Desktop/PCOS_CNN/Dataset/train_test_split"
train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "val")
test_dir  = os.path.join(base_dir, "test")

img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values to [0,1]
    rotation_range=15,        # Randomly rotate images ±15 degrees
    width_shift_range=0.1,    # Shift horizontally up to 10%
    height_shift_range=0.1,   # Shift vertically up to 10%
    shear_range=0.1,          # Apply slight shearing
    zoom_range=0.1,           # Zoom in/out randomly by ±10%
    horizontal_flip=True,     # Flip images horizontally (like mirror)
    fill_mode="nearest"       # Fill in missing pixels after transforms
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False
)



base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=preds)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_resnet50_pcos.h5', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)

loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {acc:.4f}")

model.save("resnet50_pcos_final.h5")