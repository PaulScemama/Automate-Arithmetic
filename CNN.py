import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import model_from_json


# Creates batches of tensor image data with real-time data augmentation
# holds 20% of dataset for validation
train_datagen = ImageDataGenerator(rescale = 1./255,    
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    validation_split = .2)

# Define path to directory where my data is
path = '/Users/paulscemama/AutomatingArith/train'

# get batches of augmented data for training
train = train_datagen.flow_from_directory(path,
    target_size = (64,64),
    color_mode = "grayscale",
    classes = ['-','+','=','0','1','2','3','4','5','6','7','8','9','*'],
    class_mode = 'categorical',
    subset = 'training',
    seed = 121)

val = train_datagen.flow_from_directory(path,
    target_size = (64,64),
    color_mode = 'grayscale',
    classes = ['-','+','=','0','1','2','3','4','5','6','7','8','9','*'],
    class_mode = 'categorical',
    subset = 'validation',
    seed = 121
)


def prepare_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (64,64,1)))
    model.add(MaxPooling2D(pool_size = (2,2)))    
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(14, activation = 'softmax'))

    # compile 
    optimizer = SGD(lr = 0.01, momentum = 0.9)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model



classify = prepare_model()

classify.fit(train,
                    validation_data = val,
                    steps_per_epoch = train.n//train.batch_size,
                    validation_steps = val.n//val.batch_size,
                    epochs=5)

# Evaluate model
score = classify.evaluate(val)
print('Val Accuracy', score[1])

# save weights and model

# Serialize model to json
classify_json = classify.to_json()
with open('classify_json','w') as json_file:
    json_file.write(classify_json)

# Serialize weights to HDF5
classify.save_weights('classify_weights.h5')


# Ensure it worked by loading in the model and testing its performance
# load model
json_file = open('classify_json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_classify = model_from_json(loaded_model_json)
# load weights
loaded_classify.load_weights('classify_weights.h5')

# evaluate model
loaded_classify.compile(loss = 'binary_crossentropy', optimizer = SGD(lr = 0.01, momentum = 0.9), metrics = ['accuracy'])
loaded_score = loaded_classify.evaluate(val, verbose = 0)
print('Accuracy', loaded_score[1])