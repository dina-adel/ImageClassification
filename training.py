import sys
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import callbacks
import time
import os

def build_model (img_width,img_height, classes_num):
    
    #model parameters
    nb_filters1 = 64
    nb_filters2 = 32
    conv1_size = 5
    conv2_size = 3
    pool_size = 3
    lr = 0.002
    
    # constructing the model
    model = Sequential()
    
    model.add(Conv2D(nb_filters1, conv1_size, conv1_size, padding="same",
                     input_shape=(img_width, img_height, 3)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(pool_size, pool_size)))

    model.add(Conv2D(nb_filters2, conv2_size, conv2_size, padding ="same"))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(pool_size, pool_size)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))
    return model

def fit_model(img_width,img_height, classes_num, train_generator, validation_generator):

    #Model Parameters
    batch_size = 32
    steps_per_epoch = 6364//batch_size
    validation_steps = 1590//batch_size
    epochs = 50

    #Tensorboard log
    log_dir = './tf-log/'
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    cbks = [tb_cb]
    
    #build model
    print("Build model")
    model = build_model (img_width,img_height, classes_num)
    #compile
    print("Compile model")
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    #fit
    print("Now fitting the model")
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=cbks,
        validation_steps=validation_steps)
    return model

def generate_data(train_data_dir,img_width,img_height, batch_size):
  train_datagen = ImageDataGenerator(
      rescale=1./255,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      validation_split=0.2) # set validation split

  train_generator = train_datagen.flow_from_directory(
      train_data_dir,
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='categorical',
      subset='training') # set as training data

  validation_generator = train_datagen.flow_from_directory(
      train_data_dir, 
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='categorical',
      subset='validation') # set as validation data
  
  return train_generator, validation_generator

def train():
    print("start training")
    #define parameters
    img_width,img_height = 300,300
    classes_num = 3
    batch_size = 32
    train_data_dir = 'Data/Train/'
    #generate data
    print("Generate data")
    train_generator, validation_generator = generate_data(train_data_dir,img_width,img_height, batch_size)
    #compile&fit model
    model = fit_model(img_width,img_height, classes_num, train_generator, validation_generator)

    #save model
    target_dir = './models/'
    if not os.path.exists(target_dir):
      os.mkdir(target_dir)
    model.save('./models/model.h5')
    model.save_weights('./models/weights.h5')

    print("Finished Training")
