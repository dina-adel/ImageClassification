 
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
import time
from keras import backend as K


#Prediction Function
def test (model, image):

    img_width, img_height = 300,300
    img = load_img(image, target_size=(img_width,img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    pred = model1.predict(img)
    class_ = pred[0]
    class_ = np.argmax(class_)
    return class_


if __name__ == "__test__":
    test()