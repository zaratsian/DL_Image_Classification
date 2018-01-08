
##############################################################################################################
#
#   Keras - Image Classification (multi-class)
#   https://keras.io/
#
##############################################################################################################

'''
conda install keras
#pip install numpy
#pip install pandas
#pip install nose
#pip install pillow
#pip install h5py
#pip install tensorflow
pip install py4j
#pip install opencv-python

cd /tmp
git clone https://github.com/fchollet/deep-learning-models
cd deep-learning-models
ls -al
'''

import os
import math
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.models import load_model
#import matplotlib.pyplot as plt
#import cv2 #pip install opencv-python



img_width, img_height   = 224, 224
top_model_path          = os.getcwd() + '/elephant_model.h5'
top_model_weights_path  = os.getcwd() + '/elephant_weights.h5'
class_indices_path      = os.getcwd() + '/elephant_class_indices.npy'
bottleneck_training     = os.getcwd() + 'bottleneck_features_training.npy'
bottleneck_validation   = os.getcwd() + 'bottleneck_features_validation.npy'
train_data_dir          = os.getcwd() + '/training'
validation_data_dir     = os.getcwd() + '/validation'
epochs                  = 40



def calculate_batch_size(total_files, min_files):
    batch_size = min_files if min_files <= 10 else 10
    while (total_files % batch_size != 0):
        batch_size -= 1
    return batch_size



total_files_training    = sum([len(files) for r, d, files in os.walk( os.getcwd() + '/training')])
min_files_training      = min([len(files) for r, d, files in os.walk( os.getcwd() + '/training')][1:])
batch_size_training     = calculate_batch_size(total_files_training, min_files_training)

total_files_validation  = sum([len(files) for r, d, files in os.walk( os.getcwd() + '/validation')])
min_files_validation    = min([len(files) for r, d, files in os.walk( os.getcwd() + '/validation')][1:])
batch_size_validation   = calculate_batch_size(total_files_validation, min_files_validation)



def save_bottleneck_features():
    
    print('[ INFO ] Loading Imagenet Model..')
    model = applications.VGG16(include_top=False, weights='imagenet')
    
    datagen = ImageDataGenerator(rescale=1. / 255)
    
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size_training,
        class_mode=None,
        shuffle=False)
    
    #print(len(generator.filenames))
    #print(generator.class_indices)
    #print(len(generator.class_indices))
    
    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)
    
    predict_size_train = int(math.ceil(nb_train_samples / batch_size_training))
    print('[ INFO ] Predicted Train Size: ' + str(predict_size_train))
    
    print('[ INFO ] Generate training bottleneck features')
    bottleneck_features_train = model.predict_generator(generator, predict_size_train)
    
    print('[ INFO ] Saving Features for Training (' + str(bottleneck_training) + ')')
    np.save(bottleneck_training, bottleneck_features_train)
    
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size_validation,
        class_mode=None,
        shuffle=False)
    
    nb_validation_samples = len(generator.filenames)
    
    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size_validation))
    
    print('[ INFO ] Generate validation bottleneck features')
    bottleneck_features_validation = model.predict_generator(generator, predict_size_validation)
    
    print('[ INFO ] Saving Features for Validation (' + str(bottleneck_validation) + ')')
    np.save(bottleneck_validation, bottleneck_features_validation)



def train_top_model():
    
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size_training,
        class_mode='categorical',
        shuffle=False)
    
    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)
    
    print('[ INFO ] Saving class_indices (' + str(class_indices_path) + ')')
    np.save(class_indices_path, generator_top.class_indices)
    
    print('[ INFO ] Load training bottleneck features ( ' + str(bottleneck_training) + ')')
    train_data = np.load(bottleneck_training)
    
    train_labels = generator_top.classes
    # Convert the training labels to categorical vectors
    # https://github.com/fchollet/keras/issues/3467
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    
    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size_validation,
        class_mode=None,
        shuffle=False)
    
    nb_validation_samples = len(generator_top.filenames)
    
    print('[ INFO ] Load validation bottleneck features ( ' + str(bottleneck_validation) + ')')
    validation_data = np.load(bottleneck_validation)
    
    validation_labels = generator_top.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)
    
    print('[ INFO ] Building Neural Network Model (hidden layers)')
    model=Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print('[ INFO ] Fitting CNN...')
    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size_training,
                        validation_data=(validation_data, validation_labels))
    
    print('[ INFO ] Saving CNN model to ' + str(top_model_path))
    model.save(top_model_path)
    print('[ INFO ] Saving CNN model weights to ' + str(top_model_weights_path))
    model.save_weights(top_model_weights_path)
    
    (eval_loss, eval_accuracy) = model.evaluate(validation_data, validation_labels, batch_size=batch_size_validation, verbose=1)
    
    print('[INFO] Accuracy: ' + str(eval_accuracy * 100) + '%')
    print('[INFO] Loss: ' + str(eval_loss))



def predict(image_path='/tmp/modeling/validation/alligator/alligator5501.jpeg'):
    # load the class_indices saved in the earlier step
    class_dictionary = np.load(class_indices_path).item()
    
    num_classes = len(class_dictionary)
    
    # add the path to your test image below
    image_path = image_path
    
    orig = cv2.imread(image_path)
    
    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    
    # important! otherwise the predictions will be '0'
    image = image / 255
    
    image = np.expand_dims(image, axis=0)
    
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    
    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)
    
    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    model.load_weights(top_model_weights_path)
    
    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)
    
    probabilities = model.predict_proba(bottleneck_prediction)
    
    inID = class_predicted[0]
    
    inv_map = {v: k for k, v in class_dictionary.items()}
    
    label = inv_map[inID]
    
    # get the prediction label
    print("Image ID: {}, Label: {}".format(inID, label))
    
    # Display the predictions with the image
    #cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
    #            cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)
    #
    #cv2.imshow("Classification", orig)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



save_bottleneck_features()
train_top_model()

#predict()
#cv2.destroyAllWindows()



#   https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#   http://www.codesofinterest.com/2017/08/bottleneck-features-multi-class-classification-keras.html
#   https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
#   https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#   https://www.pyimagesearch.com/2017/12/18/keras-deep-learning-raspberry-pi/

#ZEND

