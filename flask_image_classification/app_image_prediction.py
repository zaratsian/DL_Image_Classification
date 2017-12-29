

import os,sys,re, csv
import random
import json
from flask import Flask, render_template, json, request, redirect, jsonify, url_for, session
from werkzeug.utils import secure_filename
import flask_login
import requests
import datetime, time

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import math
import cv2 # pip install opencv-python
import base64
from PIL import Image
from io import BytesIO
import cStringIO

################################################################################################
#
#   Flask App
#
################################################################################################

app = Flask(__name__)
app.secret_key = os.urandom(24)

################################################################################################
#
#   Global Variables
#
################################################################################################

image_directory     = os.getcwd()+'/static/model_images'
#model_vgg16        = applications.VGG16(include_top=False, weights='imagenet')
model_vgg16         = load_model(os.getcwd()+'/static/assets/vgg16_model.h5')
model_weights_path  = os.getcwd()+'/static/assets/alligator_weights.h5'
class_dictionary    = np.load(os.getcwd()+'/static/assets/class_indices.npy').item()

################################################################################################
#
#   Functions
#
################################################################################################


def get_all_images(image_directory):
    return [image_directory+'/'+file for file in os.listdir(os.getcwd()+image_directory.replace('.',''))]


def get_random_image():
    all_images = [image_directory+'/'+file for file in os.listdir(image_directory)]
    return all_images[random.randint(0,len(all_images)-1)]


def predict_from_image_path(image_path, model_vgg16, model_weights_path, class_dictionary):
    
    num_classes = len(class_dictionary)
    
    # Read Image Path
    image_path = image_path
    orig = cv2.imread(image_path)
    
    #print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    
    # important! otherwise the predictions will be '0'
    image = image / 255
    
    image = np.expand_dims(image, axis=0)
    
    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model_vgg16.predict(image)
    
    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    model.load_weights(model_weights_path)
    
    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)
    
    probabilities = model.predict_proba(bottleneck_prediction)
    
    inID = class_predicted[0]
    
    inv_map = {v: k for k, v in class_dictionary.items()}
    
    prediction_prob = round(probabilities[0][0], 8)
    #prediction     = inv_map[inID]
    prediction      = 'Alligator' if prediction_prob >= .8 else 'No Alligator'
    
    return prediction, float(prediction_prob)


def predict_from_base64_image(image_base64_encoded, model_vgg16, model_weights_path, class_dictionary):
    
    num_classes = len(class_dictionary)
    
    # Read Image (as Base64 encoded)
    #image = open('/tmp/modeling/training/alligator/alligator_1.jpeg', 'rb')
    #image_read = image.read()
    #image_64_encode  = base64.encodestring(image_read)
    #image_64_encode  = image_base64_encoded
    #image_64_decode  = base64.decodestring(image_64_encode) 
    #image_64_nparray = np.frombuffer(image_64_decode, dtype=np.uint8)
    #image_64_nparray= np.frombuffer(image_64_decode, dtype=np.uint8)
    
    #print("[INFO] loading and preprocessing image...")
    #image = load_img(image_path, target_size=(224, 224))
    #image = img_to_array(image)
    
    image_data = image_base64_encoded
    #image_data = re.sub('^data:image/.+;base64,', '', image_64_encode)             # Not working
    #image = Image.open(BytesIO(base64.b64decode(image_data)))                      # Not working
    image_data = re.sub('^data:image/.+;base64,', '', image_data).decode('base64')
    image = Image.open(cStringIO.StringIO(image_data))
    image = image.resize((224,224),Image.ANTIALIAS)
    image = img_to_array(image)
    
    # important! otherwise the predictions will be '0'
    image = image / 255
    image = np.expand_dims(image, axis=0)
    
    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model_vgg16.predict(image)
    
    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    model.load_weights(model_weights_path)
    
    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)
    
    probabilities = model.predict_proba(bottleneck_prediction)
    
    inID = class_predicted[0]
    
    inv_map = {v: k for k, v in class_dictionary.items()}
    
    prediction_prob = round(probabilities[0][0], 8)
    #prediction     = inv_map[inID]
    prediction      = 'Alligator' if prediction_prob >= .8 else 'No Alligator'
    
    return prediction, float(prediction_prob)


###################################################################################################
#
#   Configure Image
#
###################################################################################################

UPLOAD_FOLDER = './static/uploaded_files'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


################################################################################################
#
#   Index
#
################################################################################################
@app.route('/', methods = ['GET','POST'])
@app.route('/index', methods = ['GET','POST'])
def index():
    
    if request.method == 'GET':
        image_path          = get_random_image()
        image_path_html     = re.sub('.*?static','./static',image_path)
        #random_prob         = random.random()*10 + 88
        prediction, prediction_prob = predict_from_image_path(image_path, model_vgg16, model_weights_path, class_dictionary)
        return render_template('index.html', image_path_html=image_path_html, prediction=prediction, prediction_prob=prediction_prob)
    
    if request.method == 'POST':
        
        try:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image_path_html = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path_html)
        except:
            image_path_html = ''
        
        prediction, prediction_prob = predict_from_image_path(image_path_html, model_vgg16, model_weights_path, class_dictionary)
        
        return render_template('index.html', image_path_html=image_path_html, prediction=prediction, prediction_prob=prediction_prob)



################################################################################################
#
#   API
#
################################################################################################
@app.route('/api', methods = ['GET','POST'])
def api():
    if request.method == 'POST':
        
        '''
        # This section is used if an image_path is being POSTed
        curl -i -H "Content-Type: multipart/form-data" -X POST -F 'file=@/Users/dzaratsian/Dropbox/code/python/flask_image_prediction/static/model_images/themepark_01.jpeg' http://localhost:4444/api
        curl -i -H "Content-Type: application/json" -X POST -d '{"file":"Spectre"}' http://localhost:5555/api
        
        try:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image_path_html = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path_html)
        except:
            image_path_html = ''
        
        prediction, prediction_prob = predict_from_image_path(image_path, model_vgg16, model_weights_path, class_dictionary)
        '''
        
        '''
        curl -i -H "Content-Type: multipart/form-data" -X POST -F 'data=@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\n' http://localhost:4444/api
        (echo -n '{"data": "'; base64 /Users/dzaratsian/hortonworks/Accounts/Disney/alligator_modeling/modeling/training/alligator/alligator_1.jpeg; echo '"}') | curl -H "Content-Type: application/json" -d @-  http://localhost:4444/api
        '''
        
        image_base64_encoded = re.sub('^data:image/.+;base64,', '', request.form['image'])
        
        prediction, prediction_prob = predict_from_base64_image(image_base64_encoded, model_vgg16, model_weights_path, class_dictionary)       
        
        response = {'prediction_label':prediction, 'prediction_prob':prediction_prob}
        
        return jsonify(response)



################################################################################################
#
#   Run App
#
################################################################################################

if __name__ == "__main__":
    #app.run(debug=True, threaded=False, host='0.0.0.0', port=4444)
    app.run(threaded=False, host='0.0.0.0', port=4444)


################################################################################################
#   
#   References:
#       
#       Encoding / Decoding
#       https://code.tutsplus.com/tutorials/base64-encoding-and-decoding-using-python--cms-25588
#       https://stackoverflow.com/questions/26070547/decoding-base64-from-post-to-use-in-pil
#
#
################################################################################################

#ZEND
