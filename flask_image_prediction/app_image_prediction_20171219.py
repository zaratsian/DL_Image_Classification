

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


def predict(image_path, model_vgg16, model_weights_path, class_dictionary):
    
    num_classes = len(class_dictionary)
    
    # add the path to your test image below
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
        random_prob         = random.random()*10 + 88
        prediction, prediction_prob = predict(image_path, model_vgg16, model_weights_path, class_dictionary)
        return render_template('index.html', image_path=image_path_html, prediction=prediction, prediction_prob=prediction_prob, random_prob=random_prob)
    
    if request.method == 'POST':
        row_number      = int(request.form.get('row_number',''))
        datestamp       = request.form.get('datestamp','')
        posteam         = request.form.get('posteam','')
        DefensiveTeam   = request.form.get('DefensiveTeam','')
        #drive          = request.form.get('drive','')
        qtr             = int(request.form.get('quarter',''))
        down            = int(request.form.get('down',''))
        TimeSecs        = int(request.form.get('timesecs',''))
        yrdline100      = int(request.form.get('yrdline100',''))
        ydstogo         = int(request.form.get('ydstogo',''))
        ydsnet          = int(request.form.get('ydsnet',''))
        month_day       = int( datestamp[5:7] + datestamp[8:10] )
        PlayType_lag    = request.form.get('playtype_lag','')
        
        best_play, passing_yards, running_yards = predict_play(model_pass, model_run, qtr, down, TimeSecs, yrdline100, ydstogo, ydsnet, month_day, posteam, DefensiveTeam, PlayType_lag)
        #best_play, passing_yards, running_yards = predict_play(model_pass, model_run, qtr, down, TimeSecs, yrdline100, ydstogo, ydsnet, month_day, posteam, DefensiveTeam, PlayType_lag)
        
        row_number = row_number + 1
        next_play, date = get_next_play(rawdata, row_number)
        
        #return render_template('index.html', playtypes=playtypes, next_play=next_play, date=date, row_number=row_number)
        return render_template('index.html', playtypes=playtypes, posteams=list_of_teams, DefensiveTeams=list_of_teams, next_play=next_play, date=date, row_number=row_number, best_play=best_play, passing_yards=round(passing_yards,2), running_yards=round(running_yards,2))



################################################################################################
#
#   API
#
################################################################################################
@app.route('/api', methods = ['GET','POST'])
def api():
    if request.method == 'POST':
        '''
        curl -i -H "Content-Type: application/json" -X POST -d '{"movie_title":"Spectre"}' http://localhost:5555/api
        '''
        
        movie_title = request.json.get('movie_title','')
        response = score_movie(movie_title)
        
        return jsonify(response)



################################################################################################
#
#   Run App
#
################################################################################################

if __name__ == "__main__":
    
    #app.run(debug=True, threaded=False, host='0.0.0.0', port=4444)
    app.run(threaded=False, host='0.0.0.0', port=4444)




#ZEND
