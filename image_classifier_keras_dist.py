
####################################################################################################
#
#   Distributed Deep Learning  -  Using Keras and Apache Spark
#
#   https://github.com/cerndb/dist-keras
#
####################################################################################################

import os
import datetime, time
import requests
import numpy as np

from keras.optimizers import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from distkeras.trainers import *
from distkeras.predictors import *
from distkeras.transformers import *
from distkeras.evaluators import *
from distkeras.utils import *
#from distkeras.utils import serialize_keras_model
#from distkeras.utils import deserialize_keras_model

application_name    = "dist_keras"
local               = False

if local:
    master          = "local[*]"
    num_cores       = 3
    num_executors   = 1
else:
    master          = "yarn-client"
    num_executors   = 6
    num_cores       = 2 

num_workers = num_executors * num_cores
print("Number of desired executors: " + `num_executors`)
print("Number of desired cores / executor: " + `num_cores`)
print("Total number of workers: " + `num_workers`)

conf = SparkConf()
conf.set("spark.app.name", application_name)
conf.set("spark.master", master)
conf.set("spark.executor.cores", `num_cores`)
conf.set("spark.executor.instances", `num_executors`)
conf.set("spark.locality.wait", "0")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
spark = SparkSession.builder.config(conf=conf)  \
        .appName(application_name)              \
        .getOrCreate()

'''
Structure of "list_of_image_urls.csv":
label, image_path_or_url
'''

raw_dataset = spark.read.csv                                        \
                    .options(header='true', inferSchema='true')     \
                    .load("/tmp/list_of_image_urls.csv")

raw_dataset.printSchema()

def convert_image_to_array():
    return None

# Define (extract) features
features = raw_dataset.columns
features.remove('Label')

vector_assembler = VectorAssembler(inputCols=features, outputCol="features")

dataset = vector_assembler.transform(raw_dataset)

dataset.select("features").show()

standard_scaler = StandardScaler(inputCol="features", outputCol="features_normalized", withStd=True, withMean=True)
standard_scaler_model = standard_scaler.fit(dataset)

dataset = standard_scaler_model.transform(dataset)

label_indexer = StringIndexer(inputCol="Label", outputCol="label_index").fit(dataset)

dataset = label_indexer.transform(dataset)

dataset.select("Label", "label_index").show()

# Neural Net Properties
nb_classes  = 2 # Number of output classes / categories 
nb_features = len(features)
optimizer   = 'adagrad'
loss        = 'categorical_crossentropy'

transformer = OneHotTransformer(output_dim=nb_classes, input_col="label_index", output_col="label")
dataset = transformer.transform(dataset)
dataset = dataset.select("features_normalized", "label_index", "label")

dataset.select("label_index", "label").show()

dataset = shuffle(dataset)

(training_set, test_set) = dataset.randomSplit([0.6, 0.4])
training_set.cache()
test_set.cache()

model = Sequential()
model.add(Dense(500, input_shape=(nb_features,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()

def evaluate_accuracy(model):
    global test_set
    # Allocate a Distributed Keras Accuracy evaluator.
    evaluator = AccuracyEvaluator(prediction_col="prediction_index", label_col="label_index")
    # Clear the prediction column from the testset.
    test_set = test_set.select("features_normalized", "label_index", "label")
    # Apply a prediction from a trained model.
    predictor = ModelPredictor(keras_model=trained_model, features_col="features_normalized")
    test_set = predictor.predict(test_set)
    # Allocate an index transformer.
    index_transformer = LabelIndexTransformer(output_dim=nb_classes)
    # Transform the prediction vector to an indexed label.
    test_set = index_transformer.transform(test_set)
    # Fetch the score.
    score = evaluator.evaluate(test_set)
    return score

def add_result(trainer, accuracy, dt):
    global results;
    # Store the metrics.
    results[trainer] = {}
    results[trainer]['accuracy'] = accuracy;
    results[trainer]['time_spent'] = dt
    # Display the metrics.
    print("Trainer: " + str(trainer))
    print(" - Accuracy: " + str(accuracy))
    print(" - Training time: " + str(dt))

results = {}

####################################################################################################
#
#   Single Optimizer (used as benchmark)
#
####################################################################################################
trainer = SingleTrainer(keras_model=model, worker_optimizer=optimizer,
                        loss=loss, features_col="features_normalized",
                        label_col="label", num_epoch=1, batch_size=32)

trained_model = trainer.train(training_set)

accuracy = evaluate_accuracy(trained_model)
dt = trainer.get_training_time()

# Add the metrics to the results.
add_result('single', accuracy, dt)


####################################################################################################
#
#   Distributed Optimizer: ADAG (Recommended)
#   Other optimizers located here: https://github.com/cerndb/dist-keras#optimization-algorithms
#
####################################################################################################
trainer = ADAG(keras_model=model, worker_optimizer=optimizer, loss=loss, metrics=["accuracy"], 
                num_workers=2, batch_size=32, num_epoch=1,
                features_col="features_normalized", label_col="label",
                communication_window=12)

trainer.set_parallelism_factor(1)
trained_model = trainer.train(training_set)

# Save / Write Model
# https://github.com/cerndb/dist-keras/blob/a6d56dd4127f7f2079e32e69f46108f8c5514bad/distkeras/job_deployment.py
with open("/models/dist_keras_model01", "w") as f:
    f.write(pickle_object(serialize_keras_model(trained_model)))

accuracy = evaluate_accuracy(trained_model)
dt = trainer.get_training_time()

# Add the metrics to the results.
add_result('adag', accuracy, dt)



#ZEND
