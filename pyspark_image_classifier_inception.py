
############################################################################################################
#
#   Image Classification based on Pre-Trained Inception Model
#   
#   This script will score a directory against the pre-trained Inception model. The output will then 
#   be used to flag all alligator images. 
#
#   Accuracy: 0.92563634021 
#
'''
pip install numpy
pip install pandas
pip install nose
pip install pillow
pip install keras
pip install h5py
pip install py4j
pip install tensorflow

/usr/hdp/current/spark2-client/bin/spark-submit                                     \
    --master yarn                                                                   \
    --deploy-mode client                                                            \
    --driver-memory 10G                                                             \
    --executor-memory 10G                                                           \
    --num-executors 3                                                               \
    --packages databricks:spark-deep-learning:0.2.0-spark2.1-s_2.11                 \
    --conf "spark.pyspark.python=/opt/anaconda2/bin/python"                         \
    pyspark_image_classifier_inception.py                                           \
    /tmp/images_alligators                                                          \
    alligator
'''
############################################################################################################

import os, sys, re
import glob
sys.path.extend(glob.glob(os.path.join(os.path.expanduser("~"), ".ivy2/jars/*.jar")))
from sparkdl import readImages, DeepImagePredictor
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf, col, lit, monotonically_increasing_id

spark = SparkSession.builder.appName('alligator_inception').getOrCreate()

try:
    images_filepath = sys.argv[1]
    print '[ INFO ] Using image directory ' + str(images_filepath)
except:
    print '[ WARNING ] No image directory provided, using default image directory /tmp/images_alligators'
    images_filepath = '/tmp/images_alligators'

try:
    search_term = sys.argv[2]
    print '[ INFO ] Using search term ' + str(search_term)
except:
    print '[ WARNING ] No search term provided, using default search term "alligator"'
    search_term = 'alligator'

def score_inceptionV3(images_filepath):
    inceptionV3 = DeepImagePredictor(inputCol="image", outputCol="predicted_labels", modelName="InceptionV3", decodePredictions=True, topK=5)
    image_df    = readImages(images_filepath)
    predictions = inceptionV3.transform(image_df)
    return predictions

print '[ INFO ] Scoring images directory against pre-trained Inception model...'
inceptionV3_predictions = score_inceptionV3(images_filepath)

def flag_string_match(search_term, df=inceptionV3_predictions):
    '''
    Flag all of the record matches that contain the "search_term"
    "df" is the predictions DataFrame generated from the pre-trained inception model.
    '''
    def search_term_flag(predicted_labels):
        search_term_flags = [1 if re.search(search_term,str(label).lower()) else 0 for label in predicted_labels]
        return 1 if sum(search_term_flags)>=1 else 0
    
    udf_search_term_flag = udf(search_term_flag, IntegerType())
    
    results = df.withColumn('search_term', udf_search_term_flag('predicted_labels')  )
    
    return results

print '[ INFO ] Scanning results for search term = ' + str(search_term)
results = flag_string_match(search_term, df=inceptionV3_predictions)

# Count the number of matches (and misclassifications)
results_agg = results.groupBy( col('search_term') ) \
                     .count() \
                     .withColumn('pct', col('count') / lit(108) ) \

results_agg.show(10,False)

print '[ INFO ] Printing filepath(s) of misclassified images...'
results.filter('search_term == 0').select('filePath').withColumnRenamed('filePath','Misclassified_Images').show(10,False)

print '[ INFO ] Accuracy Score:    ' + str(results_agg.select('pct').filter('search_term==1').collect()[0][0])


#ZEND

