
############################################################################################################
#
#   Spark Deep Learning - Image classifiation (tensorflow + spark) 
#
#   This script uses Spark DL transfer learning plus a Spark ML algorigthm to classify images
#
#   Accuracy: 0.948717948718 
#
#   Usage:
'''
/usr/hdp/current/spark2-client/bin/spark-submit                                     \
    --master yarn                                                                   \
    --deploy-mode client                                                            \
    --driver-memory 10G                                                             \
    --executor-memory 10G                                                           \
    --num-executors 3                                                               \
    --packages databricks:spark-deep-learning:0.2.0-spark2.1-s_2.11                 \
    --conf "spark.pyspark.python=/opt/anaconda2/bin/python"                         \
    pyspark_image_classifier_alligator_sparkdl.py
'''
############################################################################################################

import os, sys, re
import glob
sys.path.extend(glob.glob(os.path.join(os.path.expanduser("~"), ".ivy2/jars/*.jar")))
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf, col, lit, monotonically_increasing_id
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from sparkdl import readImages

spark = SparkSession.builder.appName('alligator_training').getOrCreate()

img_alligator_df = readImages("/tmp/modeling/training/alligator").withColumn("label", lit(1)).where(col("image").isNotNull())
img_other_df     = readImages("/tmp/modeling/training/not_alligator").withColumn("label", lit(0)).where(col("image").isNotNull())

#img_other_df.withColumn('uid',monotonically_increasing_id()).filter('uid < 10').count()
#img_other_df.show()

# Testing and Train Split (I'm using 40/60 because I was running out of memory when doing a higher training pct)
training_pct = 0.30
testing_pct  = 0.70

alligator_train, alligator_test  = img_alligator_df.randomSplit([training_pct, testing_pct])
other_train,     other_test      = img_other_df.randomSplit([training_pct, testing_pct])

train_df = alligator_train.unionAll(other_train)
print('[ INFO ] Number of Training Records: ' + str(train_df.count()))

test_df = alligator_test.unionAll(other_test)
print('[ INFO ] Number of Training Records: ' + str(test_df.count()))

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
gb = GBTClassifier(featuresCol="features", labelCol="label", predictionCol="prediction", maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, lossType="logistic", maxIter=5, stepSize=0.1, seed=None, subsamplingRate=1.0)
pipe = Pipeline(stages=[featurizer, lr])
print('[ INFO ] Fitting model pipeline...')
pipe_model = pipe.fit(train_df)
# Save pipe_model
#pipe_model.write().overwrite().save("/tmp/spark_alligator_model_lr")

predictions = pipe_model.transform(test_df)

#print('[ INFO ] Printing filepath and predictions (1=alligator)...')
#predictions.select("filePath", "prediction").show(150,False)

predictionAndLabels = predictions.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print('[ INFO ] Training set accuracy = ' + str(evaluator.evaluate(predictionAndLabels)))


#ZEND
