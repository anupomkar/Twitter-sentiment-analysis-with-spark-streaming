
import time
from pyspark import SparkContext
from pyspark.sql.functions import udf, variance
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
import text_preprocessing as tp
import feature_extraction as fe
import numpy as np
import models
import test
import clustering

import warnings
warnings.filterwarnings("ignore")



sc = SparkContext.getOrCreate()
sc.setLogLevel('OFF')
ssc = StreamingContext(sc,1)
spark = SparkSession(sc)


data = ssc.socketTextStream("localhost", 3000)


def readMyStream(rdd):
    if (not rdd.isEmpty()):

        df = spark.read.json(rdd)
        processed_df = tp.preProcessText(df)
        feature,label = fe.extractFeatures(processed_df)
        '''
            Undo Comment for models.fit to train the models
            1. MultinominalNB
            2. Perceptron
            3. PassiveAggressiveClassifier
            4. SGDClassifier

            Undo the test.test_model to test and evaluate the model

            Undo the clustering.cluster to apply k-means cluster

        '''
        #models.fit_model(feature,label)
        #test.test_model(feature,label)
        #clustering.cluster(feature,label)

data.foreachRDD(lambda rdd: readMyStream(rdd))


ssc.start()
ssc.awaitTermination()
ssc.stop(stopSparkContext=False)


