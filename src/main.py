
import time
from pyspark import SparkContext
from pyspark.sql.functions import udf, variance
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
import text_preprocessing as tp
import feature_extraction as fe
import numpy as np
import GBmodel as gb


sc = SparkContext.getOrCreate()
sc.setLogLevel('OFF')
ssc = StreamingContext(sc, 1)
spark = SparkSession(sc)

data = ssc.socketTextStream("localhost", 3000)


def readMyStream(rdd):
    df = spark.read.json(rdd)
    processed_df = tp.preProcessText(df)
    feature,label = fe.extractFeatures(processed_df)
    #decomposed_features = fe.Decompose(feature)
    score1,score2,score3,score4= gb.fitGaussianNB_model(feature,label)
    print(score1,score2,score3,score4)



data.foreachRDD( lambda rdd: readMyStream(rdd) )



ssc.start()
#ssc.awaitTermination()
time.sleep(10000)
ssc.stop(stopSparkContext=False)


