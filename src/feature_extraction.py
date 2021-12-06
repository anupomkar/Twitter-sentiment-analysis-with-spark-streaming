from pyspark.ml.feature import HashingTF
import pyspark.sql.functions as F
import pyspark.sql.types as T
import ast
from operator import attrgetter
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import numpy as np


# Helper functions to tranform the sparse vectors in pyspark dataframe to required matrix

def parse_array_from_string(x):
    x = ast.literal_eval(str(x))
    res = [n.strip() for n in x]
    return res

retrieve_array = F.udf(parse_array_from_string, T.ArrayType(T.StringType()))

def as_matrix(vec):
    data, indices = vec.values, vec.indices
    shape = 1, vec.size
    return csr_matrix((data, indices, np.array([0, vec.values.size])), shape)



def extractFeatures(df): 
    '''
        Input : Pyspark Dataframe
        Output: feature vector and label

        It uses HashingTF vectorizer to convert a collection of text documents to a matrix of token occurrences
        With HashingVectorizer, each token directly maps to a column position in a matrix, where its size is pre-defined.
        For example, if you have 10,000 columns in your matrix, each token maps to 1 of the 10,000 columns. 
    '''   

    df = df.withColumn("words_lemmatised",retrieve_array(F.col("words_lemmatised"))).select('sentiment','words_lemmatised')
    hashingTF = HashingTF(inputCol="words_lemmatised", outputCol="features",numFeatures=2**18)
    rescaledData = hashingTF.transform(df)
    rescaledData = rescaledData.select('features','sentiment')

    X = rescaledData.select('features')
    y = rescaledData.select(rescaledData.sentiment.cast(T.IntegerType()))
    y = np.array(y.collect()).T
    y = np.array(y[0])

    features = X.rdd.map(attrgetter("features"))
    mats = features.map(as_matrix)
    mat = mats.reduce(lambda x, y: vstack([x, y]))
    mat = np.array(mat.todense())

    # print(mat.shape)

    return mat,y






