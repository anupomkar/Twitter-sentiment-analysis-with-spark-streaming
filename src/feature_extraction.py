from pyspark.ml.feature import HashingTF,IDF
import pyspark.sql.functions as F
import pyspark.sql.types as T
import ast
from pyspark.sql.types import IntegerType,DoubleType,ArrayType
from operator import attrgetter
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import numpy as np

from pyspark.sql.functions import udf

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

    df = df.withColumn("words_lemmatised",retrieve_array(F.col("words_lemmatised"))).select('sentiment','words_lemmatised')
    hashingTF = HashingTF(inputCol="words_lemmatised", outputCol="features",numFeatures=2**18)
    rescaledData = hashingTF.transform(df)
    rescaledData = rescaledData.select('features','sentiment')

    X = rescaledData.select('features')
    y = rescaledData.select(rescaledData.sentiment.cast(IntegerType()))
    y = np.array(y.collect()).T
    y = np.array(y[0])

    features = X.rdd.map(attrgetter("features"))
    mats = features.map(as_matrix)
    mat = mats.reduce(lambda x, y: vstack([x, y]))

    mat = np.array(mat.todense())
    # print(mat.shape)


    return mat,y



def Decompose(X):
    svd = TruncatedSVD(n_components=min(X.shape[0],X.shape[1]),n_iter=7, random_state=42)
    svd.fit(X)
    transformed = svd.transform(X)
    return transformed


# # Find the optimal number of PCA's
# for i in range(np.cumsum(ratios).shape[0]):
#   if np.cumsum(ratios)[i] >= 0.99:
#     num_pca = i + 1
#     print "The optimal number of PCA's is: {}".format(num_pca)
#     break
#   else:
#     continue




