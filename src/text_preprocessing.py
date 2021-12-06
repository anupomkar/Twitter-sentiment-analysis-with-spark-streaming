import re
from pyspark.sql.functions import udf, col, lower
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from nltk.stem import WordNetLemmatizer

#nltk.download('wordnet')   


urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern = '@[^\s]+'
hashtagPatten = r'\B#\S+'
alphaPattern = "[^a-zA-Z0-9]"
sequencePattern = r"(.)\1\1+"
seqReplacePattern = r"\1\1"


clean_url = lambda s: re.sub(urlPattern,'URL',s) # Replace all URls with 'URL'
clean_twitter_handler = lambda s: re.sub(userPattern,'',s) # Replace @USERNAME to 'USER'.
clean_hashtag = lambda s: re.sub(hashtagPatten,'',s) #remove hashtags
clean_sequence = lambda s: re.sub(sequencePattern,seqReplacePattern,s)  # Replace 3 or more consecutive letters by 2 letter.
clean_extra_spaces = lambda s:re.sub(r'\s+',' ', s, flags=re.I) # Substituting multiple spaces with single space
clean_quotes = lambda s:re.sub(r'"','',s)  #remove quotation marks
clean_alpha = lambda x:' '.join(re.findall(r'\w+', x)) # Remove all the special characters


# Helper Functions for preprocessing

def cleanText(df):

    df = df.withColumn('cleaned_tweet', lower(col('tweet'))).select('sentiment','cleaned_tweet')
    df = df.withColumn('cleaned_tweet', udf(clean_url, StringType())('cleaned_tweet')).select('sentiment','cleaned_tweet')
    df = df.withColumn('cleaned_tweet', udf(clean_twitter_handler, StringType())('cleaned_tweet')).select('sentiment','cleaned_tweet')
    df = df.withColumn('cleaned_tweet', udf(clean_hashtag, StringType())('cleaned_tweet')).select('sentiment','cleaned_tweet')
    df = df.withColumn('cleaned_tweet', udf(clean_sequence, StringType())('cleaned_tweet')).select('sentiment','cleaned_tweet')
    df = df.withColumn('cleaned_tweet', udf(clean_quotes, StringType())('cleaned_tweet')).select('sentiment','cleaned_tweet')
    df = df.withColumn('cleaned_tweet', udf(clean_alpha, StringType())('cleaned_tweet')).select('sentiment','cleaned_tweet')
    df = df.withColumn('cleaned_tweet', udf(clean_extra_spaces, StringType())('cleaned_tweet')).select('sentiment','cleaned_tweet')
    
    return df

def tokenize(df):
    tokenizer = Tokenizer(inputCol='cleaned_tweet', outputCol='words_token')
    df = tokenizer.transform(df).select('sentiment', 'words_token')
    return df

def removeStopWords(df):
    remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean')
    df = remover.transform(df).select('sentiment', 'words_clean')
    return df


def lemmatization(df):
    wnl = WordNetLemmatizer()
    lemmatizer_udf = udf(lambda tokens: [wnl.lemmatize(token) for token in tokens], StringType())
    df = df.withColumn("words_lemmatised", lemmatizer_udf("words_clean")).select('sentiment', 'words_lemmatised')
    return df



def preProcessText(df):

    '''
        Input : Pyspark Dataframe df
        Output : Processed Dataframe df'

        The Preprocessing Steps include:
        1. Cleaning the raw tweets
        2. Tokenization
        3. Removal of stop words
        4. Lemmatisation
    '''

    return lemmatization(removeStopWords(tokenize(cleanText(df))))


