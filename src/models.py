from pyspark.sql.functions import count
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB,GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle




clf1 = MultinomialNB(alpha=0.01)
#clf1 =  GaussianNB()
clf2 =  SGDClassifier(max_iter=5)
clf3 =  Perceptron()
clf4 =  PassiveAggressiveClassifier()

count = 0




def fit_model(X,y):

  '''
    Input: Array like vectors X,y
    Output: None

    The models are fitted and trained incrementally using partial_fit and the model is dumped into pickle file and stored on disk to test on test data
  '''

  no = np.unique(y)
  global count
  count +=1

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=42)
  

  try:
    clf1.partial_fit(X_train,y_train,classes=no)
    clf2.partial_fit(X_train,y_train,classes=no)
    clf3.partial_fit(X_train,y_train,classes=no)
    clf4.partial_fit(X_train,y_train,classes=no)
  
    acc1,acc2,acc3,acc4 = model_Evaluate(clf1,X_test,y_test),model_Evaluate(clf2,X_test,y_test),model_Evaluate(clf3,X_test,y_test),model_Evaluate(clf4,X_test,y_test)
    print("==============================Batch "+ str(count) + " completed Processing========================================")
    print('Model Accuracies:')
    print('MultinominalNB '+str(acc1))
    print('SGDClassifier '+str(acc2))
    print('Perceptron '+str(acc3))
    print('PassiveAggressiveClassifier '+str(acc4))


    f1_acc = open('MultinominalNB_acc.txt','a')
    f1_acc.write('%s\n' %acc1)
    f2_acc = open('SGDClassifier_acc.txt','a')
    f2_acc.write('%s\n' %acc2)
    f3_acc = open('Perceptron_acc.txt','a')
    f3_acc.write('%s\n' %acc3)
    f4_acc = open('PassiveAgressiveClassifier_acc.txt','a')
    f4_acc.write('%s\n' %acc4)

    f1_acc.close()
    f2_acc.close()
    f3_acc.close()
    f4_acc.close()

    f1 = open('MultinominalNB','wb')
    f2 = open('SGDClassifier','wb')
    f3 = open('Perceptron','wb')
    f4 = open('PassiveAggressiveClassifier','wb')

    pickle.dump(clf1, f1)
    pickle.dump(clf2, f2)
    pickle.dump(clf3, f3)
    pickle.dump(clf4, f4)

    f1.close()
    f2.close()
    f3.close()
    f4.close()


  except Exception as e:
    print(e)

def model_Evaluate(model,X_test,y_test):

  y_pred = model.predict(X_test)
  accuracy= accuracy_score (y_test, y_pred)
  return accuracy



