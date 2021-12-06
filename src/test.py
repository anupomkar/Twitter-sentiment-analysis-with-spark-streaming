from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

import pickle

cm1 = cm2 = cm3 = cm4 = np.zeros((2,2))
count = 0

def test_model(X_test,y_test):

    global cm1,cm2,cm3,cm4,count

    MultinominalNB_model = pickle.load(open('MultinominalNB', 'rb'))
    Perceptron_model = pickle.load(open('Perceptron','rb'))
    PassiveAggressiveClassifier_model = pickle.load(open('PassiveAggressiveClassifier','rb'))
    SGDClassifier_model = pickle.load(open('SGDClassifier','rb'))


    y_pred1 = MultinominalNB_model.predict(X_test)
    y_pred2 = Perceptron_model.predict(X_test)
    y_pred3 = PassiveAggressiveClassifier_model.predict(X_test)
    y_pred4 = SGDClassifier_model.predict(X_test)

    cm1 = np.add(cm1,confusion_matrix(y_test,y_pred1))
    cm2 = np.add(cm2,confusion_matrix(y_test,y_pred2))
    cm3 = np.add(cm3,confusion_matrix(y_test,y_pred3))
    cm4 = np.add(cm4,confusion_matrix(y_test,y_pred4))


    with open('Confusion_Matrix.txt','w') as conf_mat:
        conf_mat.write('\nMultinominalNB\n')
        for item in cm1.flatten():
            conf_mat.write("%s\n" % item)
        conf_mat.write('\nPerceptron\n')
        for item in cm2.flatten():
            conf_mat.write("%s\n" % item)
        conf_mat.write('\nPassiveAggressiveClassifier\n')
        for item in cm3.flatten():
            conf_mat.write("%s\n" % item)
        conf_mat.write('\nSGDClassifier\n')
        for item in cm4.flatten():
            conf_mat.write("%s\n" % item)



    

    with open('MultinominalNB_performance.txt', 'a') as f1:
        f1.writelines(classification_report(y_test,y_pred1))
        f1.write('\n===============================================================================\n')
    
    with open('Perceptron_performance.txt', 'a') as f2:
        f2.writelines(classification_report(y_test,y_pred2))
        f2.write('\n===============================================================================\n')

    with open('PassiveAggressiveClassifier_performance.txt', 'a') as f3:
        f3.writelines(classification_report(y_test,y_pred3))
        f3.write('\n===============================================================================\n')

    with open('SGDClassifier_performance.txt', 'a') as f4:
        f4.writelines(classification_report(y_test,y_pred4))
        f4.write('\n===============================================================================\n')

    
    count+=1
    print('Batch '+str(count)+' Completed')

