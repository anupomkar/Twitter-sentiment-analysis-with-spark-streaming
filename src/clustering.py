from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

clf = MiniBatchKMeans(n_clusters=2)
count = 0

def cluster(X,y):

    global count
    count +=1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=42)
    clf.partial_fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    f = open('Clustering','wb')
    pickle.dump(clf, f)
    f.close()

    print('accuracy: '+str(accuracy_score(y_test,y_pred)))
    print("==============================Batch "+ str(count) + " Clustered========================================")


