import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


lines = []
with open('Confusion_Matrix.txt') as f:
    lines = f.readlines()

l = []
for line in lines:
    line = line.strip()
    if line=='':
        continue
    l.append(line)

# Plot the graph to vizualize the evaluation metrices among 4 models

TP1,FP1,FN1,TN1 = [int(float(i)) for i in l[1:5]]     # Multinomial Naive Bayes
TP2,FP2,FN2,TN2 = [int(float(i)) for i in l[6:10]]   #Perceptron
TP3,FP3,FN3,TN3 = [int(float(i)) for i in l[11:15]]  #PassiveAggressiveClassifier
TP4,FP4,FN4,TN4 = [int(float(i)) for i in l[16:20]]   #SGD classifier


Accuracy1 = (TP1+TN1)/(TP1+FP1+FN1+TN1)
Precision1 = TP1/(TP1+FP1)
Recall1 = TP1/(TP1+FN1)
F1_Score1 = 2*(Recall1 * Precision1) / (Recall1 + Precision1)

Accuracy2 = (TP2+TN2)/(TP2+FP2+FN2+TN2)
Precision2 = TP2/(TP2+FP2)
Recall2 = TP2/(TP2+FN2)
F1_Score2 = 2*(Recall2 * Precision2) / (Recall2 + Precision2)

Accuracy3 = (TP3+TN3)/(TP3+FP3+FN3+TN3)
Precision3 = TP3/(TP3+FP3)
Recall3 = TP3/(TP3+FN3)
F1_Score3 = 2*(Recall3 * Precision3) / (Recall3 + Precision3)

Accuracy4 = (TP4+TN4)/(TP4+FP4+FN4+TN4)
Precision4 = TP4/(TP4+FP4)
Recall4 = TP4/(TP4+FN4)
F1_Score4 = 2*(Recall4 * Precision4) / (Recall4 + Precision4)



X = ['MultinomialNB','Perceptron','PassiveAggressiveClassifier','SGDClassifier']
Acc = [Accuracy1,Accuracy2,Accuracy3,Accuracy4]
Pre = [Precision1,Precision2,Precision3,Precision4]
Rec = [Recall1,Recall2,Recall3,Recall4]
F1_Sc = [F1_Score1,F1_Score2,F1_Score3,F1_Score4]

ind = np.arange(4)
width = 0.1
bar1 = plt.bar(ind, Acc, width, color = 'r')
bar2 = plt.bar(ind+width, Pre, width, color='g')
bar3 = plt.bar(ind+width*2, Rec, width, color = 'b')
bar4 = plt.bar(ind+width*3, F1_Sc, width, color = 'black')

  
plt.xlabel("Models")
plt.ylabel('Scores')
plt.title("Model Scores")


plt.xticks(ind+width,X)
plt.legend((bar1, bar2, bar3,bar4),('Accuracy','Precision','Recall','F1_Score'))
plt.show()


# Plot the confusion matrix using heat map

data1 = np.array([TP1,FP1,FN1,TN1]).reshape(2,2)
data2 = np.array([TP2,FP2,FN2,TN2]).reshape(2,2)
data3 = np.array([TP3,FP3,FN3,TN3]).reshape(2,2)
data4 = np.array([TP4,FP4,FN4,TN4]).reshape(2,2)

ax = plt.axes()
sns.heatmap(data1, xticklabels=[4,0], yticklabels=[4,0],annot=True,fmt='d')
ax.set_title('MultiNominalNB')
plt.show()
ax = plt.axes()
sns.heatmap(data2, xticklabels=[4,0], yticklabels=[4,0],annot=True,fmt='d')
ax.set_title('Perceptron')
plt.show()
ax = plt.axes()
sns.heatmap(data3, xticklabels=[4,0], yticklabels=[4,0],annot=True,fmt='d')
ax.set_title('PassiveAggressiveClassifier')
plt.show()
ax = plt.axes()
sns.heatmap(data4, xticklabels=[4,0], yticklabels=[4,0],annot=True,fmt='d')
ax.set_title('SGDclassifer')
plt.show()





        