import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# Load data
df = pd.read_csv('ISB_Data_Prediction.csv')
print (df)

X = df.ix[:,0:3]
Y = df.ix[:,3]

le = LabelEncoder()
y = le.fit_transform(Y)

print('Class labels:', np.unique(y))

print("After Extraction")
print(X)
print(y)

#from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y,
	test_size=0.25)

# train and evaluate a ANN classifier on the raw pixel intensities
from sklearn.neural_network import MLPClassifier
# Initialize ANN classifier
mlp = MLPClassifier(hidden_layer_sizes=(8), activation='logistic', max_iter =1500)

# Train the classifier with the traning data
mlp.fit(trainX,trainY)

# generate evaluation metrics
print ("Train - Accuracy :", metrics.accuracy_score(trainY, mlp.predict(trainX)))
print ("Train - Confusion matrix :",metrics.confusion_matrix(trainY, mlp.predict(trainX)))
print ("Train - classification report :", metrics.classification_report(trainY, mlp.predict(trainX)))
print ("Test - Accuracy :", metrics.accuracy_score(testY, mlp.predict(testX)))
print ("Test - Confusion matrix :",metrics.confusion_matrix(testY, mlp.predict(testX)))
print ("Test - classification report :", metrics.classification_report(testY, mlp.predict(testX)))

prd_rs = mlp.predict(testX)
test_acc = metrics.accuracy_score(testY, prd_rs) * 100.
loss_values_s = mlp.loss_curve_
plt.plot(loss_values_s,'r--',label='Test')

prd_rt = mlp.predict(trainX)
train_acc = metrics.accuracy_score(trainY, prd_rt) * 100.
loss_values_t = mlp.loss_curve_
plt.plot(loss_values_t,'b',label='Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.title('')
plt.legend()
plt.show()

from sklearn.cross_validation import StratifiedKFold
kfold = StratifiedKFold(y, n_folds=10, random_state=2019)
    
train_scores = []
test_scores = []
for k, (train, test) in enumerate(kfold):
    mlp.fit(X[train], y[train])
    train_score = mlp.score(X[train], y[train])
    train_scores.append(train_score)
    # score for test set
    test_score = mlp.score(X[test], y[test])
    test_scores.append(test_score)
    print('Fold: %s, Class dist.: %s, Train Acc: %.3f, Test Acc: %.3f' % (k+1, np.bincount(y[train]), train_score, test_score))
    print('Fold: %s, Train Acc: %.3f, Test Acc: %.3f' % (k+1, train_score, test_score))
print('\nTrain CV accuracy: %.3f' % (np.mean(train_scores)))
print('Test CV accuracy: %.3f' % (np.mean(test_scores)))




