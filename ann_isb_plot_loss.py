import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
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
	test_size=0.20, random_state=42)

# train and evaluate a ANN classifier on the raw pixel intensities
from sklearn.neural_network import MLPClassifier
# Initialize ANN classifier
mlp = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', max_iter =1500)
#mlp = MLPClassifier(hidden_layer_sizes=(100),activation='relu',solver='adam',learning_rate='adaptive', early_stopping=True)
# Train the classifier with the traning data
mlp.fit(trainX,trainY)

# generate evaluation metrics
print ("Train - Accuracy :", metrics.accuracy_score(trainY, mlp.predict(trainX)))
print ("Train - Confusion matrix :",metrics.confusion_matrix(trainY, mlp.predict(trainX)))
print ("Train - classification report :", metrics.classification_report(trainY, mlp.predict(trainX)))
print ("Test - Accuracy :", metrics.accuracy_score(testY, mlp.predict(testX)))
print ("Test - Confusion matrix :",metrics.confusion_matrix(testY, mlp.predict(testX)))
print ("Test - classification report :", metrics.classification_report(testY, mlp.predict(testX)))

##prd_rs = mlp.predict(testX)
##test_acc = metrics.accuracy_score(testY, prd_rs) * 100.
##loss_values_s = mlp.loss_curve_
##plt.plot(loss_values_s,'r--',label='Test - 20%')

prd_rt = mlp.predict(trainX)
train_acc = metrics.accuracy_score(trainY, prd_rt) * 100.
loss_values_t = mlp.loss_curve_
plt.plot(loss_values_t,'r--',label='Training - 80%')
##plt.xlabel('Epoch')
##plt.ylabel('Loss')
###plt.title('')
##plt.legend()
##plt.show()


(trainX, testX, trainY, testY) = train_test_split(X, y,	test_size=0.25, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', max_iter =1500)
#mlp = MLPClassifier(hidden_layer_sizes=(100),activation='relu',solver='adam',learning_rate='adaptive', early_stopping=True)
# Train the classifier with the traning data
mlp.fit(trainX,trainY)
##prd_rs = mlp.predict(testX)
##test_acc = metrics.accuracy_score(testY, prd_rs) * 100.
##loss_values_s = mlp.loss_curve_
##plt.plot(loss_values_s,'b-.',label='Test - 25%')

prd_rt = mlp.predict(trainX)
train_acc = metrics.accuracy_score(trainY, prd_rt) * 100.
loss_values_t = mlp.loss_curve_
plt.plot(loss_values_t,'b-.',label='Training - 75%')

(trainX, testX, trainY, testY) = train_test_split(X, y,	test_size=0.30, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', max_iter =1500)
#mlp = MLPClassifier(hidden_layer_sizes=(100),activation='relu',solver='adam',learning_rate='adaptive', early_stopping=True)
# Train the classifier with the traning data
mlp.fit(trainX,trainY)
##prd_rs = mlp.predict(testX)
##test_acc = metrics.accuracy_score(testY, prd_rs) * 100.
##loss_values_s = mlp.loss_curve_
##plt.plot(loss_values_s,'g',label='Test - 30%')

prd_rt = mlp.predict(trainX)
train_acc = metrics.accuracy_score(trainY, prd_rt) * 100.
loss_values_t = mlp.loss_curve_
plt.plot(loss_values_t,'g',label='Training - 70%')

plt.xlabel('Epoch')
plt.ylabel('Loss')
###plt.title('')
plt.legend()
plt.show()
