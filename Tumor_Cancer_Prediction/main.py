import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import joblib


from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
Tumor = pd.read_csv("Tumor Cancer Prediction_Data.csv")

# replace all null values by 0
Tumor.fillna(0, inplace=True)

# drop all duplicate rows
Tumor.drop_duplicates(inplace=True)

# change (diagnosis) column to (0 and 1) instead of (B and M) to be able to normalize the data
lb = LabelEncoder()
Tumor.iloc[:, 31] = lb.fit_transform(Tumor.iloc[:, 31].values)

# Normalizing the data
scaler = MinMaxScaler()
Tumor = pd.DataFrame(scaler.fit_transform(Tumor))

print(Tumor)

# Split the dataset into independent(X) and dependent(Y) datasets
X = Tumor.iloc[:, 1:31]
Y = Tumor.iloc[:, 31]

# Split the dataset into 70% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

print('\n\n')

# -----------------------------------------------------------------------------------------------------
# -----Logistic Regression-----

# Define the Model
log = LogisticRegression(solver='liblinear')

# Train the Model
log.fit(X_train, Y_train)

# Print the Training Accuracy score
print('[1]Logistic Regression Training Accuracy score : ', log.score(X_train, Y_train))

# Predict the response for test dataset
Y_pred_log = log.predict(X_test)

# confusion matrix
print("confusion matrix of Logistic Regression:", '\n', confusion_matrix(Y_test, Y_pred_log))

# Model Accuracy: how often is the classifier correct
# Accuracy = (tp + tn) / (tp + tn + fn + fp)
print("Accuracy of Logistic Regression:", metrics.accuracy_score(Y_test, Y_pred_log))

# Model Precision: total number of all observations that have been predicted to belong
# to the positive class and are actually positive
# Precision = tp / (tp + fp)
print("Precision of Logistic Regression:", metrics.precision_score(Y_test, Y_pred_log))

# Model Recall: This is the proportion of observation predicted to belong to the positive
# class, that truly belongs to the positive class.
# Recall = tp / (tp + fn)
print("Recall of Logistic Regression:", metrics.recall_score(Y_test, Y_pred_log))
print('\n')

# -----------------------------------------------------------------------------------------------------
# -----Decision Tree-----

# Define the Model
tree = DecisionTreeClassifier()

# Train the Model
tree.fit(X_train, Y_train)

# Print the Training Accuracy score
print('[2]Decision Tree Training Accuracy score : ', tree.score(X_train, Y_train))

# Predict the response for test dataset
Y_predtree = tree.predict(X_test)

# confusion matrix
print("confusion matrix of Decision Tree:", '\n', confusion_matrix(Y_test, Y_predtree))

# Model Accuracy: how often is the classifier correct
# Accuracy = (tp + tn) / (tp + tn + fn + fp)
print("Accuracy of Decision Tree:", metrics.accuracy_score(Y_test, Y_predtree))

# Model Precision: total number of all observations that have been predicted to belong
# to the positive class and are actually positive
# Precision = tp / (tp + fp)
print("Precision of Decision Tree:", metrics.precision_score(Y_test, Y_predtree))

# Model Recall: This is the proportion of observation predicted to belong to the positive
# class, that truly belongs to the positive class.
# Recall = tp / (tp + fn)
print("Recall of Decision Tree:", metrics.recall_score(Y_test, Y_predtree))
print('\n')

# -----------------------------------------------------------------------------------------------------
# -----Support Vector Machine (SVM)-----

# Define the Model
sv = svm.SVC(kernel='linear')

# Train the Model
sv.fit(X_train, Y_train)

# Print the Training Accuracy score
print('[3]SVM Training Accuracy score by linear kernel : ', sv.score(X_train, Y_train))

# sv = SVC(kernel='poly')
# sv.fit(X_train, Y_train)
# print('[2]SVM Training Accuracy by polynomial kernel: ', sv.score(X_train, Y_train))

# sv = SVC(kernel='rbf')
# sv.fit(X_train, Y_train)
# print('[2]SVM Training Accuracy by Radial basis function kernel: ', sv.score(X_train, Y_train))

# Predict the response for test dataset
Y_predsv = sv.predict(X_test)

# confusion matrix
print("confusion matrix of SVM:", '\n', confusion_matrix(Y_test, Y_predsv))

# Model Accuracy: how often is the classifier correct
# Accuracy = (tp + tn) / (tp + tn + fn + fp)
print("Accuracy of SVM:", metrics.accuracy_score(Y_test, Y_predsv))

# Model Precision: total number of all observations that have been predicted to belong
# to the positive class and are actually positive
# Precision = tp / (tp + fp)
print("Precision of SVM:", metrics.precision_score(Y_test, Y_predsv))

# Model Recall: This is the proportion of observation predicted to belong to the positive
# class, that truly belongs to the positive class.
# Recall = tp / (tp + fn)
print("Recall of SVM:", metrics.recall_score(Y_test, Y_predsv))
print('\n')


# -----------------------------------------------------------------------------------------------------
# -----Random forest Classifier-----

# Define the Model
forest = RandomForestClassifier()

# Train the Model
forest.fit(X_train, Y_train)

# Print the Training Accuracy score
print('[4]Random forest Classifier Training Accuracy score : ', forest.score(X_train, Y_train))

# Predict the response for test dataset
Y_predforest = forest.predict(X_test)

# confusion matrix
print("confusion matrix of Random forest Classifier:", '\n', confusion_matrix(Y_test, Y_predforest))

# Model Accuracy: how often is the classifier correct
# Accuracy = (tp + tn) / (tp + tn + fn + fp)
print("Accuracy of Random forest Classifier:", metrics.accuracy_score(Y_test, Y_predforest))

# Model Precision: total number of all observations that have been predicted to belong
# to the positive class and are actually positive
# Precision = tp / (tp + fp)
print("Precision of Random forest Classifier:", metrics.precision_score(Y_test, Y_predforest))

# Model Recall: This is the proportion of observation predicted to belong to the positive
# class, that truly belongs to the positive class.
# Recall = tp / (tp + fn)
print("Recall of Random forest Classifier:", metrics.recall_score(Y_test, Y_predforest))
print('\n')

# -----------------------------------------------------------------------------------------------------
# Print the Prediction (Logistic Regression)
print("prediction of Logistic Regression", "\n", Y_pred_log)
print('\n')

# Print the Prediction (Decision Tree)
print("prediction of Decision Tree", "\n", Y_predtree)
print('\n')

# Print the Prediction (SVM)
print("prediction of SVM", "\n", Y_predsv)
print('\n')

# Print the Prediction (Random forest Classifier)
print("prediction of Random forest Classifier", "\n", Y_predforest)
print('\n')
# -----------------------------------------------------------------------------------------------------
# Voting the prediction
counter1 = 0
counter0 = 0
for i in range(len(Y_pred_log)):

    if Y_pred_log[i] == 1:
        counter1 = counter1 + 1
    else:
        counter0 = counter0 + 1

    if Y_predtree[i] == 1:
        counter1 = counter1 + 1
    else:
        counter0 = counter0 + 1

    if Y_predsv[i] == 1:
        counter1 = counter1 + 1
    else:
        counter0 = counter0 + 1

    if counter1 > counter0:
        print("patient num ", i+1, " will have tumor")
    else:
        print("patient num ", i+1, " will not have tumor")
    counter1 = 0
    counter0 = 0
print('\n')
# -----------------------------------------------------------------------------------------------------
# Data Scaling for data set
# initialize normalizer
data_norm = Normalizer()

# Fit the data
# Normalization formula(Z) = (X - Mean)/ Variance
Normalize = data_norm.fit_transform(X.values)

# Distribution plot
# and we pur with the normalization the standardization by put [kde = True]
sns.displot(Normalize[:, 5], fill=True, color='red', kde=True)

# Add the axis labels
plt.xlabel('patient data')
plt.ylabel('diagnosis')

# Display the plot
plt.show()

# -----------------------------------------------------------------------------------------------------
# Saving and Loading Models

# Save Model to file in the current working directory (Logistic Regression)
Save_file1 = 'Tumor_Cancer_Prediction by logistic regression.sav'
joblib.dump(log, Save_file1)

# Save Model to file in the current working directory (Decision Tree)
Save_file2 = 'Tumor_Cancer_Prediction by Decision Tree.sav'
joblib.dump(tree, Save_file2)

# Save Model to file in the current working directory (SVM)
Save_file3 = 'Tumor_Cancer_Prediction by svm.sav'
joblib.dump(sv, Save_file3)

# Save Model to file in the current working directory (Random forest Classifier)
Save_file4 = 'Tumor_Cancer_Prediction by Random forest Classifier.sav'
joblib.dump(forest, Save_file4)

# Load from file (Logistic Regression)
Load_file1 = joblib.load(Save_file1)

# Load from file (Decision Tree)
Load_file2 = joblib.load(Save_file2)

# Load from file (SVM)
Load_file3 = joblib.load(Save_file3)

# Load from file (Random forest Classifier)
Load_file4 = joblib.load(Save_file4)
# -----------------------------------------------------------------------------------------------------
# this for test the new data by save and load model
# Read players data
Tumor_test = pd.read_csv("Test Data.csv")

# replace all null values by 0
Tumor_test.fillna(0, inplace=True)

# drop all duplicate rows
Tumor_test.drop_duplicates(inplace=True)

# change (diagnosis) column to (0 and 1) instead of (B and M) to be able to normalize the data
lb = LabelEncoder()
Tumor_test.iloc[:, 31] = lb.fit_transform(Tumor_test.iloc[:, 31].values)

# Normalizing the data
scaler = MinMaxScaler()
Tumor_test = pd.DataFrame(scaler.fit_transform(Tumor_test))

print(Tumor_test)

# Split the dataset into independent(X) and dependent(Y) datasets
XL = Tumor_test.iloc[:, 1:31]
YL = Tumor_test.iloc[:, 31]

# Split the dataset into 70% training and 25% testing
Xtrain, Xtest, Ytrain, Ytest = train_test_split(XL, YL, test_size=0.25, random_state=0)

print('\n\n')
# -----------------------------------------------------------------------------------------------------
# Print the Accuracy score(Logistic Regression)
print('[1]Logistic Regression Training Accuracy score (Load file) : ', Load_file1.score(Xtrain, Ytrain))
print('\n')

# Print the Accuracy score(Decision Tree)
print('[2]Decision Tree Training Accuracy score (Load file) : ', Load_file2.score(Xtrain, Ytrain))
print('\n')

# Print the Accuracy score(SVM)
print('[3]SVM Training Accuracy score by linear kernel (Load file) : ', Load_file3.score(Xtrain, Ytrain))
print('\n')

# Print the Accuracy score(Random forest Classifier)
print('[4]Random forest Classifier Training Accuracy score (Load file) : ', Load_file4.score(Xtrain, Ytrain))
print('\n')
# -----------------------------------------------------------------------------------------------------

count = Load_file1.predict(Xtest)

# Print the Prediction of load file (Logistic Regression)
print("prediction of Logistic Regression (Load file)", "\n", Load_file1.predict(Xtest))
print('\n')

# Print the Prediction of load file (Decision Tree)
print("prediction of Decision Tree (Load file)", "\n", Load_file2.predict(Xtest))
print('\n')

# Print the Prediction of load file (SVM)
print("prediction of SVM (Load file)", "\n", Load_file3.predict(Xtest))
print('\n')

# Print the Prediction of load file (Random forest Classifier)
print("prediction of Random forest Classifier (Load file)", "\n", Load_file4.predict(Xtest))
print('\n')
# -----------------------------------------------------------------------------------------------------
# Voting the prediction
counter1 = 0
counter0 = 0
for i in range(len(count)):

    if Load_file1.predict(Xtest)[i] == 1:
        counter1 = counter1 + 1
    else:
        counter0 = counter0 + 1

    if Load_file2.predict(Xtest)[i] == 1:
        counter1 = counter1 + 1
    else:
        counter0 = counter0 + 1

    if Load_file3.predict(Xtest)[i] == 1:
        counter1 = counter1 + 1
    else:
        counter0 = counter0 + 1

    if counter1 > counter0:
        print("patient num ", i+1, " will have tumor")
    else:
        print("patient num ", i+1, " will not have tumor")
    counter1 = 0
    counter0 = 0

# -----------------------------------------------------------------------------------------------------
# Data Scaling for load data set
# initialize normalizer
data_norm = Normalizer()

# Fit the data
# Normalization formula(Z) = (X - Mean)/ Variance
Normalize = data_norm.fit_transform(XL.values)

# Distribution plot
# and we pur with the normalization the standardization by put [kde = True]
sns.displot(Normalize[:, 5], fill=True, color='red', kde=True)

# Add the axis labels
plt.xlabel('patient data')
plt.ylabel('diagnosis')

# Display the plot
plt.show()
