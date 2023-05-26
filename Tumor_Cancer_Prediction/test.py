import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load players data
Tumor = pd.read_csv("Test Data.csv")

# drop all null values
Tumor = Tumor.dropna()

# drop all duplicate rows
Tumor = Tumor.drop_duplicates()

# Split the dataset into independent(X) and dependent(Y) datasets
X = Tumor.iloc[:, 1:31]
Y = Tumor['diagnosis']

# Split the dataset into 70% training and 30% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)


Save_file1 = 'Tumor_Cancer_Prediction by logistic regression.sav'
Save_file2 = 'Tumor_Cancer_Prediction by Decision Tree.sav'
Save_file3 = 'Tumor_Cancer_Prediction by svm.sav'

# Load from file (Logistic Regression)
Load_file1 = joblib.load(Save_file1)

# Load from file (Decision Tree)
Load_file2 = joblib.load(Save_file2)

# Load from file (SVM)
Load_file3 = joblib.load(Save_file3)

# Print the Accuracy (Logistic Regression)
print('[1]Logistic Regression Training Accuracy (Load file) : ', Load_file1.score(X_train, Y_train))

# Print the Accuracy(Decision Tree)
print('[2]Decision Tree Training Accuracy (Load file) : ', Load_file2.score(X_train, Y_train))

# Print the Accuracy (SVM)
print('[3]SVM Training Accuracy by linear kernel (Load file) : ', Load_file3.score(X_train, Y_train))
