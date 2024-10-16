import pandas as pd
data = pd.read_csv('D:\spyder\Machine\heart-data.csv')

mask = data.isnull().any(axis=1)
data_clean = data[~mask]

x=data_clean.iloc[:,:-1].values
y=data_clean.iloc[:,-1]


import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler 

lb=LabelEncoder()
x[:,1]=lb.fit_transform(x[:,1])
x[:,5]=lb.fit_transform(x[:,5])
x[:,6]=lb.fit_transform(x[:,6])
x[:,8]=lb.fit_transform(x[:,8])

from sklearn.compose import ColumnTransformer

onehotencoder=ColumnTransformer([('encoder',OneHotEncoder(),[2,10,12])],remainder="passthrough")

x=np.array(onehotencoder.fit_transform(x))

sc=MinMaxScaler()
x[:,10:]=sc.fit_transform(x[:,10:])

#we can find data impalance
print(data['output'].value_counts())

from sklearn.model_selection import train_test_split
X, x_test, Y, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.30, random_state=0)
print(y_train.value_counts())
y_train.value_counts().sort_index().plot.bar()

from imblearn.over_sampling import RandomOverSampler
rus = RandomOverSampler(sampling_strategy=1.0, random_state=0)
X_train_balanced, y_train_balanced = rus.fit_resample(x_train, y_train)
print(y_train_balanced.value_counts())


from sklearn.metrics import accuracy_score
from sklearn.svm import SVC #Classification #------------- For Regression Make it SVR ------------------#
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
classifier = SVC(kernel='linear', random_state = 0,C=.001)
classifier.fit(X_train_balanced, y_train_balanced)
y_pred_train = classifier.predict(X_train_balanced)

y_pred_val = classifier.predict(x_val)
acc_train = accuracy_score(y_train_balanced, y_pred_train)
acc_val = accuracy_score(y_val, y_pred_val)
print(acc_train)
print(acc_val)

y_pred_test = classifier.predict(x_test)
print(accuracy_score(y_test, y_pred_test))
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
classifier = SVC(kernel='poly', degree=3,random_state = 0,C=.01)
classifier.fit(X_train_balanced, y_train_balanced)
y_pred_train = classifier.predict(X_train_balanced)

y_pred_val = classifier.predict(x_val)
acc_train = accuracy_score(y_train_balanced, y_pred_train)
acc_val = accuracy_score(y_val, y_pred_val)
print(acc_train)
print(acc_val)

y_pred_test = classifier.predict(x_test)
print(accuracy_score(y_test, y_pred_test))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
classifier = SVC(kernel='sigmoid', gamma=.02,random_state = 0,C=500)
classifier.fit(X_train_balanced, y_train_balanced)
y_pred_train = classifier.predict(X_train_balanced)
y_pred_val = classifier.predict(x_val)

acc_train = accuracy_score(y_train_balanced, y_pred_train)
acc_val = accuracy_score(y_val, y_pred_val)

print(acc_train)
print(acc_val)

y_pred_test = classifier.predict(x_test)
print(accuracy_score(y_test, y_pred_test))
--------------------------------------------------------------------
 classifier = SVC(kernel='rbf', C = 500)
classifier.fit(X_train_balanced, y_train_balanced)
y_pred_train = classifier.predict(X_train_balanced)
y_pred_val = classifier.predict(x_val)

acc_train = accuracy_score(y_train_balanced, y_pred_train)
acc_val = accuracy_score(y_val, y_pred_val)

print(acc_train)
print(acc_val)

y_pred_test = classifier.predict(x_test)
print(accuracy_score(y_test, y_pred_test))
