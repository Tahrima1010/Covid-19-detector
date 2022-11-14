import pandas as pd
import numpy as np



from sklearn import tree

df = pd.read_csv('COVID19final12.csv')

cols_to_drop = ['Timestamp']
df.head()
df = df.drop(cols_to_drop, axis = 1)

df['COVID'] = df['COVID'].str.replace('Positive','1')
df['COVID'] = df['COVID'].str.replace('Negative','0')


C_col_dummy = pd.get_dummies(df['Fever'])
D_col_dummy = pd.get_dummies(df['FEVERS'])
E_col_dummy = pd.get_dummies(df['Cold'])
F_col_dummy = pd.get_dummies(df['Cough'])
G_col_dummy = pd.get_dummies(df['COUGHS'])
H_col_dummy = pd.get_dummies(df['BODY ACHE'])
I_col_dummy = pd.get_dummies(df['SMELL'])
J_col_dummy = pd.get_dummies(df['HEADACHE'])

df = pd.concat((df, C_col_dummy, D_col_dummy,  E_col_dummy, F_col_dummy, G_col_dummy,  H_col_dummy, I_col_dummy, J_col_dummy), axis=1)
#print(df)


df = df.drop(['Fever', 'Cold', 'FEVERS', 'Cough', 'COUGHS', 'BODY ACHE', 'SMELL', 'HEADACHE'], axis=1)
#print(df)
#df = df.interpolate()
x = df.values
y = df['COVID'].values

x = np.delete(x,1,axis=1)


#decision tree
print('Decision Tree classifier :')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

dt_clf = tree.DecisionTreeClassifier(max_depth=10)
dt_clf.fit(x_train, y_train)

dt_clf.score(x_test, y_test)
y_pred = dt_clf.predict(x_test)
#print(y_pred)
print('Accuracy :')
#print(dt_clf.score(x_test, y_test))
percentage = "{:.0%}". format(dt_clf.score(x_test, y_test))
print(percentage)

y_pred = dt_clf.predict(x_test)

#confusion matrix decision tree
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))

#random forest
print('Random Forest classifier :')
from sklearn import ensemble
rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
rf_clf.fit(x_train, y_train)
y_pred1 = rf_clf.predict(x_test)
#print(y_pred)
print('Accuracy :')
#print(rf_clf.score(x_test, y_test))
percentage1 = "{:.0%}". format(rf_clf.score(x_test, y_test))
print(percentage1)

#confusion matrix decision tree
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred1)
print(confusion_matrix(y_test, y_pred1))





#naive bayess classifier
print('Naive Bayes classifier :')
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)
y_pred5 =nb_clf.predict(x_test)
print('Accuracy :')
#print(nb_clf.score(x_test, y_test))
percentage3 = "{:.0%}". format(nb_clf.score(x_test, y_test))
print(percentage3)

#confusion matrix decision tree
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred5)
print(confusion_matrix(y_test, y_pred5))

#support vector machine
print('SVM classifier :')
from sklearn.svm import SVC
sv_clf =SVC(probability = True, kernel = 'linear')
sv_clf.fit(x_test, y_test)
y_pred4 =sv_clf.predict(x_test)
print('Accuracy :')
#sv_clf.score(x_test, y_test)
percentage4 = "{:.0%}". format(sv_clf.score(x_test, y_test))
print(percentage4)

#confusion matrix decision tree
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred4)
print(confusion_matrix(y_test, y_pred4))



#KNN classifier
print('KNN classifier :')
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(x_train, y_train)
y_pred2 = knn_clf.predict(x_test)
print('Accuracy :')
#print(knn_clf.score(x_test, y_test))
percentage6 = "{:.0%}". format(knn_clf.score(x_test, y_test))
print(percentage6)
#confusion matrix decision tree
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred2)
print(confusion_matrix(y_test, y_pred2))


