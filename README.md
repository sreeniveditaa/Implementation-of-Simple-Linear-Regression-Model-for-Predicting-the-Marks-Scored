# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import the standard Libraries.   
2.  Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
DEVOLOPED BY : SREE NIVEDITAA SARAVANAN
REGISTER NUMBER : 212223230213
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_

```
## Output:

1. HEAD:

![image](https://github.com/sreeniveditaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473268/da260614-2786-47a1-83c6-26c550169143)

2.GRAPH OF PLOTTED DATA:

![image](https://github.com/sreeniveditaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473268/ab589ae7-6b8d-4efa-91f5-3b797f22753a)

3.TRAINED DATA:

![image](https://github.com/sreeniveditaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473268/3995e175-5db6-43c5-84e1-8045cd275734)

4.LINE OF REGRESSION:

![image](https://github.com/sreeniveditaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473268/b0689c5b-ca32-46b3-b2ba-dce7ee6163f0)

5.COEFFICIENT AND INTERCEPT VALUES:

![image](https://github.com/sreeniveditaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473268/2ccac396-e05d-431b-9f83-7045e377fe78)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
