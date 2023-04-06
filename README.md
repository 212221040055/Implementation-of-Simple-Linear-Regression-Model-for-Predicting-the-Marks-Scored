# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the needed packages
2.Assigning hours To X and Scores to Y
3.Plot the scatter plot
4.Use mse,rmse,mae formmula to find  

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Hemavathi.N
RegisterNumber:  212221040055
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
print("df.head")

df.head()

print("df.tail")

df.tail()

Y=df.iloc[:,1].values
print("Array of Y")
Y

X=df.iloc[:,:-1].values
print("Array of X")
X

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print("Values of Y prediction")
Y_pred

print("Array values of Y test")
Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
print("Training Set Graph")
plt.show()

plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours Vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
print("Test Set Graph")
plt.show()

print("Values of MSE, MAE and RMSE")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
## df.head():

![hema 1](https://user-images.githubusercontent.com/128135323/230450884-dcde6045-7842-4c67-9eb7-88f87dea17f3.png)

## df.tail():

![hema 2](https://user-images.githubusercontent.com/128135323/230451206-1d45009f-304c-4945-91e5-7c88db283a6b.png)

## Array of X:

![229763055-9d7a86d9-0970-4a6e-951c-e8981a121fe9](https://user-images.githubusercontent.com/128135323/230451553-128bdddc-2391-4476-b760-bdffaea9ddbe.png)

## Array of Y:

![hema 3](https://user-images.githubusercontent.com/128135323/230451919-a8836d37-10ed-46c5-9c3f-64576ea9ad9a.png)

## Y_Pred:

![hema 4](https://user-images.githubusercontent.com/128135323/230452231-8c4dae2f-e8b8-4c52-b2e9-f8863775c440.png)

## y_test:

![hema 5](https://user-images.githubusercontent.com/128135323/230452861-1edf5ea8-39ff-4007-9142-2cc922388810.png)

## Training set:

![hema 6](https://user-images.githubusercontent.com/128135323/230453130-6265157b-f72b-4e50-a5d7-90c5871e25fa.png)

## Test set:

![hema 6](https://user-images.githubusercontent.com/128135323/230453567-237babfc-8c2f-4298-bbf7-cc29fa6d9a4e.png)

## Values of MSE,MAE,RMSE:

![hema 8](https://user-images.githubusercontent.com/128135323/230453859-e8238eca-7800-4513-911f-a01d088c8df0.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
