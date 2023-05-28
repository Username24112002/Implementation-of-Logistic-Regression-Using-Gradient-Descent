# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:  Suriya Prakash B
RegisterNumber:  212220220048
```
```py3
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data = np.loadtxt("/content/ex2data1 (2).txt", delimiter=',')
x = data[:, [0,1]]
y = data[:, 2]
```
```py3
print("Array value of x:")
x[:5]
```
```py3
print("Array value of y:")
y[:5]
```
```py3
print("Exam 1-score graph:")
plt.figure()
plt.scatter(x[y == 1][:,0], x[y == 1][:, 1], label="Admitted")
plt.scatter(x[y == 0][:,0], x[y == 0][:, 1], label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
```
```py3
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
```py3
print("Sigmoid function graph: ")
plt.plot()
x_plot = np.linspace(-10, 10, 100)
plt.plot(x_plot, sigmoid(x_plot))
plt.show()
```
```py3
def costFunction(theta, x, y):
    h = sigmoid(np.dot(x, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / x.shape[0]
    grad = np.dot(x.T, h - y) / x.shape[0]
    return J, grad 
```
```py3
x_train = np.hstack((np.ones((x.shape[0], 1)), x))
theta = np.array([0, 0, 0])
J,grad = costFunction(theta, x_train, y) 
print("x_train_grad value:")
print(J)
print(grad)
```
```py3
x_train  =  np.hstack((np.ones((x.shape[0], 1)), x)) 
theta = np.array([-24, 0.2, 0.2])
J, grad = costFunction(theta, x_train, y)
print("y_train_grad value:")
print(J)
print(grad)
```
```py3
def cost(theta, x, y):
    h = sigmoid(np.dot(x, theta))
    J = - (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / x.shape[0]
    return J
```
```py3
def gradient(theta, x, y):
    h = sigmoid(np.dot(x, theta))
    grad = np.dot(x.T, h - y) / x.shape[0]
    return grad
```
```py3
x_train = np.hstack((np.ones((x.shape[0], 1)), x))
theta = np.array([0, 0, 0])
res = optimize.minimize(fun=cost, x0=theta, args=(x_train, y),
                        method='Newton-CG', jac=gradient)
print("res.x:")
print(res.fun)
print(res.x)
```
```py3
def plotDecisionBoundary(theta, x, y):
    x_min, x_max = x[:, 0].min() -1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() -1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    x_plot = np.c_[xx.ravel(), yy.ravel()]
    x_plot = np.hstack((np.ones((x_plot.shape[0], 1)), x_plot))
    y_plot = np.dot(x_plot, theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], label="Admitted")
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], label="Not  Admitted")
    plt.contour(xx, yy, y_plot, levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
```
```py3
print("Descision Boundary - graph for exam score:")
plotDecisionBoundary(res.x, x,y)
```
```py3
print("probability value:")
prob = sigmoid(np.dot(np.array([1, 45, 85]), res.x))
print(prob)
```
```py3
def predict(theta, x):
  x_train = np.hstack((np.ones((x.shape[0], 1)), x))
  prob = sigmoid(np.dot(x_train, theta))
  return (prob >= 0.5).astype(int)
```
```py3
print("Prediction value of mean:")
np.mean(predict(res.x, x)  == y)
```

## Output:
1.Array value of X:<br>
![image](https://user-images.githubusercontent.com/128135616/233592268-982cf456-c9f1-4b41-9884-068699afcef0.png)<br>
2.Array value of Y:<br>
![image](https://user-images.githubusercontent.com/128135616/233593128-1aacd6e3-fbdb-4ce3-ad6e-7a7998e99ba0.png)
3.Exam-1 score graph:<br>
<img src="https://user-images.githubusercontent.com/128135616/233594334-7367732a-195c-4a82-8f71-d715f6444381.png"> < alt=alt text" width="150" height="150"><br>
4.Sigmoid function graph:
![image](https://user-images.githubusercontent.com/128135616/233594908-9e38f4fb-b5f6-4186-a707-db305ea9b7cd.png)
![image](https://user-images.githubusercontent.com/128135616/233595279-028133da-c7e0-4590-872c-66b99647af6a.png)
![image](https://user-images.githubusercontent.com/128135616/233595489-cded251d-6e60-42fb-a317-3fc57aee9e8c.png)
![image](https://user-images.githubusercontent.com/128135616/233596071-5d0e5464-0ea5-43f8-b64f-572f3499d5db.png)
![image](https://user-images.githubusercontent.com/128135616/233596346-5a174fce-f981-47ca-a5d8-44fca7c1a32b.png)
![image](https://user-images.githubusercontent.com/128135616/233596520-b5ce8969-5263-4b85-b6f7-92b61f1209db.png)
![image](https://user-images.githubusercontent.com/128135616/233596664-41e82d47-d99a-4818-994a-346a5b7d095d.png)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
