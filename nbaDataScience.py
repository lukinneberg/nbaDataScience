# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

from sklearn import metrics


NBAdata = pd.read_csv(r"C:\Users\lkinneberg\Desktop\Pers\dataNBAplayers.csv")
ROOKdata = pd.read_csv(r"C:\Users\lkinneberg\Desktop\Pers\dataROOKplayers.csv")

playerTrueShootNBA = NBAdata[["Name", "TS%"]]

playerTrueShootROOK = ROOKdata[["Name", "TS%"]]
playerTruePERROOK = ROOKdata[["Name", "PER"]]

# Values loaded

x = playerTrueShootROOK["TS%"]

y = playerTrueShootNBA["TS%"]

x2 = playerTruePERROOK["PER"]

# Graph below shows data for NBA players drafted in the first round in the last 10 years. The first graph below
# examines those players college data in TS% and PER to see if there's any correlation.
plt.plot(x2, x, 'ro')
plt.ylabel('College TS%')
plt.xlabel('College PER')
z = np.polyfit(x2, x, 1)
p = np.poly1d(z)
plt.plot(x2,p(x2),"r--")
plt.show()
PERaccuracy = metrics.r2_score(x2,x)
print ("PER R2:", PERaccuracy)


# Plot below compares players drafted in last 10 years and the comparison between NBA TS% and College TS%
plt.plot(x, y, 'ro')
plt.ylabel('NBA TS%')
plt.xlabel('College TS%')
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.show()

# The data is trained below

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

plt.plot(xtest, ytest, 'ro')
plt.ylabel('NBA TS%')
plt.xlabel('College TS%')

z = np.polyfit(xtest, ytest, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.show()


# The Training data is plotted and fitted to model
plt.plot(xtrain, ytrain, 'ro')
plt.ylabel('NBA TS%')
plt.xlabel('College TS%')

z = np.polyfit(xtest, ytest, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.show()

lm = linear_model.LinearRegression()
xtrain= xtrain.values.reshape(-1, 1)
ytrain= ytrain.values.reshape(-1, 1)
xtest = xtest.values.reshape(-1, 1)
model = lm.fit(xtrain, ytrain)

predictions = lm.predict(xtest)
plt.scatter(ytest, predictions)
plt.ylabel('Predicted Value')
plt.xlabel('Actual Value')

print ("Score1:", model.score(xtest, ytest))
accuracy = metrics.r2_score(ytest,predictions)
print ("Cross-Predicted Accuracy:", accuracy)
