import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#importing csv dataset

data = pd.read_csv("train.csv").values


clf = DecisionTreeClassifier()


#training the data
xtrain = data[0:21000,1:]
train_label = data[0:21000,0]



clf.fit(xtrain,train_label)

#testing the data

xtest = data[21000:, 1:]
actual_label= data[21000:,0]

p=clf.predict(xtest)

d= xtest[44]
d.shape=(28,28)
pt.imshow(255-d, cmap="turbo")
print(clf.predict([xtest[44]]))

pt.show()

#finding the accuracy

count=0
for i in range(0,21000):
    count+=1 if p[i]==actual_label[i] else 0
accuracy = (count / len(actual_label)) * 100
print("Accuracy =", accuracy)

 