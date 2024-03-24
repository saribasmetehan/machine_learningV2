import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv("data.csv")

data.drop(["id","Unnamed: 32"],axis = 1,inplace = True)

data.diagnosis = [ 1 if each == "M"  else 0 for each in data.diagnosis]

y = data.diagnosis.values

x_data = data.drop(["diagnosis"],axis = 1)

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

nb = GaussianNB()

nb.fit(x_train, y_train)

print(nb.score(x_test, y_test))









