import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


#import data
ds = pd.read_csv("linear_regression_dataset.csv")

df = ds.copy()

#plot data
print(df.head())

plt.scatter(df.experience,df.wage)
plt.xlabel("experience")
plt.ylabel("wage")
plt.show()


#fit to linear regression 
linear_reg = LinearRegression()

x_label = df.experience.values.reshape(-1,1)
y_label = df.wage.values.reshape(-1,1)

linear_reg.fit(x_label, y_label)

#formula values

b0 = linear_reg.predict([[0]])

print(b0)

b0 = linear_reg.intercept_

print(b0)

b1 = linear_reg.coef_

print(b1)


#predict

print(linear_reg.predict([[15]]))

#visualize line

array = np.arange(1,16,1).reshape(-1,1)

plt.scatter(x_label, y_label)

y_head = linear_reg.predict(array)

plt.plot(array,y_head,color = "red")

plt.show()

#r2_score

y_head1 = linear_reg.predict(x_label)

print(r2_score(y_label, y_head1))







