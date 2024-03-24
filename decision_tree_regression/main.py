import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor



df = pd.read_csv("decision_tree_regression_dataset.csv",sep = ";",header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

tree_reg =DecisionTreeRegressor()
tree_reg.fit(x, y)

x_new = np.arange(1,10,0.01 ).reshape(-1,1)
y_head = tree_reg.predict(x_new)

plt.scatter(x, y, color ="red")
plt.plot(x_new,y_head,color = "blue")
plt.xlabel("level")
plt.ylabel("price")
plt.show()


