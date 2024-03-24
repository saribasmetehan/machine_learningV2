import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("random_forest_regression_dataset.csv",sep=";",header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf.fit(x, y)

predict_input1 = np.array(0.5).reshape(1,-1)

prediction = rf.predict(predict_input1)

print(prediction)


x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = rf.predict(x_)

plt.scatter(x, y,color = "red")
plt.plot(x_,y_head,color="blue")
plt.show()

y_head1 = rf.predict(x)

print(r2_score(y, y_head1))














