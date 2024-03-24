import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#fit to multiple linear regression
 
df = pd.read_csv("multiple_linear_regression_dataset.csv")

x_label= df.iloc[:,[0,2]].values
y_label = df.wage.values.reshape(-1,1)

multiple_ln_regression = LinearRegression()
multiple_ln_regression.fit(x_label,y_label)

#formula values

b0 = multiple_ln_regression.intercept_


print(f"b0 : {b0}")
print(f"b1,b2 : {multiple_ln_regression.coef_}")

#prediction

print(multiple_ln_regression.predict(np.array([[10,45],[5,35]])))

