import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv("polynomial_linear_regression.csv")

y = df.car_speed.values.reshape(-1,1)
x = df.car_price.values.reshape(-1,1)

linear_reg = LinearRegression()
linear_reg.fit(x,y)

plt.scatter(x, y)
plt.ylabel("car_speed")
plt.xlabel("car_price")
y_head1 = linear_reg.predict(x)
plt.plot(x,y_head1,color ="red",label="linear")


polynomial_regression = PolynomialFeatures(degree=4)

x_polynomial = polynomial_regression.fit_transform(x)

linear_reg.fit(x_polynomial,y)

y_head2 = linear_reg.predict(x_polynomial)

plt.plot(x,y_head2,color="blue",label = "polynomial")
plt.legend()
plt.show()





 




