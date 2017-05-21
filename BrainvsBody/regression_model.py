import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model



df = pd.read_csv('Animals.csv')
print df.head()

x_val = df['Brain_Weight']
y_val = df['Body_Weight']

x_val = x_val.reshape(len(x_val), 1)
y_val = y_val.reshape(len(y_val), 1)

print x_val, y_val
print "DataFrame loaded"
reg = linear_model.LinearRegression()
print "REg"
print len(x_val), len(y_val)
reg.fit(x_val, y_val)

print "Linear Regression fit"
plt.scatter(x_val, y_val)
plt.plot(x_val, reg.predict(x_val))
plt.show()
