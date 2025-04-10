import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
data_set=pd.read_csv("Salary Data.csv")
#print(data_set.to_string())

x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,1].values
#print(x)
#print(y)

# Split of Dataset into training AND testing 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size =1/3,random_state=0)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

# fit() Method
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_train)

#prediction
y_pred =regressor.predict(x_test)
x_pred =regressor.predict(x_train)
print(y_pred)
print(x_pred)

mtp.scatter(x_train, y_train, color="green")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Salary vs Experience (Training Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show() 