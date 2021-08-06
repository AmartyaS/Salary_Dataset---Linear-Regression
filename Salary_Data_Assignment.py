# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:32:11 2021

@author: amart
"""
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

dat=pd.read_csv("F:\Data Science Assignments\Python-Assignment\Linear Regression\Salary_Data.csv")
new=dat
new=new.rename(columns={'YearsExperience':'exp','Salary':'sal'})
new.sal.corr(new.exp)
plt.hist(new.sal)
plt.hist(new.exp)
plt.boxplot(new.sal)
plt.boxplot(new.exp)
new.sal.plot(kind="area")
plt.plot(new.exp, new.sal)

model1=smf.ols('sal~exp',data=new).fit()
model1.summary()                            #R-Square =0.957  and Adj R-Square =0.955
model1.conf_int(0.05)
pred1=model1.predict(new)
plt.scatter(x=new.exp,y=new.sal,color='red');plt.plot(new.exp,pred1,color='blue');plt.xlabel("Years Of Experience");plt.ylabel("Salary")
pred1.corr(new.sal)                         #0.9782416184887601
rmse=(mean_squared_error(new.sal, pred1))
sqrt(rmse)                                  #5592.043608760662



model2=smf.ols('sal~ np.log(exp)',data=new).fit()
model2.summary()                            #R-Square =0.854  and Adj R-Square =0.849
model2.conf_int(0.05)
pred2=model2.predict(new)
plt.scatter(x=new.exp,y=new.sal,color='red');plt.plot(new.exp,pred2,color='blue');plt.xlabel("Years Of Experience");plt.ylabel("Salary")
pred2.corr(new.sal)                         #0.9240610817882637
rmse2=(mean_squared_error(new.sal, pred2))
sqrt(rmse2)                                 #10302.893706228308



model3=smf.ols('sal~ (exp+exp)',data=new).fit()
model3.summary()                            #R-Square =0.957 and Adj R-Square =0.955
model3.conf_int(0.05)
pred3=model3.predict(new)
plt.scatter(x=new.exp,y=new.sal,color='red');plt.plot(new.exp,pred3,color='blue');plt.xlabel("Years Of Experience");plt.ylabel("Salary")
pred3.corr(new.sal)                         #0.9782416184887601
rmse3=sqrt(mean_squared_error(new.sal, pred3))
rmse3                                       #5592.043608760662


# Model 1 and Model3 both are best as  both share same characteristics