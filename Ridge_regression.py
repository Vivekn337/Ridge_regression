import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

df = pd.read_csv('BWGHT.csv')
x = df[['cigs','faminc','male','white']].values
y = df['bwght'].values
ones = np.ones((x.shape[0],1))
x = np.hstack((ones,x))
penalty = [0,0.05,0.1,0.5,0.75,1]
for pen in penalty:
    iden = pen*np.identity(np.shape(x)[1])
    beta = np.linalg.inv(np.add(x.T@x,iden))@x.T@y
    print("Lambda : ",pen)
    print("Optimized value Calculated:",beta)
    ridge = Ridge(alpha=pen)
    ridge.fit(x, y)
    print("Value calculated from package:",ridge.coef_)
