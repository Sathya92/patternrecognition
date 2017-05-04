import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
mean1 = [0, 2,3,6,1.3,-5,-1.5,1.8,1,5.3]
cov1= np.random.normal(5.0,1.0,(10,10))

mean2 = [1.2,5,3,2,7.6,8,4.3,3.87,5.8,2.08]
cov2= np.random.normal(0,5.87,(10,10))
#print(cov)
rndArray1 = pd.DataFrame(np.random.multivariate_normal(mean1, cov1, 1000))
a= pd.DataFrame(np.zeros(1000))
df1 = pd.concat([rndArray1,a], axis=1)
rndArray2 = pd.DataFrame(np.random.multivariate_normal(mean2, cov2, 1000))
b= pd.DataFrame(np.ones(1000))
df2 = pd.concat([rndArray2,b], axis=1)
df = pd.concat([df1,df2])
df = df.sample(frac=1)
df = df.drop(df.index[0])
df.to_csv('randomData.csv',sep=',')	

