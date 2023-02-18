# Project 2: Gramfort's Lowess

Gramfort's approach to Lowess uses a fraction of the input data to determine the neighborhood for the locally weighted regression to use in calculating the local linear regression. This method also eliminates outliers that may be skewing the local linear regressions for a robust approach.

## Gramfort's Approach with Train and Test Sets

Import statements needed for this function:

```Python
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d, LinearNDInterpolator, NearestNDInterpolator
from sklearn.decomposition import PCA
```

A multidimensional version of Gramfort's approach to Lowess is defined below along with a function to compute the Euclidean distance between points, which is necessary for n-dimensional data.

```Python
def dist(u,v):
  if len(v.shape)==1:
    v = v.reshape(1,-1)
  d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))])
  return d
  
  
def lw_ag_md(x, y, xnew, f=2/3, iter=3, intercept=True):

  n = len(x)
  r = int(ceil(f * n))
  yest = np.zeros(n)

  if len(y.shape)==1:
    y = y.reshape(-1,1)

  if len(x.shape)==1:
    x = x.reshape(-1,1)
  
  if intercept:
    x1 = np.column_stack([np.ones((len(x),1)),x])
  else:
    x1 = x

  # we compute the max bounds for the local neighborhoods
  h = [np.sort(np.sqrt(np.sum((x-x[i])**2,axis=1)))[r] for i in range(n)]

  w = np.clip(dist(x,x) / h, 0.0, 1.0)
  w = (1 - w ** 3) ** 3

  #Looping through all X-points
  delta = np.ones(n)
  for iteration in range(iter):
    for i in range(n):
      W = np.diag(w[:,i])
      b = np.transpose(x1).dot(W).dot(y)
      A = np.transpose(x1).dot(W).dot(x1)
      ##
      A = A + 0.0001*np.eye(x1.shape[1]) # if we want L2 regularization
      beta = linalg.solve(A, b)
      yest[i] = np.dot(x1[i],beta)

    residuals = y - yest
    s = np.median(np.abs(residuals))
    delta = np.clip(residuals / (6.0 * s), -1, 1)
    delta = (1 - delta ** 2) ** 2

  if x.shape[1]==1:
    f = interp1d(x.flatten(),yest,fill_value='extrapolate')
    output = f(xnew)
  else:
    output = np.zeros(len(xnew))
    for i in range(len(xnew)):
      ind = np.argsort(np.sqrt(np.sum((x-xnew[i])**2,axis=1)))[:r]
      pca = PCA(n_components=3)
      x_pca = pca.fit_transform(x[ind])
      tri = Delaunay(x_pca,qhull_options='QJ')
      f = LinearNDInterpolator(tri,y[ind])
      output[i] = f(pca.transform(xnew[i].reshape(1,-1))) # the output may have NaN's where the data points 
                                                          # from xnew are outside the convex hull of X
  if sum(np.isnan(output))>0:
    # to prevent extrapolation
    g = NearestNDInterpolator(x,y.ravel())
    output[np.isnan(output)] = g(xnew[np.isnan(output)])
  return output
```
## KFold Cross-Validations

To test the above function, we'll run some KFold cross-validations with real data.

Import statements:

```Python
from sklearn.model_selection import train_test_split as tts, KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
```

### Car data

```Python
data = pd.read_csv('drive/MyDrive/DATA441/data/cars.csv')

x = data.loc[:,'CYL':'WGT'].values
y = data['MPG'].values

mse_lwr = []
mse_rf = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)
# model_lw = lw_ag_md(f=1/3,iter=1,intercept=True) #(x, y, xnew, f=2/3, iter=3, intercept=True)

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  yhat_lw = lw_ag_md(xtrain, ytrain, xtest, f=1/3,iter=1,intercept=True)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```

### Concrete data
