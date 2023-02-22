# Project 2: Gramfort's Lowess

Gramfort's approach to Lowess uses a fraction of the input data to determine the neighborhood for the locally weighted regression to use in calculating the local linear regression. This method also eliminates outliers that may be skewing the local linear regressions for a robust approach. The only downside to his original approach is that it does not allow multidimensional input data. Thus, we rework his original function below so we can use it with data that has multiple features.

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
      W = np.diag(delta).dot(np.diag(w[i,:]))
      b = np.transpose(x1).dot(W).dot(y)
      A = np.transpose(x1).dot(W).dot(x1)
      ##
      A = A + 0.0001*np.eye(x1.shape[1]) # if we want L2 regularization
      beta = linalg.solve(A, b)
      yest[i] = np.dot(x1[i],beta)

    residuals = y.ravel() - yest
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
      pca = PCA(n_components=2)
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

To test the above function, we'll run some KFold cross-validations with real data. With each of the two datasets below, we compute the mse of a random forest regressor to compare with the mse for Lowess.

Import statements:

```Python
from sklearn.model_selection import train_test_split as tts, KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
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
scale = StandardScaler()

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  yhat_lw = lw_ag_md(xtrain, ytrain, xtest, f=1/6, iter=2, intercept=True)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```
Running this code yields the approximate results:

The Cross-validated Mean Squared Error for Locally Weighted Regression is : 21.731641567699477 \
The Cross-validated Mean Squared Error for Random Forest is : 17.113401560521112

Thus, the locally weighted regression has a slightly higher mse than random forest, but is overall close. The mse can be improved by fine-tuning the hyperparameters (f and iter), which we do in the Optimizing Hyperparameters section below by using a grid search.

### Concrete data

```Python
data2 = pd.read_csv('drive/MyDrive/DATA441/data/concrete.csv')

x = data2.loc[:,'cement':'age'].values
y = data2['strength'].values

mse_lwr = []
mse_rf = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)
scale = StandardScaler()

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  yhat_lw = lw_ag_md(xtrain, ytrain, xtest, f=1/75, iter=2, intercept=True)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```

Running this code yields the approximate results:

The Cross-validated Mean Squared Error for Locally Weighted Regression is : 45.948580062873056 \
The Cross-validated Mean Squared Error for Random Forest is : 45.4781508788599

Here, the mses are higher than with the previous dataset, but the locally weighted regression method is much closer in mse to the random forest regressor.

## A Scikit Learn Compliant Function

Next we will make a Lowess class that makes the function Scikit learn compliant. It is nice to be able to use the fit and predict methods.

Additional import needed for this function:

```Python
from sklearn.utils.validation import check_is_fitted
```

Class definition:

```Python
class Lowess_AG_MD:
    def __init__(self, f = 1/10, iter = 3, intercept=True):
        self.f = f
        self.iter = iter
        self.intercept = intercept
    
    def fit(self, x, y):
        f = self.f
        iter = self.iter
        self.xtrain_ = x
        self.yhat_ = y

    def predict(self, x_new):
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_
        f = self.f
        iter = self.iter
        intercept = self.intercept
        return lw_ag_md(x, y, x_new, f, iter, intercept)

    def get_params(self, deep=True):
        return {"f": self.f, "iter": self.iter,"intercept":self.intercept}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
```

A KFold cross-validation can be run with this class. The code below is very similar to the cross-validations above with the non-Scikit compliant function.

```Python
mse_lwr = []
mse_rf = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)
model_lw = Lowess_AG_MD(f=1/75,iter=2,intercept=True)

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  model_lw.fit(xtrain,ytrain)
  yhat_lw = model_lw.predict(xtest)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```
## Optimizing Hyperparameters

To get smaller mse values, we can run a grid search for optimized hyperparameters.

Import statements:

```Python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
```

### Car data

```Python
x = data.loc[:,'CYL':'WGT'].values
y = data['MPG'].values

lwr_pipe = Pipeline([('zscores', StandardScaler()),
                     ('lwr', Lowess_AG_MD())])

params = [{'lwr__f': [1/i for i in range(3,15)],
         'lwr__iter': [1,2,3,4]}]
         
gs_lowess = GridSearchCV(lwr_pipe,
                      param_grid=params,
                      scoring='neg_mean_squared_error',
                      cv=5)
gs_lowess.fit(x, y)
gs_lowess.best_params_
```

From this grid search, we get that the best value for f is 1/3 and the best value for iter is 1. When we plug these values into the Scikit compliant function and run a cross-validation, we get an mse of 23.536855642155476, which is slightly better than the previous mse for this data.

Unfortunately, because the grid search method is not highly efficient, it is impractical to use on the concrete dataset due to time constraints.
