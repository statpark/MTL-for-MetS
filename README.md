# MTL-for-MetS
Multi-task learning appraoch for prediction of metabolic syndrome(MetS) 
<img width="452" alt="arch1" src="https://github.com/statpark/MTL-for-MetS/assets/54830606/a0b35e74-8df5-4735-b313-0d049fc069e7">
# Notice 
All source codes were listed in file "main.py".
# Example 
Create data with 5 classes 
```
import pandas as pd
import numpy as np
from sklearn.datasets import make_multilabel_classification

x, y = make_multilabel_classification(n_samples = 1000, n_features = 40, n_classes = 5 random_state = 0)
```
Split the data
```
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
```
Create the 6th class which takes value of 1 if sum of other 5 classes is equal to or greater than 3 and takes value of 0 otherwise
```
ys_train = list(y_train.T)
ys_train.append(np.where(np.sum(y_train, axis = 1) >= 3 , 1, 0))
ys_test = list(y_test.T)
ys_test.append(np.where(np.sum(y_test, axis = 1) >= 3 , 1, 0))
```
Train the model. You can select weight of each class.
```
from main import *

model = Multi_MLP(n_features = x.shape[1], n_units = 20, dropout = 0.2)
loss_weight = {'out1': 1/6, 'out2': 1/6, 'out3': 1/6, 'out4': 1/6, 'out5': 1/6, 'out':1/6}
model.compile(optimizer='adam', 
              loss=['binary_crossentropy']*6, 
              loss_weights = loss_weight)
_ = model.fit(x_train, ys_train,  epochs = 600, batch_size = 32, verbose = 0)
```
Check prediction result
```
from sklearn.metrics import roc_auc_score

prob = model.predict(x_test, verbose = 0)[5]
roc_auc_score(ys_test[5], prob)
```
Compare results with other models 
```
from catboost import CatboostClassifier

cat = CatboostClassifier(random_state = 0, silent = True)
_ = model.fit(x_train, ys_train[5])
prob = model.predict_proba(x_test)[:,1]
roc_auc_score(ys_test[5], prob)
