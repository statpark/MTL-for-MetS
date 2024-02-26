# MTL-for-MetS
Multi-task learning appraoch for prediction of metabolic syndrome(MetS) 

# Example 
create data with 5 classes 
```
x, y = make_multilabel_classification(n_samples = 1000, n_features = 40, n_classes = 5, n_labels = 2, random_state = 0)
```
split the data
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
```
create the 6th class which takes value of 1 if sum of other 5 classes is equal to or greater than 3 and takes value of 0 otherwise
```
ys_train = list(y_train.T)
ys_train.append(np.where(np.sum(y_train, axis = 1) >= 3 , 1, 0))
ys_test = list(y_test.T)
ys_test.append(np.where(np.sum(y_test, axis = 1) >= 3 , 1, 0))
```
