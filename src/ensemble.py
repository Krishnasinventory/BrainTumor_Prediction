# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


# from sklearn.linear_model import LogisticRegression]
from sklearn.feature_selection import RFE


from preprocessing import data,X_train_fs,X_train,X_test_fs,X_test,y_Test,y_train
# %% [markdown]
# <h1>ENSEMBLE LEARNING

# %%
from sklearn.ensemble import StackingClassifier
# Define base models Level 0
base_models = [
    ('svm', SVC(C = 10, gamma=0.01,kernel='linear', random_state=56)),
    ('rf', RandomForestClassifier(n_estimators=50,min_samples_split=10,max_depth=20, random_state=56)),
    ('adaboost', AdaBoostClassifier(n_estimators=50,learning_rate=0.1, random_state=56))
]
# base_models = [
#     ('svm', SVC()),
#     ('rf', RandomForestClassifier()),
#     ('adaboost', AdaBoostClassifier())
# ]
# Define the meta-model Level 1
meta_model = LogisticRegression(
    C = 1,
    solver='liblinear'
)


# Initialize the Stacking ensemble
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10,)

# Train the ensemble model
stacking_model.fit(X_train_fs, y_train)

# Make predictions
y_pred = stacking_model.predict(X_test_fs)

# Evaluate performance
# print("Accuracy of Stacking Model:", accuracy_score(y_Test, y_pred))
print(classification_report(y_pred,y_Test))
accuracy_score(y_pred,y_Test)

# %%
from sklearn.ensemble import StackingClassifier
# Define base models Level 0
base_models = [
    ('svm', SVC(C = 10, gamma=0.01,kernel='linear', random_state=56)),
    ('rf', RandomForestClassifier(n_estimators=50,min_samples_split=10,max_depth=20, random_state=56)),
    ('adaboost', AdaBoostClassifier(n_estimators=50,learning_rate=0.1, random_state=56))
]
# base_models = [
#     ('svm', SVC()),
#     ('rf', RandomForestClassifier()),
#     ('adaboost', AdaBoostClassifier())
# ]
# Define the meta-model Level 1
meta_model = LogisticRegression(
    C = 1,
    solver='liblinear'
)


# Initialize the Stacking ensemble
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10,)

# Train the ensemble model
stacking_model.fit(X_train, y_train)

# Make predictions
y_pred = stacking_model.predict(X_test)

# Evaluate performance
# print("Accuracy of Stacking Model:", accuracy_score(y_Test, y_pred))
print(classification_report(y_pred,y_Test))
accuracy_score(y_pred,y_Test)

# %%
from sklearn.ensemble import StackingClassifier
# Define base models Level 0
# base_models = [
#     ('svm', SVC(C = 10, gamma=0.01,kernel='linear', random_state=56)),
#     ('rf', RandomForestClassifier(n_estimators=50,min_samples_split=10,max_depth=20, random_state=56)),
#     ('adaboost', AdaBoostClassifier(n_estimators=50,learning_rate=0.1, random_state=56))
# ]
base_models = [
    ('svm', SVC(random_state=56)),
    ('rf', RandomForestClassifier(random_state=56)),
    ('adaboost', AdaBoostClassifier(random_state=56))
]
# Define the meta-model Level 1
meta_model = LogisticRegression(
    C = 1,
    solver='liblinear'
)


# Initialize the Stacking ensemble
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10,)

# Train the ensemble model
stacking_model.fit(X_train_fs, y_train)

# Make predictions
y_pred = stacking_model.predict(X_test_fs)

# Evaluate performance
# print("Accuracy of Stacking Model:", accuracy_score(y_Test, y_pred))
print(classification_report(y_pred,y_Test))
accuracy_score(y_pred,y_Test)

# %%
from sklearn.ensemble import StackingClassifier
# Define base models Level 0
# base_models = [
#     ('svm', SVC(C = 10, gamma=0.01,kernel='linear', random_state=56)),
#     ('rf', RandomForestClassifier(n_estimators=50,min_samples_split=10,max_depth=20, random_state=56)),
#     ('adaboost', AdaBoostClassifier(n_estimators=50,learning_rate=0.1, random_state=56))
# ]
base_models = [
    ('svm', SVC(random_state=56)),
    ('rf', RandomForestClassifier(random_state=56)),
    ('adaboost', AdaBoostClassifier(random_state=56))
]
# Define the meta-model Level 1
meta_model = LogisticRegression(
    C = 1,
    solver='liblinear'
)


# Initialize the Stacking ensemble
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10,)

# Train the ensemble model
stacking_model.fit(X_train, y_train)

# Make predictions
y_pred = stacking_model.predict(X_test)

# Evaluate performance
# print("Accuracy of Stacking Model:", accuracy_score(y_Test, y_pred))
print(classification_report(y_pred,y_Test))
accuracy_score(y_pred,y_Test)

# %% [markdown]
# Without feature selection & not hyper-tuned : 0.7916666666666666

# %% [markdown]
# With feature selection but not hyper-tuned : 0.8035714285714286

# %% [markdown]
# Without feature selection and hyper-tuned :  0.8154761904761905

# %% [markdown]
# With feature selection and hyper-tuned : 0.8273809523809523

# %%
print(classification_report(y_pred,y_Test))

# %% [markdown]
# TAKING FOUR MODELS

# %%
from sklearn.ensemble import StackingClassifier
# Define base models Level 0
base_models = [
    ('svm', SVC(C = 10, gamma=0.01,kernel='linear', random_state=56)),
    ('rf', RandomForestClassifier(n_estimators=50,min_samples_split=10,max_depth=20, random_state=56)),
    ('adaboost', AdaBoostClassifier(n_estimators=50,learning_rate=0.1, random_state=56)),
    ('knn',KNeighborsClassifier(n_neighbors= 3, weights =  'distance'))
]
# base_models = [
#     ('svm', SVC()),
#     ('rf', RandomForestClassifier()),
#     ('adaboost', AdaBoostClassifier()),
#     ('knn',KNeighborsClassifier())
# ]
# Define the meta-model Level 1
meta_model = LogisticRegression(
    C = 1,
    solver='liblinear'
)


# Initialize the Stacking ensemble
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10)

# Train the ensemble model
stacking_model.fit(X_train, y_train)

# Make predictions
y_pred = stacking_model.predict(X_test)

# Evaluate performance
# print("Accuracy of Stacking Model:", accuracy_score(y_Test, y_pred))
print(classification_report(y_pred,y_Test))
accuracy_score(y_pred,y_Test)

# %%
from sklearn.ensemble import StackingClassifier
# Define base models Level 0
# base_models = [
#     ('svm', SVC(C = 10, gamma=0.01,kernel='linear', random_state=56)),
#     ('rf', RandomForestClassifier(n_estimators=50,min_samples_split=10,max_depth=20, random_state=56)),
#     ('adaboost', AdaBoostClassifier(n_estimators=50,learning_rate=0.1, random_state=56)),
#     ('knn',KNeighborsClassifier(n_neighbors= 3, weights =  'distance'))
# ]
base_models = [
    ('svm', SVC(random_state=56)),
    ('rf', RandomForestClassifier(random_state=56)),
    ('adaboost', AdaBoostClassifier(random_state=56)),
    ('knn',KNeighborsClassifier(random_state=56))
]
# Define the meta-model Level 1
meta_model = LogisticRegression(
    C = 1,
    solver='liblinear'
)


# Initialize the Stacking ensemble
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10)

# Train the ensemble model
stacking_model.fit(X_train, y_train)

# Make predictions
y_pred = stacking_model.predict(X_test)

# Evaluate performance
# print("Accuracy of Stacking Model:", accuracy_score(y_Test, y_pred))
print(classification_report(y_pred,y_Test))
accuracy_score(y_pred,y_Test)

# %%
from sklearn.ensemble import StackingClassifier
# Define base models Level 0
base_models = [
    ('svm', SVC(C = 10, gamma=0.01,kernel='linear', random_state=56)),
    ('rf', RandomForestClassifier(n_estimators=50,min_samples_split=10,max_depth=20, random_state=56)),
    ('adaboost', AdaBoostClassifier(n_estimators=50,learning_rate=0.1, random_state=56)),
    ('knn',KNeighborsClassifier(n_neighbors= 3, weights =  'distance'))
]
# base_models = [
#     ('svm', SVC()),
#     ('rf', RandomForestClassifier()),
#     ('adaboost', AdaBoostClassifier()),
#     ('knn',KNeighborsClassifier())
# ]
# Define the meta-model Level 1
meta_model = LogisticRegression(
    C = 1,
    solver='liblinear'
)


# Initialize the Stacking ensemble
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10)

# Train the ensemble model
stacking_model.fit(X_train_fs, y_train)

# Make predictions
y_pred = stacking_model.predict(X_test_fs)

# Evaluate performance
# print("Accuracy of Stacking Model:", accuracy_score(y_Test, y_pred))
print(classification_report(y_pred,y_Test))
accuracy_score(y_pred,y_Test)

# %%
from sklearn.ensemble import StackingClassifier
# Define base models Level 0
# base_models = [
#     ('svm', SVC(C = 10, gamma=0.01,kernel='linear', random_state=56)),
#     ('rf', RandomForestClassifier(n_estimators=50,min_samples_split=10,max_depth=20, random_state=56)),
#     ('adaboost', AdaBoostClassifier(n_estimators=50,learning_rate=0.1, random_state=56)),
#     ('knn',KNeighborsClassifier(n_neighbors= 3, weights =  'distance'))
# ]
base_models = [
    ('svm', SVC(random_state=56)),
    ('rf', RandomForestClassifier(random_state=56)),
    ('adaboost', AdaBoostClassifier(random_state=56)),
    ('knn',KNeighborsClassifier(random_state=56))
]
# Define the meta-model Level 1
meta_model = LogisticRegression(
    C = 1,
    solver='liblinear'
)


# Initialize the Stacking ensemble
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10)

# Train the ensemble model
stacking_model.fit(X_train_fs, y_train)

# Make predictions
y_pred = stacking_model.predict(X_test_fs)

# Evaluate performance
# print("Accuracy of Stacking Model:", accuracy_score(y_Test, y_pred))
print(classification_report(y_pred,y_Test))
accuracy_score(y_pred,y_Test)

# %%



