# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
# %%
# df_unGraded = pd.read_csv(r'..\data\TCGA_GBM_LGG_Mutations_all.csv')
df_graded = pd.read_csv(r'..\data\TCGA_InfoWithGrade.csv')

# %%
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import accuracy_score

# %%
X = df_graded.loc[:,df_graded.columns != 'Grade']
y = df_graded['Grade']
X_train,X_test,y_train,y_Test = split(X,y,test_size=0.20,random_state=56)

# %%
X_molecular = X.loc[:,'IDH1':]

# %%
X.columns # no. of features in our dataset

# %%
X.head()

# %%
X_molecular_count = X_molecular.loc[:,'IDH1':].aggregate('sum').reset_index()
X_molecular_count =  X_molecular_count.sort_values(by=0,ascending=False)
X_molecular_count


# %%
# plotting the maximum no. of Genes that are mutated 
plt.figure(figsize=(10, 6))
plt.bar(X_molecular_count['index'], X_molecular_count[0], color='skyblue')
plt.xlabel('Molecular Name')
plt.ylabel('Count of Number of Mutated')
plt.title('Count of Molecular Features Attaining 1')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# <h2>Adaboost

# %%
from sklearn.ensemble import AdaBoostClassifier
classifier_adb = AdaBoostClassifier(
    n_estimators=10,
    random_state=56
)

# %%
#training the Model
classifier_adb.fit(X_train,y_train)

# %%
#testing the model
y_pred = classifier_adb.predict(X_test)

# %%
#metric report
from sklearn.metrics import classification_report
result_adaboost = classification_report(y_Test,y_pred)
print(result_adaboost)
accuracy_score(y_pred,y_Test)

# %% [markdown]
# <h2>k-nn (K - nearest neighbour)
# 
# 

# %%
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = AdaBoostClassifier(
    random_state=56,
    n_estimators=10
)

# %%
#training the model
classifier_knn.fit(X_train,y_train)

# %%
#testing the Model
y_predict = classifier_knn.predict(X_test)

# %%
result_knn = classification_report(y_Test,y_pred)
print(result_knn)
accuracy_score(y_pred,y_Test)

# %% [markdown]
# <h2>Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
classifier_Lor=LogisticRegression(
    max_iter=200 # this is the number of iteration for the model to converge on the Loss function to global minimum
)
#training thre model
classifier_Lor.fit(X_train,y_train)


# %% [markdown]
# 

# %%
y_pred = classifier_Lor.predict(X_test) # teting the data
result_LoReg = classification_report(y_pred,y_Test)
print(result_LoReg)
accuracy_score(y_pred,y_Test)

# %% [markdown]
# <h2>Random Forest

# %%

classifier_rf = RandomForestClassifier(
    random_state=56
) 
# training the model
classifier_rf.fit(X_train,y_train)


# %%
# testing the Model
y_pred = classifier_rf.predict(X_test)
result = classification_report(y_pred,y_Test)
print(result)
accuracy_score(y_pred,y_Test)

# %%
from sklearn.svm import SVC
classifier_svc = SVC(
    gamma=0.9,
    random_state=56
)

# %%
# training the Model
classifier_svc.fit(X_train,y_train)

# %%
# testing the Model
y_predict = classifier_svc.predict(X_test)

# %%
result_svc = classification_report(y_predict,y_Test)
print(result_svc)
accuracy_score(y_pred,y_Test)

# %%
len(X_train)

# %% [markdown]
# <h1>GRID SEARCH

# %% [markdown]
# <h2
# >SVM

# %%

# Define the model
svm = SVC()

# Define the parameter grid
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

# Initialize GridSearchCV
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5)

# Fit the model
grid_search_svm.fit(X_train, y_train)

# Best parameters
print("Best parameters for SVM:", grid_search_svm.best_params_)

# %% [markdown]
# <h2>K-nn

# %%
from sklearn.model_selection import GridSearchCV

# Define the model
knn = KNeighborsClassifier()

# Define the parameter grid
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

# Initialize GridSearchCV
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5)

# Fit the model
grid_search_knn.fit(X_train, y_train)

# Best parameters
print("Best parameters for K-NN:", grid_search_knn.best_params_)

# %% [markdown]
# <h2>Random Forest

# %%
# Define the model
rf = RandomForestClassifier()

# Define the parameter grid
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5)

# Fit the model
grid_search_rf.fit(X_train, y_train)

# Best parameters
print("Best parameters for Random Forest:", grid_search_rf.best_params_)

# %% [markdown]
# <h2>Adaboost

# %%
# Define the model
ada = AdaBoostClassifier()

# Define the parameter grid
param_grid_ada = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1]
}

# Initialize GridSearchCV
grid_search_ada = GridSearchCV(ada, param_grid_ada, cv=5)

# Fit the model
grid_search_ada.fit(X_train, y_train)

# Best parameters
print("Best parameters for AdaBoost:", grid_search_ada.best_params_)

# %% [markdown]
# <h2> Logistic Regresion

# %%
# Define the model
log_reg = LogisticRegression()

# Define the parameter grid
param_grid_log_reg = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'max_iter':[3000]
}

# Initialize GridSearchCV
grid_search_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5)

# Fit the model
grid_search_log_reg.fit(X_train, y_train)

# Best parameters
print("Best parameters for Logistic Regression:", grid_search_log_reg.best_params_)

# %%
X_train

# %% [markdown]
# <h3>Training on Best hyperparams with the selected features 

# %%
X_train_fs = X_train[['Age_at_diagnosis',
    'PTEN',
 'GRIN2A',
 'PIK3R1',
 'NF1',
 'RB1',
 'NOTCH1',
 'IDH2',
 
 'IDH1',
 'PDGFRA',
 'EGFR',
 'TP53',
 'ATRX',
 'MUC16',
 'Race']]

# %%
X_test_fs = X_test[['Age_at_diagnosis',
    'PTEN',
 'GRIN2A',
 'PIK3R1',
 'NF1',
 'RB1',
 'NOTCH1',
 'IDH2',
 
 'IDH1',
 'PDGFRA',
 'EGFR',
 'TP53',
 'ATRX',
 'MUC16',
 'Race']]

# %%
len(X_train.columns)

# %% [markdown]
# <h2>Random Forest (fs)

# %%
classifier_rf_fs = RandomForestClassifier(
    max_depth= 10, min_samples_split =  10, n_estimators = 100,random_state=56
)
classifier_rf_fs.fit(X_train_fs,y_train)
y_pred = classifier_rf_fs.predict(X_test_fs)
print(classification_report(y_pred,y_Test))
print(f'Accuracy : {accuracy_score(y_pred,y_Test)}')

# %%
classifier_rf_fs = RandomForestClassifier(random_state=56)
classifier_rf_fs.fit(X_train_fs,y_train)
y_pred = classifier_rf_fs.predict(X_test_fs)
print(classification_report(y_pred,y_Test))
print(f'Accuracy : {accuracy_score(y_pred,y_Test)}')

# %% [markdown]
# <h3> K-nn (fs)

# %%
classifier_knn_fs = KNeighborsClassifier(
    n_neighbors= 3, weights =  'distance',
)
classifier_knn_fs.fit(X_train_fs,y_train)
y_pred = classifier_knn_fs.predict(X_test_fs)
print(classification_report(y_pred,y_Test))
print(accuracy_score(y_pred,y_Test))
print(f'Accuracy : {accuracy_score(y_pred,y_Test)}')


# %%
classifier_knn_fs = KNeighborsClassifier()
classifier_knn_fs.fit(X_train_fs,y_train)
y_pred = classifier_knn_fs.predict(X_test_fs)
print(classification_report(y_pred,y_Test))
print(accuracy_score(y_pred,y_Test))
print(f'Accuracy : {accuracy_score(y_pred,y_Test)}')


# %% [markdown]
# <h3>SVC (fs) 

# %%
classifier_SVM_fs = SVC(
    C =  10, gamma = 0.01, kernel = 'linear',random_state=56
)
classifier_SVM_fs.fit(X_train_fs,y_train)
y_pred = classifier_SVM_fs.predict(X_test_fs)
print(classification_report(y_pred,y_Test))
print(f'Accuracy : {accuracy_score(y_pred,y_Test)}')


# %%
classifier_SVM_fs = SVC(random_state=56)
classifier_SVM_fs.fit(X_train_fs,y_train)
y_pred = classifier_SVM_fs.predict(X_test_fs)
print(classification_report(y_pred,y_Test))
print(f'Accuracy : {accuracy_score(y_pred,y_Test)}')

# %% [markdown]
# <h2>LogisticRegression (fs)

# %%
classifier_LogReg_fs = LogisticRegression(
    C =  0.1, max_iter =  600, solver = 'liblinear',random_state=56
)
classifier_LogReg_fs.fit(X_train,y_train)
y_pred = classifier_LogReg_fs.predict(X_test)
print(classification_report(y_pred,y_Test))
print(f'Accuracy : {accuracy_score(y_pred,y_Test)}')


# %%
classifier_LogReg_fs = LogisticRegression(random_state=56)
classifier_LogReg_fs.fit(X_train,y_train)
y_pred = classifier_LogReg_fs.predict(X_test)
print(classification_report(y_pred,y_Test))
print(f'Accuracy : {accuracy_score(y_pred,y_Test)}')

# %% [markdown]
# <h2>AdaBoost (fs)

# %%
classifier_Adaboost_fs = AdaBoostClassifier(
    learning_rate= 0.1, n_estimators=50,random_state=56
)
classifier_Adaboost_fs.fit(X_train,y_train)
y_pred = classifier_Adaboost_fs.predict(X_test)
print(classification_report(y_pred,y_Test))
print(f'Accuracy : {accuracy_score(y_pred,y_Test)}')


# %%
classifier_Adaboost_fs = AdaBoostClassifier(random_state=56)
classifier_Adaboost_fs.fit(X_train,y_train)
y_pred = classifier_Adaboost_fs.predict(X_test)
print(classification_report(y_pred,y_Test))
print(f'Accuracy : {accuracy_score(y_pred,y_Test)}')
