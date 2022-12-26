# %% [markdown]
# # Machine Learning I Final Project
# 
# Team 4
# 
# Rebecca Li, Jackie Li, Chris Chou, Niki Baskar

# %%
# import packages
import pandas as pd
from datetime import datetime
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Embedding, Flatten, Input, concatenate
from keras_tuner import BayesianOptimization, RandomSearch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import keras.backend as K
import sklearn
import category_encoders as ce
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras import initializers
from datatable import dt, f, by, g, join, sort, update, ifelse
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import seaborn as sns

# %% [markdown]
# # Data Exploration

# %% [markdown]
# ## Data Import

# %%
# Import the training data
train = pd.read_csv('Project Data/ProjectTrainingData.csv')

# %%
# Import the test data
test = pd.read_csv('Project Data/ProjectTestData.csv')

# %%
train.shape
# there are 31991090 rows and 24 columns

# %%
# shuffle the train dataset into 10 parts
shuffled = train.sample(frac=1,random_state=42)
result = np.array_split(shuffled, 10)  

# %%
# name those parts with number
for i in range(len(result)):
    exec(f'train_{i} = result[i]')

# %%
print(train_3.shape)

# %% [markdown]
# ## Data Overview

# %%
# Explore the training data
train_3.head()

# %%
# set test set naive prediction value as 0.5
test['click'] = 0.5
# set test set column names
test = test[['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 
                    'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']]

# %%
test.shape

# %%
train_3.describe(include='all')

# %%
# # concat all data to explore
# df = pd.concat([train, test])
# # transform into a datatable just in case of speed limit
# all_dt = dt.Frame(df)

# %% [markdown]
# ## Categorical Data Encoding
# 
# Category with names:
# * id: drop
# * site categories:
#   * site_id: base 5 encoding
#   * site_domain: base 5 encoding
#   * site_category: base 10 encoding
# * app cats:
#   * app_id: base 5 encoding
#   * app_domain: one-hot encoding
#   * app_category: base 10 encoding
# * device cats:
#   * device_id: one-hot encoding
#   * device_ip: drop
#   * device_model: base 10 encoding
# 
# 
# Category without names:
# * C1: stay same
# * C14: base 5 encoding
# * C15: stay same
# * C16: stay same
# * C17: base 5 encoding
# * C18: stay same
# * C19: base 5 encoding
# * C20: base 5 encoding
# * C21: base 5 encoding

# %%
# drop id in column
del train_3['id']
del test['id']

# %%
# define a function to transform skewed categorical value based on frequency
def categorical_replace(train_data, test_data, column, pct = 0.01):
    """
    train_data: train dataset to input
    test_data: test to input
    column: column name string to input
    pct: transform frequency treshold, default 0.01
    """
    cond = train_data[column].value_counts(normalize = True) > pct
    non_others = cond[cond].index  # define a list to save main category

    train_data['temp'] = 'other'
    train_data.loc[train_data[column].isin(non_others),'temp'] = train_data[column]
    train_data[column] = train_data['temp'].values
    del train_data['temp']
    print("Train Test Replace Finished!")

    test_data['temp'] = 'other'
    test_data.loc[test_data[column].isin(non_others),'temp'] = test_data[column]
    test_data[column] = test_data['temp'].values
    del test_data['temp']
    print("Test data replace finished!")

# %%
# define function to draw category distribution
def hist_bar_cat(data, column):
    categories = data[column].value_counts().index.astype('str')
    counts = data[column].value_counts().values
    plt.bar(categories, counts, width=0.5)
    plt.title('Distribution for column {}'.format(column))

# %% [markdown]
# ### Site Categories

# %% [markdown]
# #### site_id

# %%
# check the number of unique values for each column
train_3['site_id'].nunique()

# %%
train_3['site_id'].value_counts(normalize = True).skew()
# as shown below, the skewness it is quite high

# %%
categorical_replace(train_3, test, 'site_id')

# %%
hist_bar_cat(train_3, "site_id")
# much better

# %% [markdown]
# #### site_domain

# %%
train_3['site_domain'].nunique()
# check non-dup value

# %%
train_3['site_domain'].value_counts(normalize = True).skew()
# as shown below, the skewness it is quite high

# %%
categorical_replace(train_3, test, 'site_domain')

# %%
hist_bar_cat(train_3, "site_domain")
# much better

# %% [markdown]
# #### site_category
# 
# Use Label Encoder Later

# %%
train_3['site_category'].nunique()

# %% [markdown]
# ### App Categories
# #### app_id

# %%
train_3['app_id'].nunique()

# %%
train_3['app_id'].value_counts(normalize = True).skew()
# as shown below, the skewness it is quite high

# %%
categorical_replace(train_3, test, 'app_id')
hist_bar_cat(train_3, "app_id")

# %% [markdown]
# #### app_domain

# %%
train_3['app_domain'].nunique()

# %%
train_3['app_domain'].value_counts(normalize = True).skew()
# as shown below, the skewness it is quite high

# %%
categorical_replace(train_3, test, 'app_domain', pct = 0.05)
hist_bar_cat(train_3, "app_domain")

# %% [markdown]
# #### app_category
# 
# Label Encode Later

# %%
train_3['app_category'].nunique()

# %% [markdown]
# ### Device Categoris
# #### device_id

# %%
train_3['device_id'].value_counts(normalize = True).skew()
# as shown below, the skewness it is quite high

# %%
categorical_replace(train_3, test, 'device_id', pct = 0.0005)
hist_bar_cat(train_3, "device_id")

# %% [markdown]
# #### device_ip

# %%
train_3['device_ip'].value_counts(normalize = True).skew()
# as shown below, the skewness it is quite high

# %%
categorical_replace(train_3, test, 'device_ip')
hist_bar_cat(train_3, "device_ip")

# %% [markdown]
# This variable is too skew that we decide to drop it.

# %%
del train_3['device_ip']
del test['device_ip']

# %% [markdown]
# #### device_model

# %%
train_3['device_model'].value_counts(normalize = True).skew()
# as shown below, the skewness it is quite high

# %%
categorical_replace(train_3, test, 'device_model')
hist_bar_cat(train_3, "device_model")

# %% [markdown]
# ### Anonymized Categorical Variables

# %% [markdown]
# #### C14

# %%
train_3['C14'].nunique()

# %%
train_3['C14'].value_counts(normalize = True).skew()
# as shown below, the skewness it is quite high

# %%
categorical_replace(train_3, test, 'C14', pct = 0.01)
hist_bar_cat(train_3, "C14")

# %% [markdown]
# #### C17

# %%
train_3['C17'].nunique()

# %%
train_3['C17'].value_counts(normalize = True).skew()
# as shown below, the skewness it is quite high

# %%
categorical_replace(train_3, test, 'C17')
hist_bar_cat(train_3, "C17")

# %% [markdown]
# #### C19

# %%
train_3['C19'].nunique()

# %%
train_3['C19'].value_counts(normalize = True).skew()
# as shown below, the skewness it is quite high

# %%
categorical_replace(train_3, test, 'C19')
hist_bar_cat(train_3, "C19")

# %% [markdown]
# #### C20

# %%
train_3['C20'].nunique()

# %%
train_3['C20'].value_counts(normalize = True).skew()
# as shown below, the skewness it is quite high

# %%
categorical_replace(train_3, test, 'C20')
hist_bar_cat(train_3, "C20")

# %% [markdown]
# #### C21

# %%
train_3['C21'].nunique()

# %%
train_3['C21'].value_counts(normalize = True).skew()
# as shown below, the skewness it is not quite high

# %%
categorical_replace(train_3, test, 'C21')

# %%
hist_bar_cat(train_3, "C21")

# %% [markdown]
# ### Encoding

# %% [markdown]
# #### One-hot

# %%
# one-hot encoding: app_domain, device_id
encoder=ce.OneHotEncoder(cols=['app_domain','device_id'],handle_unknown='return_nan',return_df=True,use_cat_names=True)
one_hot_encoder = encoder.fit(train_3)

# %%
train_3 = one_hot_encoder.transform(train_3)

# %%
test = one_hot_encoder.transform(test)

# %% [markdown]
# #### Base 5 

# %%
# base 5 encoding: site_id, site_domain, app_id, C14, C17, C19, C20, C21
encoder1 = ce.BaseNEncoder(cols=['site_id','site_domain','app_id','C14','C17','C19','C20','C21'], return_df=True, base=5)
base_5_encoder = encoder1.fit(train_3)

# %%
train_3 = base_5_encoder.transform(train_3)

# %%
test = base_5_encoder.transform(test)

# %% [markdown]
# #### Base 10

# %%
# base 10 encoding: site_category, app_category, device_model
encoder2 = ce.BaseNEncoder(cols=['site_category','app_category','device_model'], return_df=True, base=10)
base_10_encoder = encoder2.fit(train_3)

# %%
train_3 = base_10_encoder.transform(train_3)

# %%
test = base_10_encoder.transform(test)

# %% [markdown]
# ## Numerical Variables
# * hour: detail shown below
# * banner_pos: encoded by professor
# * device_type: encoded by professor
# * device_conn_type: encoded by professor

# %% [markdown]
# ### Hour
# - Day: week number
# - Hour:
#   - 1: 00-06
#   - 2: 07-12
#   - 3: 13-18
#   - 4: 19-24

# %%
splitat = 6
train_3['day'], train_3['time'] = train_3['hour'].astype('str').str[:splitat], train_3['hour'].astype('str').str[splitat:].astype('int')

# %%
# time of day
def add_time_of_day(data):
    data['day'], data['time'] = data['hour'].astype('str').str[:splitat], data['hour'].astype('str').str[splitat:].astype('int')
    conditions = [(data.time <=6),
                (data.time > 6) & (data.time <= 12),
                (data.time > 12) & (data.time <= 18),
                (data.time >18)
                ]
    values = [1, 2, 3, 4]
    data['time_of_day'] = np.select(conditions, values, 0)

# %%
add_time_of_day(train_3)
add_time_of_day(test)

# %%
train_3[['day','time','hour','time_of_day']].head(5)

# %%
def add_day_of_week(data):
    data['day']= pd.to_datetime(data['day'],format="%y%m%d")
    data['day_of_week'] = data['day'].dt.dayofweek

# %%
add_day_of_week(train_3)
add_day_of_week(test)

# %% [markdown]
# ## Finalize the data

# %%
train_3.head()

# %%


# %%
x_col = ['C1', 'banner_pos', 'site_id_0', 'site_id_1',
       'site_domain_0', 'site_domain_1', 'site_category_0', 'site_category_1',
       'app_id_0', 'app_id_1', 'app_domain_7801e8d9', 'app_domain_ae637522',
       'app_domain_2347f47a', 'app_domain_other', 'app_category_0',
       'app_category_1', 'device_id_other', 'device_id_a99f214a',
       'device_id_c357dbff', 'device_model_0', 'device_model_1', 'device_type', 'device_conn_type',
       'C14_0', 'C14_1', 'C15', 'C16', 'C17_0', 'C17_1', 'C18', 'C19_0',
       'C19_1', 'C20_0', 'C20_1', 'C21_0', 'C21_1',
       'time_of_day', 'day_of_week']
y_col = 'click'

# %%
X_train3 = train_3[x_col]
y_train3 = train_3[y_col]
X_test = test[x_col]
y_pred_naive = test[y_col]

# %% [markdown]
# # Model Training

# %%
# Split to Sub-train & Validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train3, y_train3, test_size=0.2, random_state=42)

# %% [markdown]
# ## Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression

# %%
lr = LogisticRegression(random_state=42, n_jobs=-1, penalty="l2").fit(X_train, y_train)

# %%
y_pred = lr.predict_proba(X_val)

# %%
# print log loss of the validation data
log_loss(y_val, y_pred)

# %% [markdown]
# Since the log-loss is quite high, LR is not a great model to be chosen as a final step.

# %% [markdown]
# ## Random Forest

# %% [markdown]
# ### TuneGridsearchCV

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rfc=RandomForestClassifier(random_state=42)

# %% [markdown]
# Random Serach using random forest

# %%
rs_space={'max_depth':[20,30],
              'max_features':['sqrt', 'log2'],
              'min_samples_split':[50, 100]
         }

# %%
# param_grid = { 
#     'n_estimators': [100, 200, 300, 400],
#     'max_features': ['sqrt', 'log2'],
#     'min_samples_split': [2, 10, 100],
#     'max_depth' : [10,20,30],
# }
# gs_rfc = GridSearchCV(estimator=rfc,
#     param_grid=param_grid,
#     cv=3,
#     scoring='neg_log_loss',
#     n_jobs=-1, 
#     verbose = 3)
# gs_rfc.fit(X_train, y_train)

# %%
from tune_sklearn import TuneGridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# %%
gs_rfc = TuneGridSearchCV(rfc, rs_space, 
                                scoring='neg_log_loss', 
                                n_jobs=-1, 
                                cv=3,
                                max_iters=10,
                                early_stopping = True,
                                verbose=2)

# %%
gs_rfc.fit(X_train, y_train)

# %%
# rfc_random = RandomizedSearchCV(rfc, rs_space, 
#                                 scoring='neg_log_loss', 
#                                 n_jobs=-1, 
#                                 cv=2,
#                                 n_iter=10,
#                                 verbose=30)
# rfc_random.fit(X_train,y_train)

# %%
print("Parameter: ", gs_rfc.best_params_)
print("Non-nested LogLoss: ", gs_rfc.best_score_)
print("Best Estimator: ", gs_rfc.best_estimator_)

# %%
rf = RandomForestClassifier(n_estimators = 10, 
                            random_state = 42,
                            min_samples_split=100, 
                            max_depth=20,
                            n_jobs=-1)

# %%
rf.fit(X_train, y_train)

# %%
y_pred_rf = rf.predict_proba(X_val)
log_loss(y_val, y_pred_rf)

# %% [markdown]
# ## Neural Net

# %%
# initialize nn
nnc = Sequential()
# add input layer
nnc.add(Dense(10, kernel_regularizer = regularizers.l2(0.003), kernel_initializer=initializers.RandomNormal(stddev=0.01), activation='relu', input_shape = (X_train.shape[1],), use_bias=True))
nnc.add(Dropout(0.4))
# add hidden layer
nnc.add(Dense(10, kernel_regularizer = regularizers.l2(0.003), kernel_initializer=initializers.RandomNormal(stddev=0.01), activation='relu'))
nnc.add(Dropout(0.3))
nnc.add(Dense(units = 6, activation = 'sigmoid', use_bias = True))
# add output layer
nnc.add(Dense(1, activation = 'sigmoid', use_bias = True))
nnc.compile(optimizer = Adam(learning_rate=1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])

# %%
nnc.fit(X_train, y_train, epochs=10, batch_size = 512, validation_data = (X_val, y_val))

# %%
y_pred_nn = nnc.predict(X_val)
log_loss(y_val, y_pred_nn)

# %% [markdown]
# ## LightGBM

# %%
lgb = LGBMClassifier(random_state=42)

# %%
lgb.fit(X_train,y_train,verbose=3,eval_metric='logloss')

# %%
parameters = {
    'learning_rate': [0.01,0.05,0.09,0.1],
    'num_leaves': [31,250,300],
}

gbm = LGBMClassifier(objective='binary', 
                          n_jobs=-1,
                          metric = 'binary_logloss',
                          boosting_type='gbdt',
                          cat_smooth= 35)
gs_lgb = GridSearchCV(gbm, param_grid=parameters, scoring='neg_log_loss', cv=3, verbose = 1)

# %%
gs_lgb.fit(X_train, y_train)

# %%
print('best parameter:{0}'.format(gs_lgb.best_params_))

# %%
lgb = LGBMClassifier(learning_rate=0.1,num_leaves = 300,random_state=42, metric = 'binary_logloss',cat_smooth= 35).\
    fit(X_train, y_train,eval_metric='logloss')

# %%
y_pred_lgb=lgb.predict_proba(X_val)[:, 1]

# %%
log_loss(y_val, y_pred_lgb)

# %% [markdown]
# ## CatBoost

# %%
from catboost import CatBoostClassifier, Pool

# %%
ctb = CatBoostClassifier(random_seed=42)

# %%
ctb.fit(X_train, y_train)

# %%
y_pred_ctb=ctb.predict_proba(X_val)[:, 1]

# %%
log_loss(y_val, y_pred_ctb)

# %% [markdown]
# # Final Result
# 
# Lightgbm
# 
# ## Feature Importance

# %%
feature_imp = pd.DataFrame(sorted(zip(lgb.feature_importances_,X_train.columns)), columns=['Value','Feature'])

# %%
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')

# %% [markdown]
# ## ROC

# %%
############################### Import Libraries & Modules #################################
from sklearn.metrics import roc_curve, auc
from numpy import interp
from sklearn.model_selection import StratifiedKFold
lgb = LGBMClassifier(learning_rate=0.1,num_leaves = 300,random_state=42, metric = 'binary_logloss',cat_smooth= 35)

# %%
#################################### Cross - Validation ####################################

# This cross-validation object is a variation of KFold that returns stratified folds
# The folds are made by preserving the percentage of samples for each class
cv = list(StratifiedKFold(n_splits=5,                 # number of folds. Must be at least 2
                          ).split(X_train, y_train))  # generate indices to split data into training and test set

# %%
##################################### Visualization ######################################

fig = plt.figure(figsize=(7, 5)) # set figure size

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv): # enumarate allows us to loop over something and have an automatic counter (e.g, 0, 1, 2, etc.)
    probas = lgb.fit(X_train.iloc[train],y_train.iloc[train]).predict_proba(X_train.iloc[test]) # make predictions based on classifiers
    
    # roc_curve will compute Receiver operating characteristic (ROC)
    fpr, tpr, thresholds = roc_curve(y_train.iloc[test], # data for ROC curves (true labels)
                                     probas[:, 1],  # predictions based on estimators
                                     pos_label=1)   # the label of the positive class
    mean_tpr += interp(mean_fpr, fpr, tpr)          # one-dimensional linear interpolation (continuous ROC curve)

    mean_tpr[0] = 0.0
    # auc will compute Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,                                   # the horizontal coordinates of the data points
             tpr,                                   # the vertical coordinates of the data points
             label='ROC fold %d (area = %0.2f)'
                   % (i+1, roc_auc))

plt.plot([0, 1],                                    # plot random guessing classifier
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='random guessing')

mean_tpr /= len(cv)                                 # plot mearn ROC curve
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, 
         mean_tpr, 
         'k--',
         label='mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.plot([0, 0, 1],                                 # plot perfect classifier
         [0, 1, 1],
         linestyle=':',
         color='black',
         label='perfect performance')

# Figure paramaters: x axis limits, y axis limits, labels of axes, legend position
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc="lower right")

plt.tight_layout()
# plt.savefig('ROC_CrossValidation_Oneclassifier.png', dpi=300)
plt.show()                                          # display figure

# %% [markdown]
# ## Submission

# %%
y_sub = lgb.predict_proba(X_test)[:,1]

# %%
# Import the submission data
sub = pd.read_csv('Project Data/ProjectSubmission-Team4.csv')

# %%
sub.iloc[:,1] = y_sub

# %%
sub.to_csv('ProjectSubmission-Team4.csv',index = False)

# %%



