#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# ## Business Use Case
# 
# There has been a revenue decline for a Portuguese bank and they would like to know what actions to take. After investigation, they found out that the root cause is that their clients are not depositing as frequently as before. Knowing that term deposits allow banks to hold onto a deposit for a specific amount of time, so banks can invest in higher gain financial products to make a profit. In addition, banks also hold better chance to persuade term deposit clients into buying other products such as funds or insurance to further increase their revenues. As a result, the Portuguese bank would like to identify existing clients that have higher chance to subscribe for a term deposit and focus marketing efforts on such clients.
# 
# ## Data Science Problem Statement
# 
# Predict if the client will subscribe to a term deposit based on the analysis of the marketing campaigns the bank performed.
# 
# ## Evaluation Metric
# 
# We will be using AUC - Probability to discriminate between subscriber and non-subscriber.
# 
# # Understanding the Dataset
# 
# The data is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be subscribed ('yes') or not ('no') subscribed.
# There are two datasets: train.csv with all examples (32950) and 14 inputs including the target feature, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
# 
# test.csv which is the test data that consists of 8238 observations and 13 features without the target feature
# 
# Goal:- The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).
# 
# ### Features

# In[1]:


{ ["|Feature|Feature_Type|Description|\n",
 "|-----|-----|-----|\n",
 "|age|numeric|age of a person|  \n",
 "|job |Categorical,nominal|type of job ('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')|  \n",
 "|marital|categorical,nominal|marital status ('divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)|  \n",
 "|education|categorical,nominal| ('basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown') | \n",
 "|default|categorical,nominal| has credit in default? ('no','yes','unknown')|  \n",
 "|housing|categorical,nominal| has housing loan? ('no','yes','unknown')|  \n",
 "|loan|categorical,nominal| has personal loan? ('no','yes','unknown')|  \n",
 "|contact|categorical,nominal| contact communication type ('cellular','telephone')|  \n",
 "|month|categorical,ordinal| last contact month of year ('jan', 'feb', 'mar', ..., 'nov', 'dec')| \n",
 "|day_of_week|categorical,ordinal| last contact day of the week ('mon','tue','wed','thu','fri')|  \n",
 "|duration|numeric| last contact duration, in seconds . Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no')|\n",
 "|campaign|numeric|number of contacts performed during this campaign and for this client (includes last contact)|   \n",
 "|poutcome|categorical,nominal| outcome of the previous marketing campaign ('failure','nonexistent','success')|  \n",
 "\n",
 "**Target variable (desired output):**  \n",
 "\n",
 "|Feature|Feature_Type|Description|\n",
 "|-----|-----|-----|\n",
 "|y | binary| has the client subscribed a term deposit? ('yes','no')|"
]
  }


# ### Importing Necessary libraries
# 
# The following code is written in Python 3. Libraries provide pre-written functionality to perform necessary tasks.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ### Loading Data Modelling Libraries
# 
# We use the scikit-learn library to develope our machine learning algorithms.
# For data visualization, we will use the matplotlib and seaborn library. Below are common classes to load.

# In[3]:


from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier ,GradientBoostingClassifier
#from xgboost import XGBClassifier 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import roc_auc_score ,mean_squared_error,accuracy_score,classification_report,roc_curve,confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns',None)
import six
import sys
sys.modules['sklearn.externals.six'] = six


# # Data Loading and Cleaning
# 
# ### Load the Preprocessed dataset
# - In this task, we'll load the dataframe in pandas, drop the unnecessary columns and display the top five rows of the dataset.

# In[5]:


# accessing to the folder where the file is stored
file = open(r'C:\Users\munna electronics\Desktop\Bootcamp_Problem_Statement-master\data\preprocessed_data.csv', encoding='utf-8')

# Load the dataframe
dataframe = pd.read_csv(file)

print('Shape of the data is: ',dataframe.shape)

dataframe.head()


# ## Applying Vanilla Models on the Data
# 
# Since we have performed preprocessing on our data and also done with the EDA part, it is now time to apply vanilla models on the data and check their performance.
# 
# ### Fit Vanilla Classification models
# 
# There are many Classification algorithms are present in machine learning, which are used for different classification applications. Some of the main classification algorithms are as follows-
# 
# Logistic Regression
# DecisionTree Classifier
# RandomForest Classfier
# The code we have written below internally splits the data into training data and validation data. It then fits the classification model on the train data and then makes a prediction on the validation data and outputs the scores for this prediction.
# 
# #### Preparing the train and test data

# In[6]:


# Predictors
X = dataframe.iloc[:,:-1]

# Target
y = dataframe.iloc[:,-1]

# Dividing the data into train and test subsets
x_train,x_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=5)


# #### Fitting the model and predicting the values

# In[7]:


# run Logistic Regression model
model = LogisticRegression()
# fitting the model
model.fit(x_train, y_train)
# predicting the values
y_scores = model.predict(x_val)


# #### Getting the metrics to check our model performance

# In[8]:


# getting the auc roc curve
auc = roc_auc_score(y_val, y_scores)
#print('Classification Report:')
#print(classification_report(y_val,y_scores))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_scores)
print('ROC_AUC_SCORE is',roc_auc_score(y_val, y_scores))
    
#fpr, tpr, _ = roc_curve(y_test, predictions[:,1])
    
plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# ### *The above two steps are combined and run in a single cell for all the remaining models respectively*

# In[9]:


# Run Decision Tree Classifier
model = DecisionTreeClassifier()

model.fit(x_train, y_train)
y_scores = model.predict(x_val)
auc = roc_auc_score(y_val, y_scores)
#print('Classification Report:')
#print(classification_report(y_val,y_scores))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_scores)
print('ROC_AUC_SCORE is',roc_auc_score(y_val, y_scores))
    
#fpr, tpr, _ = roc_curve(y_test, predictions[:,1])
    
plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# In[10]:


# run Random Forrest Classifier
model = RandomForestClassifier()

model.fit(x_train, y_train)
y_scores = model.predict(x_val)
auc = roc_auc_score(y_val, y_scores)
#print('Classification Report:')
#print(classification_report(y_val,y_scores))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_scores)
print('ROC_AUC_SCORE is',roc_auc_score(y_val, y_scores))
    
#fpr, tpr, _ = roc_curve(y_test, predictions[:,1])
    
plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# # Grid-Search & Hyperparameter Tuning
# 
# Hyperparameters are function attributes that we have to specify for an algorithm.
# 
# In the next step we will be using Grid search to come up with the best set and values for our hyperparameters.
# 
# ### Grid Search for Random Forest
# 
# In the below task, we write a code that performs hyperparameter tuning for a random forest classifier. We have used the hyperparameters max_features, max_depth and criterion for this task. Feel free to play around with this function by introducing a few more hyperparameters and chaniging their values

# In[11]:


# splitting the data
x_train,x_val,y_train,y_val = train_test_split(X,y, test_size=0.3, random_state=42, stratify=y)
# selecting the classifier
rfc = RandomForestClassifier()
# selecting the parameter
param_grid = { 
'max_features': ['auto', 'sqrt', 'log2'],
'max_depth' : [4,5,6,7,8],
'criterion' :['gini', 'entropy']
             }
# using grid search with respective parameters
grid_search_model = GridSearchCV(rfc, param_grid=param_grid)
# fitting the model
grid_search_model.fit(x_train, y_train)
# printing the best parameters
print('Best Parameters are:',grid_search_model.best_params_)


# ## Applying the best parameters obtained using Grid Search on Random Forest model
# 
# In the task below, we fit a random forest model using the best parameters obtained using Grid Search. Since the target is imbalanced, we apply Synthetic Minority Oversampling (SMOTE) for undersampling and oversampling the majority and minority classes in the target respectively.
# 
# <strong>Kindly note that SMOTE should always be applied only on the training data and not on the validation and test data.</strong>
# 
# You can try experimenting with and without SMOTE and check for the difference in recall.

# In[12]:


from sklearn.metrics import roc_auc_score,roc_curve,classification_report
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from yellowbrick.classifier import roc_auc


# A function to use smote
def grid_search_random_forest_best(dataframe,target):
    
    # splitting the data
    x_train,x_val,y_train,y_val = train_test_split(dataframe,target, test_size=0.3, random_state=42)
    
    # Applying Smote on train data for dealing with class imbalance
    smote = SMOTE()
    
    X_sm, y_sm =  smote.fit_sample(x_train, y_train)
    # Intializing the Random Forrest Classifier
    rfc = RandomForestClassifier(max_features='log2', max_depth=8, criterion='gini',random_state=42)
    # Fit the model on data
    rfc.fit(X_sm, y_sm)
    # Get the predictions on the validation data
    y_pred = rfc.predict(x_val)
    # Evaluation of result with the auc_roc graph
    visualizer = roc_auc(rfc,X_sm,y_sm,x_val,y_val)


grid_search_random_forest_best(X,y)


# ## Prediction on the test data
# 
# In the below task, we have performed a prediction on the test data. We have used Random Forrest for this prediction.
# 
# We have to perform the same preprocessing operations on the test data that we have performed on the train data. For demonstration purposes, we have preprocessed the test data and this preprocessed data is present in the csv file new_test.csv
# 
# We then make a prediction on the preprocessed test data using the random forrest model with the best parameter values we've got. And as the final step, we will read the submission.csvand concatenate this prediction with the Id column which is the unique client id and then convert this into a csv file which becomes the final_submission.csv

# In[13]:


test = pd.read_csv(r'C:\Users\munna electronics\Desktop\Bootcamp_Problem_Statement-master\data\new_test.csv', encoding='utf-8')
test.head()


# In[14]:


# Initialize Smote
smote = SMOTE()

# Applying SMOTE
X_sm, y_sm =  smote.fit_sample(x_train, y_train)

# Initialize our Random forrest model with the best parameter values derived
rfc = RandomForestClassifier(max_features='log2', max_depth=8, criterion='gini',random_state=42)

# Fitting the model
rfc.fit(X_sm,y_sm)

# Predict on the preprocessed test file
y_pred = rfc.predict(test)

# storing the predictions
prediction = pd.DataFrame(y_pred,columns=['y'])

# reading the submission file with client ids
submission = pd.read_csv('../data/submission.csv')

# Concatenate predictions and create our final submission file
final_submission = pd.concat([submission['Id'],prediction['y']],1)

# Results
final_submission.head()


# In[ ]:




