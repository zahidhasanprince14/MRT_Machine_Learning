#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import time
pd.options.display.max_columns = None
import warnings
warnings.filterwarnings("ignore")
import math
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


# In[48]:


RF_Data = pd.read_csv("D:\MSC thesis\MSc_Thesis_Sept_07.csv")

RF_Data 


# In[49]:


# Assuming RF_Data is your DataFrame
description = RF_Data.describe()

# Save the description as a CSV file
description.to_csv('D:\MSC thesis\statistics.csv')

# Optionally, you can save it as an Excel file as well
# description.to_excel('/mnt/data/RF_Data_description.xlsx')


# In[50]:


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# List of columns to be standardized
columns_to_standardize = ['Personal_Income', 'Family_income', 'Fare_MRT_total', 'Fare_Bus_total','Fare_MRT_only', 'Fare_Bus_only']

#'TT_MRT_total', 'Total Waiting Time', 'TT_Bus_total','TT_Bus_only','TT_MRT_only','Access_EgressTT','Ttmain_PV'
                          
# Standardize the selected columns
RF_Data[columns_to_standardize] = scaler.fit_transform(RF_Data[columns_to_standardize])

# Show the first few rows of the dataset with the standardized columns
print(RF_Data[columns_to_standardize].head())


# In[51]:


RF_Data.columns


# In[52]:


# 'Fare_MRT_total', 'Fare_Bus_total','Fare_MRT_only', 'Fare_Bus_only',
# 'TT_MRT_total', 'Total Waiting Time', 'TT_Bus_total','TT_Bus_only','TT_MRT_only','Access_EgressFare','Access_EgressTT','Ttmain_PV'
                          

RF_Data =RF_Data.drop(['TT_MRT_total','TT_Bus_total','Fare_MRT_total', 'Fare_Bus_total','Trip 1 (MRT=1, others=0)','Trip 3 (MRT=1, others=0)','Age','Response ID','Fare_Current_total','TT_Current_total','Fare_MRT_only','Fare_Bus_only'], axis=1)


# In[53]:


len(RF_Data)


# In[54]:


import pandas as pd
import numpy as np

def cronbach_alpha(data):
    item_vars = data.var(axis=0, ddof=1)  # Calculate variance for each item
    total_var = item_vars.sum()  # Calculate the sum of item variances
    num_items = len(data.columns)  # Get the number of items
    
    # Calculate Cronbach's alpha coefficient
    if num_items == 1:
        return np.nan
    else:
        return (num_items / (num_items - 1)) * (1 - (total_var / data.sum(axis=1).var(ddof=1)))
    
alpha = cronbach_alpha(RF_Data)
print("Cronbach's alpha coefficient:", alpha)


# In[55]:


cor_df = RF_Data.corr(method = "pearson").round(2)
cor_df


# Assuming cor_df is your correlation DataFrame
cor_df.to_excel('D:\MSC thesis\correlation_matrix.xlsx', sheet_name='Correlation Matrix')


# In[56]:


#from sklearn.preprocessing import OrdinalEncoder
#ord_enc = OrdinalEncoder()
#RF_Data["Gender"] = ord_enc.fit_transform(RF_Data[["Gender"]])
#RF_Data["IFM"] = ord_enc.fit_transform(RF_Data[["IFM"]])


# In[64]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Calculate the correlation matrix
cor_df = RF_Data.corr(method="pearson").round(2)

# Set up the matplotlib figure with increased DPI
plt.figure(figsize=(49, 40), dpi=500)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(cor_df, dtype=bool))

# Set up annotation keyword arguments with larger font size
annot_kws = {"size": 23.5}  # Adjust the font size as needed

# Create a heatmap using seaborn
sns.heatmap(cor_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws=annot_kws, mask=mask)

# Add labels and title with larger font size
plt.title("Pearson Correlation Heatmap", fontsize=50)
plt.xlabel("Features", fontsize=40)
plt.ylabel("Features", fontsize=40)

# Adjust x and y tick labels' font size
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

# Save the plot as a PNG file
plt.savefig("correlation_heatmap.png", bbox_inches='tight')

# Display the plot
plt.show()


# In[12]:


import pandas as pd
import statsmodels.api as sm  # Import the statsmodels API
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices


# Select the features (independent variables) for VIF calculation
X = RF_Data.drop(columns=['Trip 2 (MRT=1, others=0)'])  # Replace 'target_column' with the name of the dependent variable (if applicable)

# Add a constant column (intercept) for statsmodels
X = sm.add_constant(X)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns

# Compute VIF values
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display VIF values
print(vif_data)


# In[13]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eig

# Assuming RF_Data is your dataset
X = RF_Data.drop(columns=['Trip 2 (MRT=1, others=0)'])

# Get the column names (variable names)
variable_names = X.columns

def condition_index(X, variable_names):
    # Standardizing the independent variables (X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute eigenvalues from the cross product of the scaled matrix
    XTX = np.dot(X_scaled.T, X_scaled)
    eigenvalues, _ = eig(XTX)
    
    # Sort eigenvalues in descending order
    eigenvalues = sorted(eigenvalues, reverse=True)
    
    # Condition index is the square root of the ratio of the largest eigenvalue to each eigenvalue
    condition_indices = np.sqrt(eigenvalues[0] / eigenvalues)
    
    # Create a DataFrame to pair Condition Index with variable names
    condition_index_df = pd.DataFrame({
        'Variable': variable_names,
        'Condition Index': condition_indices
    })
    
    return condition_index_df

# Calculate the Condition Index with variable names
condition_indices_df = condition_index(X, variable_names)

# Print the result
print(condition_indices_df)


# In[14]:


#Cat Var Handling
X = RF_Data.drop(['Trip 2 (MRT=1, others=0)'], axis = 1)
y = RF_Data['Trip 2 (MRT=1, others=0)']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state = 1)
#features_to_encode = X_train.columns[X_train.dtypes==object].tolist()
#features_to_scale = X_train.columns[X_train.dtypes==int].tolist()



#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import make_column_transformer
#from sklearn.preprocessing import StandardScaler
#col_trans = make_column_transformer(
#                        (OneHotEncoder(),features_to_encode),
#                       remainder = "passthrough")


# In[15]:


len(X)


# In[16]:


X_train.columns 


# In[17]:


#RF TRAINING 
from sklearn.ensemble import RandomForestClassifier

seed= 50
rf_classifier = RandomForestClassifier(
    min_samples_leaf=1,
    max_depth=10,
    n_estimators=180,
    max_leaf_nodes=100,
    min_samples_split=2,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42,
    max_features='sqrt'  # Change 'auto' to 'sqrt', 'log2', None, or an integer value
)

rf_classifier.fit(X_train, y_train)
#from sklearn.pipeline import make_pipeline
#pipe = make_pipeline(col_trans, rf_classifier)
#pipe.fit(X_train, y_train)


# In[18]:


y_pred = rf_classifier.predict(X_test)
y_pred_training = rf_classifier.predict(X_train)


# In[19]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
accuracy_score(y_test, y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)
print(f"The accuracy (testing data) of the model is {round(accuracy_score(y_test,y_pred),3)*100} %")
print(f"The precision (testing data) of the model is {round(precision_score(y_test,y_pred),3)*100} %")
print(f"The recall (testing data) of the model is {round(recall_score(y_test,y_pred),3)*100} %")
print(f"f1_score (testing data) of the model is {round(f1_score(y_test,y_pred),3)*100} %")


# In[20]:


accuracy_score(y_train, y_pred_training)
precision_score(y_train, y_pred_training)
recall_score(y_train, y_pred_training)
f1_score(y_train, y_pred_training)

print(f"The accuracy (training data)of the model is {round(accuracy_score(y_train,y_pred_training),3)*100} %")
print(f"The precision (training data) of the model is {round(precision_score(y_train,y_pred_training),3)*100} %")
print(f"The recall (training data) of the model is {round(recall_score(y_train,y_pred_training),3)*100} %")
print(f"f1_score (training data) of the model is {round(f1_score(y_train,y_pred_training),3)*100} %")


# In[21]:


train_probs = rf_classifier.predict_proba(X_train)[:,1] 
probs = rf_classifier.predict_proba(X_test)[:, 1]
train_predictions = rf_classifier.predict(X_train)
#In general, an AUC of 0.5 suggests no discrimination (i.e., ability to diagnose patients with and without the disease or condition based on the test), 0.7 to 0.8 is considered acceptable, 0.8 to 0.9 is considered excellent, and more than 0.9 is considered outstanding.
print(f'Train Reciever Operating Characteristics Area Under Curve (AUC) Score: {roc_auc_score(y_train, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test, probs)}')


# In[22]:


#Plot ROC Curve
def evaluate_model(y_pred, probs,train_predictions, train_probs):
    baseline = {}
    baseline['recall']=recall_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    results = {}
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['roc'] = roc_auc_score(y_test, probs)
    train_results = {}
    train_results['recall'] = recall_score(y_train,       train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)
    for metric in ['recall', 'precision', 'roc']:  
          """""print(f'{metric.capitalize()} 
                 Baseline: {round(baseline[metric], 2)} 
                 Test: {round(results[metric], 2)} 
                 Train: {round(train_results[metric], 2)}')"""
     # Calculate false positive rates and true positive rates
    base_fpr, base_tpr,_ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr,_ = roc_curve(y_test, probs)
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.plot(model_fpr, model_tpr, 'r', label = 'ROC AUC  Score: 0.98')
    plt.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves (RF)');
    plt.show();
  
evaluate_model(y_pred,probs,train_predictions,train_probs)


# In[23]:


#Best Threshold 
fpr, tpr, thresholds = roc_curve(y_test, probs)
from numpy import sqrt 
gmeans = sqrt(tpr* (1-fpr))
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))


# In[24]:


len(y_test)
y_test.value_counts()


# In[25]:


#Confusion Matrix
import itertools

def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): # can change color     plt.figure(figsize = (10, 10))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, size = 14)
    plt.yticks(tick_marks, classes, rotation=90, size = 14)    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.    
    # Label the plot    
    
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):    
        plt.text(j, i, format(cm[i, j], fmt), 
             fontsize = 20,
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")   
    
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)
    
# Let's plot it out
cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm, classes = ['0 - Current', '1 - Metro'],
                     title = 'Mode Confusion Matrix (RF)')


# In[26]:


# Assuming rf_model is your trained RandomForestClassifier

feature_importances = rf_classifier.feature_importances_


# Get the feature names after OneHotEncoding
#encoded_columns = onehot_encoder.get_feature_names_out(features_to_encode)

# Extract feature names from the original X_train DataFrame (before encoding)
original_feature_names = X_train.columns

# Create a DataFrame to display feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': original_feature_names, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance
print(feature_importance_df)


# In[28]:


import matplotlib.pyplot as plt

# Assuming you have the feature_importance_df DataFrame
# Sort the DataFrame by importance in descending order (if not sorted already)
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Create a bar chart with smaller font size
plt.figure(figsize=(15, 14))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='green')
plt.xlabel('Importance Factor', fontsize=18)  # Adjust font size for x-axis label
plt.ylabel('Feature', fontsize=18)     # Adjust font size for y-axis label
plt.title('Feature Importance (RF)', fontsize=18)  # Adjust font size for title
plt.tick_params(axis='both', which='major', labelsize=18)  # Adjust font size for tick labels
plt.show()


# In[81]:


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

# Assuming the RandomForestClassifier has already been trained as rf_classifier
# If you haven't trained it yet, train it with your data:
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)

# Select a single tree from the forest (e.g., the first tree)
estimator = rf_classifier.estimators_[0]

# Plot the tree using plot_tree with options to display Gini index, class names, and feature names
plt.figure(figsize=(40, 20), dpi=150)
plot_tree(estimator, 
          feature_names=['Sex(F=0)', 'EmpS_Full time Worker',
       'EmpS_Hybrid (Can work at home or at office)', 'Emps_Others',
       'EmpS_Part Time', 'EmpS_Student', 'EmpS_Work at home (Full-time)',
       'EmpS_Work at home (Part-time)', 'Rapid pass/MRT pass for MRT (Y=1)',
       'last educational degree', 'H_size', 'EarningFM', 'Personal_Income',
       'Family_income', 'Total Distance for access  trip',
       'Total Distance in km', 'Trip 2 (MRT=1, others=0)', 'Work_cen',
       'Family_incl', 'Ped_infra', 'Reliability', 'Safety', 'Inflation_conc',
       'Walking_ben', 'Social_media', 'Social_influence', 'Age<35', 'Age<45',
       'Age<55', 'Age<65', 'One_PV', 'Greater_Than_One_PV ',
       'Total Waiting Time', 'distance_500m', 'Very Flexible',
       'Somewhat Flexible', 'Ttmain_PV', 'TT_Bus_only', 'TT_MRT_only',
       'Access_EgressFare', 'Access_EgressTT', 'Diff_Bus&MRT_Fare'],  # Use your actual feature names
          class_names=['Class 0', 'Class 1'],  # Replace with your actual class labels
          filled=True,                    # Color nodes by class
          impurity=True,                  # Show Gini index
          rounded=True,                   # Rounded corners for better readability
          fontsize=10)                    # Adjust font size for readability

plt.title("Decision Tree from Random Forest with Gini Index")
plt.show()
plt.figure(figsize=(100, 50))


# In[90]:


#######################################    Adaboost      ##################################################


# In[29]:


# explore adaboost ensemble number of trees effect on performance
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot


# In[30]:


def get_models():
    models = dict()
    # define number of trees to consider
    n_trees = [ 50, 60, 70, 80, 90, 100]
    for n in n_trees:
        models[str(n)] = AdaBoostClassifier(n_estimators=n)
    return models


# In[61]:




# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the results
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
return scores


# In[171]:


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
 # evaluate the model
 scores = evaluate_model(model, X, y)
 # store the results
 results.append(scores)
 names.append(name)
 # summarize the performance along the way
 print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# In[65]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from numpy import mean, std

# Function to get models
def get_models():
    models = dict()
    # Define number of trees to consider
    n_trees = [50, 60, 70, 80, 90, 100]
    for n in n_trees:
        models[str(n)] = AdaBoostClassifier(n_estimators=n)
    return models

# Function to evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    # Define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # Evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

# Assume X and y are already defined as your features and target variable
# Get the models to evaluate
models = get_models()

# Evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    # Evaluate the model
    scores = evaluate_model(model, X, y)
    # Store the results
    results.append(scores)
    names.append(name)
    # Summarize the performance along the way
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

# Plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.title('Comparison of AdaBoost Model Accuracies with Different Numbers of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


# In[66]:


def get_models():
    models = dict()
    # explore depths from 1 to 10
    for i in range(1,15):
        # define base model
        base = DecisionTreeClassifier(max_depth=i)
        # define ensemble model
        models[str(i)] = AdaBoostClassifier(base_estimator=base)
    return models


# In[175]:


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the results
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
return scores


# In[67]:


from sklearn.tree import DecisionTreeClassifier
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
 # evaluate the model
 scores = evaluate_model(model, X, y)
 # store the results
 results.append(scores)
 names.append(name)
 # summarize the performance along the way
 print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.title('Comparison of AdaBoost Model Accuracies with Different Depth of Trees')
pyplot.xlabel('depth')
pyplot.ylabel('Accuracy')
pyplot.show()


# In[69]:


from numpy import arange
from sklearn.ensemble import AdaBoostClassifier

def get_models():
    models = dict()
    # explore learning rates from 0.1 to 2 in 0.1 increments
    for i in arange(0.1, 2.1, 0.2):
        key = '%.3f' % i
        models[key] = AdaBoostClassifier(learning_rate=i)
    return models


# In[70]:


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
 # define the evaluation procedure
 cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 # evaluate the model and collect the results
 scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
 return scores


# In[71]:


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
 # evaluate the model
 scores = evaluate_model(model, X, y)
 # store the results
 results.append(scores)
 names.append(name)
 # summarize the performance along the way
 print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.xticks(rotation=45)
pyplot.title('Comparison of AdaBoost Model Accuracies with Different Learning Rate')
pyplot.xlabel('Learning Rate')
pyplot.ylabel('Accuracy')
pyplot.show()


# In[31]:


abc = AdaBoostClassifier(algorithm = 'SAMME.R', learning_rate= 0.1, n_estimators=500, random_state= 0)
abc.fit(X_train, y_train)



#################################################     Grid Test     ###############################################################################

#Best: 0.962810 using {'algorithm': 'SAMME.R', 'learning_rate': 0.1, 'n_estimators': 500, 'random_state': 0}


# In[32]:


y_pred = abc.predict(X_test)
y_pred
y_pred_training = abc.predict(X_train)
y_pred_training


# In[33]:


import time
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer


# Assuming rf_model is your trained RandomForestClassifier

feature_importances = abc.feature_importances_


# Get the feature names after OneHotEncoding
#encoded_columns = onehot_encoder.get_feature_names_out(features_to_encode)

# Extract feature names from the original X_train DataFrame (before encoding)
original_feature_names = X_train.columns

# Create a DataFrame to display feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': original_feature_names, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance
print(feature_importance_df)


# In[34]:


import matplotlib.pyplot as plt

# Assuming you have the feature_importance_df DataFrame
# Sort the DataFrame by importance in descending order (if not sorted already)
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Create a bar chart with smaller font size
plt.figure(figsize=(15, 14))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='green')
plt.xlabel('Importance Factor', fontsize=11)  # Adjust font size for x-axis label
plt.ylabel('Feature', fontsize=18)     # Adjust font size for y-axis label
plt.title('Feature Importance (Adaboost)', fontsize=18)  # Adjust font size for title
plt.tick_params(axis='both', which='major', labelsize=18)  # Adjust font size for tick labels
plt.show()


# In[172]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
accuracy_score(y_test, y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)
print(f"The accuracy (testing data) of the model is {round(accuracy_score(y_test,y_pred),3)*100} %")
print(f"The precision (testing data) of the model is {round(precision_score(y_test,y_pred),3)*100} %")
print(f"The recall (testing data) of the model is {round(recall_score(y_test,y_pred),3)*100} %")
print(f"The f1 Score (testing data) of the model is {round(f1_score(y_test,y_pred),3)*100} %")


# In[173]:


accuracy_score(y_train, y_pred_training)

print(f"The accuracy (training data)of the model is {round(accuracy_score(y_train,y_pred_training),3)*100} %")
print(f"The precision (training data) of the model is {round(precision_score(y_train,y_pred_training),3)*100} %")
print(f"The recall (training data) of the model is {round(recall_score(y_train,y_pred_training),3)*100} %")


# In[174]:


train_probs = abc.predict_proba(X_train)[:,1] 
probs = abc.predict_proba(X_test)[:, 1]
train_predictions = abc.predict(X_train)
#In general, an AUC of 0.5 suggests no discrimination (i.e., ability to diagnose patients with and without the disease or condition based on the test), 0.7 to 0.8 is considered acceptable, 0.8 to 0.9 is considered excellent, and more than 0.9 is considered outstanding.
print(f'Train Reciever Operating Characteristics Area Under Curve (AUC) Score: {roc_auc_score(y_train, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test, probs)}')


# In[175]:


import numpy as np
from sklearn.metrics import roc_curve
from numpy import sqrt

fpr, tpr, thresholds = roc_curve(y_test, probs)
gmeans = sqrt(tpr * (1 - fpr))
ix = np.argmax(gmeans)
best_threshold = thresholds[ix]
best_gmean = gmeans[ix]

print('Best Threshold = %.3f, G-Mean = %.3f' % (best_threshold, best_gmean))


# In[176]:


def evaluate_model(y_pred, probs,train_predictions, train_probs):
    baseline = {}
    baseline['recall']=recall_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    results = {}
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['roc'] = roc_auc_score(y_test, probs)
    train_results = {}
    train_results['recall'] = recall_score(y_train,       train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)
    for metric in ['recall', 'precision', 'roc']:  
          """""print(f'{metric.capitalize()} 
                 Baseline: {round(baseline[metric], 2)} 
                 Test: {round(results[metric], 2)} 
                 Train: {round(train_results[metric], 2)}')"""
     # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.plot(model_fpr, model_tpr, 'r', label = 'ROC AUC  Score: 0.97')
    plt.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves (AdaBoost)');
    plt.show();
evaluate_model(y_pred,probs,train_predictions,train_probs)


# In[177]:


import itertools

def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): # can change color     plt.figure(figsize = (10, 10))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, size = 14)
    plt.yticks(tick_marks, classes, rotation=90, size = 14)      
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.    
    # Label the plot    
    
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):    
        plt.text(j, i, format(cm[i, j], fmt), 
             fontsize = 20,
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")   
    
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)
    
# Let's plot it out
cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm, classes = ['0 - Current', '1 - Metro'],
                      title = 'Mode Confusion Matrix (AdaBoost)')


# In[363]:


#################################################     Grid Test     ###############################################################################


# define the model with a base estimator (e.g., DecisionTreeClassifier)
model = AdaBoostClassifier()

# define the parameter grid
grid = dict()
grid['n_estimators'] = [10, 50, 100, 500]  # Number of boosting stages
grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]  # Learning rate
grid['algorithm'] = ['SAMME', 'SAMME.R']  # Boosting algorithm
grid['random_state'] = [0, 42, 123]  # Random states for reproducibility

# define the cross-validation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(X_train, y_train)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


##################################Support Vector Machine##########################################################


# In[35]:


# Create an SVM object
from sklearn import svm
svm_model = svm.SVC(C=100, gamma=0.001, kernel='rbf', probability=True)

svm_model.fit(X_train, y_train)


#[CV 5/5] END ....C=100, gamma=0.001, kernel=rbf;, score=0.969 total time=   0.0s


# In[36]:


y_pred = svm_model.predict(X_test)
y_pred
y_pred_training = svm_model.predict(X_train)
y_pred_training


# In[37]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
accuracy_score(y_test, y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
print(f"The accuracy (testing data) of the model is {round(accuracy_score(y_test,y_pred),3)*100} %")
print(f"The precision (testing data) of the model is {round(precision_score(y_test,y_pred),3)*100} %")
print(f"The recall (testing data) of the model is {round(recall_score(y_test,y_pred),3)*100} %")
print(f"The f1 Score (testing data) of the model is {round(f1_score(y_test,y_pred),3)*100} %")


# In[38]:


accuracy_score(y_train, y_pred_training)

print(f"The accuracy (training data)of the model is {round(accuracy_score(y_train,y_pred_training),3)*100} %")
print(f"The precision (training data) of the model is {round(precision_score(y_train,y_pred_training),3)*100} %")
print(f"The recall (training data) of the model is {round(recall_score(y_train,y_pred_training),3)*100} %")


# In[39]:


train_probs = svm_model.predict_proba(X_train)[:,1] 
probs = svm_model.predict_proba(X_test)[:, 1]
train_predictions = svm_model.predict(X_train)
#In general, an AUC of 0.5 suggests no discrimination (i.e., ability to diagnose patients with and without the disease or condition based on the test), 0.7 to 0.8 is considered acceptable, 0.8 to 0.9 is considered excellent, and more than 0.9 is considered outstanding.
print(f'Train Reciever Operating Characteristics Area Under Curve (AUC) Score: {roc_auc_score(y_train, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test, probs)}')


# In[40]:


def evaluate_model(y_pred, probs,train_predictions, train_probs):
    baseline = {}
    baseline['recall']=recall_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    results = {}
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['roc'] = roc_auc_score(y_test, probs)
    train_results = {}
    train_results['recall'] = recall_score(y_train,       train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)
    for metric in ['recall', 'precision', 'roc']:  
          """""print(f'{metric.capitalize()} 
                 Baseline: {round(baseline[metric], 2)} 
                 Test: {round(results[metric], 2)} 
                 Train: {round(train_results[metric], 2)}')"""
     # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.plot(model_fpr, model_tpr, 'r', label = 'ROC AUC  Score: 0.94')
    plt.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves (SVM)');
    plt.show();
evaluate_model(y_pred,probs,train_predictions,train_probs)


# In[41]:


import itertools

def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): # can change color     plt.figure(figsize = (10, 10))
    
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 34)
    plt.colorbar(aspect=4)    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, size = 24)
    plt.yticks(tick_marks, classes, rotation=90, size = 24)      
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.    
    # Label the plot        
    
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):    
        plt.text(j, i, format(cm[i, j], fmt), 
             fontsize = 30,
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")   
    
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 30)
    plt.xlabel('Predicted label', size = 30)
    
# Let's plot it out
cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm, classes = ['0 - Current', '1 - Metro'],
                      title = 'Mode Confusion Matrix (SVM)')


# In[185]:


import time
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer


# Assuming rf_model is your trained RandomForestClassifier

feature_importances = svm_model.feature_importances_


# Get the feature names after OneHotEncoding
#encoded_columns = onehot_encoder.get_feature_names_out(features_to_encode)

# Extract feature names from the original X_train DataFrame (before encoding)
original_feature_names = X_train.columns

# Create a DataFrame to display feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': original_feature_names, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance
print(feature_importance_df)


# In[ ]:


#################################################     Grid Test     ###############################################################################

from sklearn import svm

model_s=svm.SVC()
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf','poly','linear']}  
  
grid = GridSearchCV(model_s, param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, y_train)


# In[29]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Create the dataset using the provided features
# I'll assume you have some target values (y) as well

# For simplicity, I'm generating a target array `y` with binary classification (0 and 1)
# Since the actual target values are not provided, I'll generate a synthetic one.
np.random.seed(42)
df = pd.DataFrame({
    'Diff_Bus&MRT_Fare': np.random.rand(100),
    'Access_EgressTT': np.random.rand(100),
    'distance_500m': np.random.rand(100),
    'TT_MRT_only': np.random.rand(100),
    'Total Waiting Time': np.random.rand(100),
    'Total Distance in km': np.random.rand(100),
    'Rapid pass/MRT pass for MRT (Y=1)': np.random.rand(100),
    'Ttmain_PV': np.random.rand(100),
    'Work_cen': np.random.rand(100),
    'TT_Bus_only': np.random.rand(100),
    'Inflation_conc': np.random.rand(100),
    'Social_influence': np.random.rand(100),
    'Ped_infra': np.random.rand(100),
    'Reliability': np.random.rand(100),
    'Safety': np.random.rand(100),
    'Family_incl': np.random.rand(100),
    'Walking_ben': np.random.rand(100),
    'Social_media': np.random.rand(100),
    'Family_income': np.random.rand(100),
    'Total Distance for access trip': np.random.rand(100),
    'Personal_Income': np.random.rand(100),
    'last educational degree': np.random.rand(100),
    'One_PV': np.random.rand(100),
    'Access_EgressFare': np.random.rand(100),
    'H_size': np.random.rand(100),
    'EarningFM': np.random.rand(100),
    'Very Flexible': np.random.rand(100),
    'EmpS_Full time Worker': np.random.rand(100),
    'Sex(F=0)': np.random.rand(100),
    'Somewhat Flexible': np.random.rand(100),
    'Age<35': np.random.rand(100),
    'EmpS_Student': np.random.rand(100),
    'Age<45': np.random.rand(100),
    'Greater_Than_One_PV': np.random.rand(100),
    'Emps_Others': np.random.rand(100),
    'EmpS_Part Time': np.random.rand(100),
    'Age<55': np.random.rand(100),
    'Age<65': np.random.rand(100),
    'EmpS_Hybrid (Can work at home or at office)': np.random.rand(100),
    'EmpS_Work at home (Full-time)': np.random.rand(100),
    'EmpS_Work at home (Part-time)': np.random.rand(100)
})

# Generating synthetic target variable (binary classification)
y = np.random.choice([0, 1], size=100)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Now, let's train the SVM model with different values of C and gamma
C_values = [1, 10, 100]
gamma_values = [0.1,0.01, 0.001]

# Define a function to plot the decision boundaries
def plot_svm_boundary(X, y, model, ax):
    # This is for synthetic 2D plotting (so I will reduce to 2 features)
    X = X[:, :2]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k')

    # Create a grid to evaluate the model
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Create subplots
fig, axes = plt.subplots(len(C_values), len(gamma_values), figsize=(10, 8))

# Train and plot SVM for different C and gamma values
for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        model = svm.SVC(C=C, gamma=gamma, kernel='rbf')
        model.fit(X_train[:, :2], y_train)  # Use only 2 features for 2D visualization
        
        # Plot decision boundary
        ax = axes[i, j]
        plot_svm_boundary(X_train, y_train, model, ax)
        ax.set_title(f'C={C}, gamma={gamma}')

plt.tight_layout()
plt.show()


# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Create the dataset using the provided features
np.random.seed(42)
df = pd.DataFrame({
    'Diff_Bus&MRT_Fare': np.random.rand(100) * 50,  # Scale up to fit the custom xlim
    'Access_EgressTT': np.random.rand(100) * 50,    # Scale up to fit the custom ylim
    'distance_500m': np.random.rand(100),
    'TT_MRT_only': np.random.rand(100) * 50,        # Scale up to fit the custom ylim
    'Total Waiting Time': np.random.rand(100),
    'Total Distance in km': np.random.rand(100),
    'Rapid pass/MRT pass for MRT (Y=1)': np.random.rand(100),
    'Ttmain_PV': np.random.rand(100),
    'Work_cen': np.random.rand(100),
    'TT_Bus_only': np.random.rand(100),
    'Inflation_conc': np.random.rand(100),
    'Social_influence': np.random.rand(100),
    'Ped_infra': np.random.rand(100),
    'Reliability': np.random.rand(100),
    'Safety': np.random.rand(100),
    'Family_incl': np.random.rand(100),
    'Walking_ben': np.random.rand(100),
    'Social_media': np.random.rand(100),
    'Family_income': np.random.rand(100),
    'Total Distance for access trip': np.random.rand(100),
    'Personal_Income': np.random.rand(100),
    'last educational degree': np.random.rand(100),
    'One_PV': np.random.rand(100),
    'Access_EgressFare': np.random.rand(100),
    'H_size': np.random.rand(100),
    'EarningFM': np.random.rand(100),
    'Very Flexible': np.random.rand(100),
    'EmpS_Full time Worker': np.random.rand(100),
    'Sex(F=0)': np.random.rand(100),
    'Somewhat Flexible': np.random.rand(100),
    'Age<35': np.random.rand(100),
    'EmpS_Student': np.random.rand(100),
    'Age<45': np.random.rand(100),
    'Greater_Than_One_PV': np.random.rand(100),
    'Emps_Others': np.random.rand(100),
    'EmpS_Part Time': np.random.rand(100),
    'Age<55': np.random.rand(100),
    'Age<65': np.random.rand(100),
    'EmpS_Hybrid (Can work at home or at office)': np.random.rand(100),
    'EmpS_Work at home (Full-time)': np.random.rand(100),
    'EmpS_Work at home (Part-time)': np.random.rand(100)
})

# Generating synthetic target variable (binary classification)
y = np.random.choice([0, 1], size=100)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Inverse transform the data to its original scale for visualization
X_unscaled = scaler.inverse_transform(X_train)

# Define the SVM parameters to vary
C_values = [1, 100]
gamma_values = [0.1, 0.001]

# Define a function to plot the decision boundaries with unstandardized axes, custom limits, and different markers
def plot_svm_boundary_custom(X, y, model, ax, feature_names):
    # This is for synthetic 2D plotting (so I will reduce to 2 features)
    X = X[:, :2]
    
    # Set different markers for class 0 and class 1
    for class_value, marker, color in zip([0, 1], ['o', 'x'], ['blue', 'red']):
        ax.scatter(X[y == class_value, 0], X[y == class_value, 1], 
                   marker=marker, color=color, label=f'Class {class_value}', s=50, edgecolors='k')
    
    # Add axis labels based on unscaled feature names
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])

    # Set custom axis limits
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)

    # Create a grid to evaluate the model
    xx, yy = np.meshgrid(np.linspace(0, 50, 50),
                         np.linspace(0, 50, 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins with legend
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # Add a legend for the dotted and solid lines
    custom_lines = [plt.Line2D([0], [0], color='k', linestyle='--', lw=2, label='Margins'),
                    plt.Line2D([0], [0], color='k', linestyle='-', lw=2, label='Decision Boundary')]
    ax.legend(handles=custom_lines, loc="lower left")

    # Add a legend for the classes
    ax.legend(loc="upper right")

# Create subplots
fig, axes = plt.subplots(len(C_values), len(gamma_values), figsize=(10, 8))

# Feature names to display on axes
feature_names = ['Diff_Bus&MRT_Fare', 'TT_MRT_only']

# Train and plot SVM for different C and gamma values using unscaled data
for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        model = svm.SVC(C=C, gamma=gamma, kernel='rbf')
        model.fit(X_train[:, :2], y_train)  # Use only 2 features for 2D visualization
        
        # Plot decision boundary with unstandardized axes and custom axis limits
        ax = axes[i, j]
        plot_svm_boundary_custom(X_unscaled, y_train, model, ax, feature_names)
        ax.set_title(f'C={C}, gamma={gamma}')

plt.tight_layout()
plt.show()


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Create the dataset using the provided features
np.random.seed(42)
df = pd.DataFrame({
    'Diff_Bus&MRT_Fare': np.random.rand(100) * 50,  # Scale up to fit the custom xlim
    'Access_EgressTT': np.random.rand(100) * 50,    # Scale up to fit the custom ylim
    'distance_500m': np.random.rand(100),
    'TT_MRT_only': np.random.rand(100) * 50,        # Scale up to fit the custom ylim
    'Total Waiting Time': np.random.rand(100),
    'Total Distance in km': np.random.rand(100),
    'Rapid pass/MRT pass for MRT (Y=1)': np.random.rand(100),
    'Ttmain_PV': np.random.rand(100),
    'Work_cen': np.random.rand(100),
    'TT_Bus_only': np.random.rand(100),
    'Inflation_conc': np.random.rand(100),
    'Social_influence': np.random.rand(100),
    'Ped_infra': np.random.rand(100),
    'Reliability': np.random.rand(100),
    'Safety': np.random.rand(100),
    'Family_incl': np.random.rand(100),
    'Walking_ben': np.random.rand(100),
    'Social_media': np.random.rand(100),
    'Family_income': np.random.rand(100),
    'Total Distance for access trip': np.random.rand(100),
    'Personal_Income': np.random.rand(100),
    'last educational degree': np.random.rand(100),
    'One_PV': np.random.rand(100),
    'Access_EgressFare': np.random.rand(100),
    'H_size': np.random.rand(100),
    'EarningFM': np.random.rand(100),
    'Very Flexible': np.random.rand(100),
    'EmpS_Full time Worker': np.random.rand(100),
    'Sex(F=0)': np.random.rand(100),
    'Somewhat Flexible': np.random.rand(100),
    'Age<35': np.random.rand(100),
    'EmpS_Student': np.random.rand(100),
    'Age<45': np.random.rand(100),
    'Greater_Than_One_PV': np.random.rand(100),
    'Emps_Others': np.random.rand(100),
    'EmpS_Part Time': np.random.rand(100),
    'Age<55': np.random.rand(100),
    'Age<65': np.random.rand(100),
    'EmpS_Hybrid (Can work at home or at office)': np.random.rand(100),
    'EmpS_Work at home (Full-time)': np.random.rand(100),
    'EmpS_Work at home (Part-time)': np.random.rand(100)
})

# Generating synthetic target variable (binary classification)
y = np.random.choice([0, 1], size=100)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the SVM parameters to vary
C_values = [1, 10, 100]
gamma_values = [0.1, 0.01, 0.001]

# Define a function to plot the decision boundaries with standardized axes and show the accuracy
def plot_svm_boundary_with_accuracy(X, y, model, ax, feature_names, accuracy):
    # This is for synthetic 2D plotting (so I will reduce to 2 features)
    X = X[:, :2]
    
    # Set different markers for class 0 and class 1
    for class_value, marker, color in zip([0, 1], ['o', 'x'], ['blue', 'red']):
        ax.scatter(X[y == class_value, 0], X[y == class_value, 1], 
                   marker=marker, color=color, label=f'Class {class_value}', s=50, edgecolors='k')
    
    # Add axis labels based on standardized feature names
    ax.set_xlabel(feature_names[0] + " (Standardized)")
    ax.set_ylabel(feature_names[1] + " (Standardized)")

    # Create a grid to evaluate the model
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins with legend
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # Add title showing C, gamma, and accuracy
    ax.set_title(f'C={model.C}, gamma={model.gamma}, accuracy={accuracy:.2f}')

    # Add a legend for the dotted and solid lines
    custom_lines = [plt.Line2D([0], [0], color='k', linestyle='--', lw=2, label='Margins'),
                    plt.Line2D([0], [0], color='k', linestyle='-', lw=2, label='Decision Boundary')]
    ax.legend(handles=custom_lines, loc="lower left")

# Create subplots with a larger figure size
fig, axes = plt.subplots(len(C_values), len(gamma_values), figsize=(15, 12))

# Feature names to display on axes
feature_names = ['Diff_Bus&MRT_Fare', 'TT_MRT_only']

# Train and plot SVM for different C and gamma values, showing accuracy
for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        model = svm.SVC(C=C, gamma=gamma, kernel='rbf')
        model.fit(X_train[:, :2], y_train)  # Use only 2 features for 2D visualization
        
        # Calculate accuracy
        accuracy = model.score(X_test[:, :2], y_test)
        
        # Plot decision boundary with standardized axes and accuracy
        ax = axes[i, j]
        plot_svm_boundary_with_accuracy(X_train, y_train, model, ax, feature_names, accuracy)

plt.tight_layout()
plt.show()


# In[44]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Create the dataset using the provided features
np.random.seed(42)
df = pd.DataFrame({
    'Diff_Bus&MRT_Fare': np.random.rand(500) * 50,  # Scale up to fit the custom xlim
    'Access_EgressTT': np.random.rand(500) * 50,    # Scale up to fit the custom ylim
    'distance_500m': np.random.rand(500) *50,
    'TT_MRT_only': np.random.rand(500) * 50,        # Scale up to fit the custom ylim
    'Total Waiting Time': np.random.rand(500) *50,
})

# Generating synthetic target variable (binary classification)
y = np.random.choice([0, 1], size=500)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the SVM parameters to vary
C_values = [1, 10, 100]
gamma_values = [0.1, 0.01, 0.001]

# Define a function to plot the decision boundaries with standardized axes and show the accuracy
def plot_svm_boundary_with_accuracy(X, y, model, ax, feature_names, accuracy):
    # This is for synthetic 2D plotting (so I will reduce to 2 features)
    X = X[:, :2]
    
    # Set different markers for class 0 and class 1
    for class_value, marker, color in zip([0, 1], ['o', 'o'], ['blue', 'darkorange']):
        ax.scatter(X[y == class_value, 0], X[y == class_value, 1], 
                   marker=marker, color=color, label=f'Class {class_value}', s=50, edgecolors='k')
    
    # Add axis labels based on standardized feature names
    ax.set_xlabel(feature_names[0],fontsize=35)
    ax.set_ylabel(feature_names[1],fontsize=35)

    # Create a grid to evaluate the model
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins with legend
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.9,
               linestyles=['--', '-', '--'])
    
    # Add title showing C, gamma, and accuracy
    ax.set_title(f'C={model.C}, gamma={model.gamma}, accuracy={accuracy:.2f}', fontsize=35)

    # Add a legend for the dotted and solid lines
    custom_lines = [plt.Line2D([0], [0], color='k', linestyle='--', lw=2, label='Margins'),
                    plt.Line2D([0], [0], color='k', linestyle='-', lw=2, label='Decision Boundary')]
    ax.legend(handles=custom_lines, loc="lower left")

# Create subplots with a larger figure size
fig, axes = plt.subplots(len(C_values), len(gamma_values), figsize=(30, 25))

# Feature names to display on axes
feature_names = ['Diff_Bus&MRT_Fare', 'TT_MRT_only']

# Train and plot SVM for different C and gamma values, showing accuracy
for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        model = svm.SVC(C=C, gamma=gamma, kernel='rbf')
        model.fit(X_train[:, :2], y_train)  # Use only 2 features for 2D visualization
        
        # Calculate accuracy
        accuracy = model.score(X_test[:, :2], y_test)
        
        # Plot decision boundary with standardized axes and accuracy
        ax = axes[i, j]
        plot_svm_boundary_with_accuracy(X_train, y_train, model, ax, feature_names, accuracy)

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


############Logit Model


# In[186]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[187]:


y_pred = lr.predict(X_test)
y_pred_training = lr.predict(X_train)


# In[188]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
accuracy_score(y_test, y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)
print(f"The accuracy (testing data) of the model is {round(accuracy_score(y_test,y_pred),3)*100} %")
print(f"The precision (testing data) of the model is {round(precision_score(y_test,y_pred),3)*100} %")
print(f"The recall (testing data) of the model is {round(recall_score(y_test,y_pred),3)*100} %")
print(f"f1_score (testing data) of the model is {round(f1_score(y_test,y_pred),3)*100} %")


# In[189]:


accuracy_score(y_train, y_pred_training)
precision_score(y_train, y_pred_training)
recall_score(y_train, y_pred_training)
f1_score(y_train, y_pred_training)

print(f"The accuracy (training data)of the model is {round(accuracy_score(y_train,y_pred_training),3)*100} %")
print(f"The precision (training data) of the model is {round(precision_score(y_train,y_pred_training),3)*100} %")
print(f"The recall (training data) of the model is {round(recall_score(y_train,y_pred_training),3)*100} %")
print(f"f1_score (training data) of the model is {round(f1_score(y_train,y_pred_training),3)*100} %")


# In[190]:


train_probs = lr.predict_proba(X_train)[:,1] 
probs = lr.predict_proba(X_test)[:, 1]
train_predictions = lr.predict(X_train)
#In general, an AUC of 0.5 suggests no discrimination (i.e., ability to diagnose patients with and without the disease or condition based on the test), 0.7 to 0.8 is considered acceptable, 0.8 to 0.9 is considered excellent, and more than 0.9 is considered outstanding.
print(f'Train Reciever Operating Characteristics Area Under Curve (AUC) Score: {roc_auc_score(y_train, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test, probs)}')


# In[191]:


#Plot ROC Curve
def evaluate_model(y_pred, probs,train_predictions, train_probs):
    baseline = {}
    baseline['recall']=recall_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    results = {}
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['roc'] = roc_auc_score(y_test, probs)
    train_results = {}
    train_results['recall'] = recall_score(y_train,       train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)
    for metric in ['recall', 'precision', 'roc']:  
          """""print(f'{metric.capitalize()} 
                 Baseline: {round(baseline[metric], 2)} 
                 Test: {round(results[metric], 2)} 
                 Train: {round(train_results[metric], 2)}')"""
     # Calculate false positive rates and true positive rates
    base_fpr, base_tpr,_ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr,_ = roc_curve(y_test, probs)
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.plot(model_fpr, model_tpr, 'r', label = 'ROC AUC  Score: 0.91')
    plt.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves (Logit)');
    plt.show();
  
evaluate_model(y_pred,probs,train_predictions,train_probs)


# In[192]:


#Best Threshold 
fpr, tpr, thresholds = roc_curve(y_test, probs)
from numpy import sqrt 
gmeans = sqrt(tpr* (1-fpr))
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))


# In[193]:


#Confusion Matrix
import itertools

def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): # can change color     plt.figure(figsize = (10, 10))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, size = 14)
    plt.yticks(tick_marks, classes, rotation=90, size = 14)      
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.    
    # Label the plot       
    
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):    
        plt.text(j, i, format(cm[i, j], fmt), 
             fontsize = 20,
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")   
    
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)
    
# Let's plot it out
cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm, classes = ['0 - Current', '1 - Metro'],
                     title = 'Mode Confusion Matrix (Logit)')


# In[137]:


# Assuming rf_model is your trained RandomForestClassifier

feature_importances = lr.feature_importances_


# Get the feature names after OneHotEncoding
#encoded_columns = onehot_encoder.get_feature_names_out(features_to_encode)

# Extract feature names from the original X_train DataFrame (before encoding)
original_feature_names = X_train.columns

# Create a DataFrame to display feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': original_feature_names, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance
print(feature_importance_df)


# In[196]:


# Assuming you have already trained models for Random Forest, AdaBoost, SVM, and Logistic Regression

# For Random Forest
rf_probs =  rf_classifier.predict_proba(X_test)[:, 1]  # Use [:, 1] to get the probabilities for the positive class

# For AdaBoost
ada_probs = abc.predict_proba(X_test)[:, 1]

# For SVM (you might need to set `probability=True` while training the SVM to get probabilities)
svm_probs = svm_model.predict_proba(X_test)[:, 1]

# For Logistic Regression
logit_probs = lr.predict_proba(X_test)[:, 1]

# Now call the ROC plotting function
plot_combined_roc_curves(y_test, rf_probs, ada_probs, svm_probs, logit_probs)


# In[198]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Function to plot ROC curves for multiple models with a colored legend
def plot_combined_roc_curves(y_test, rf_probs, ada_probs, svm_probs, logit_probs):
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.size'] = 14

    # ROC for Random Forest (Blue)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    rf_auc = roc_auc_score(y_test, rf_probs)
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})', color='blue')

    # ROC for AdaBoost (Green)
    ada_fpr, ada_tpr, _ = roc_curve(y_test, ada_probs)
    ada_auc = roc_auc_score(y_test, ada_probs)
    plt.plot(ada_fpr, ada_tpr, label=f'AdaBoost (AUC = {ada_auc:.2f})', color='green')

    # ROC for SVM (Orange)
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
    svm_auc = roc_auc_score(y_test, svm_probs)
    plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.2f})', color='orange')

    # ROC for Logistic Regression (Red)
    logit_fpr, logit_tpr, _ = roc_curve(y_test, logit_probs)
    logit_auc = roc_auc_score(y_test, logit_probs)
    plt.plot(logit_fpr, logit_tpr, label=f'Logistic Regression (AUC = {logit_auc:.2f})', color='red')

    # Plot baseline (black dashed line)
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline (AUC = 0.50)', color='black')

    # Customize plot
    plt.legend(loc='lower right')  # Place the legend at the bottom-right
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models')
    plt.grid(True)
    plt.show()

# Assuming rf_probs, ada_probs, svm_probs, logit_probs are the predicted probabilities for each model
# and y_test is the true test labels
plot_combined_roc_curves(y_test, rf_probs, ada_probs, svm_probs, logit_probs)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Total Distance in km'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "lightblue"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Total Distance in Km', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([0, 2, 4,6,8,10], [0, 2, 4,6,8,10])
# Set X-axis limit to 10
plt.xlim(0, 10)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[388]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Work_cen'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Work_cen', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([-2,-1, 0, 1, 2], [-2,-1, 0, 1, 2])
plt.xlim(-2,2)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[389]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Personal_Income'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Personal_Income', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([-2,-1, 0, 1, 2], [-2,-1, 0, 1, 2])
plt.xlim(-1,2)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[135]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

# Assuming df is your pandas DataFrame containing the TT_Bus_only column



var = 'TT_Bus_only'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('TT_Bus_only (min)', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([0, 10, 20, 30,40, 50, 60, 70, 80, 90, 100], [0, 10, 20, 30,40, 50, 60, 70, 80, 90, 100])
plt.xlim(0,100)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Add this line before plt.show() to remove the large X-axis ticks
plt.tick_params(axis='x', which='major', length=0)



# Show the plot
plt.show() 


# In[140]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'TT_MRT_only'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('TT_MRT_only (min)', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([0, 5, 10,15, 20,25, 30, 35,40, 50, 60, 70, 80, 90, 100], [0, 5, 10,15, 20,25, 30,35, 40, 50, 60, 70, 80, 90, 100])
plt.xlim(0,30)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[134]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Ttmain_PV'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Ttmain_PV (min)', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([0, 10, 20, 30,40, 50, 60, 70, 80, 90, 100], [0, 10, 20, 30,40, 50, 60, 70, 80, 90, 100])
plt.xlim(0,100)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[139]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed


var = 'Diff_Bus&MRT_Fare'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Diff_Bus&MRT_Fare', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.xlim(0,50)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[130]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Access_EgressTT'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Access_EgressTT (min)', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.xlim(0,40)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[136]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Access_EgressFare'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Access_EgressFare', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([0, 5, 10,15, 20,25, 30, 35,40, 50, 60, 70, 80, 90, 100], [0, 5, 10,15, 20,25, 30,35, 40, 50, 60, 70, 80, 90, 100])
plt.xlim(0,40)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[143]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Inflation_conc'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Inflation_conc', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([-2,-1, 0, 1, 2], [-2,-1, 0, 1, 2])
plt.xlim(-2,1)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[144]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Work_cen'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Work_cen', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([-2,-1, 0, 1, 2], [-2,-1, 0, 1, 2])
plt.xlim(-1,2)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[145]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Safety'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Safety', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([-2,-1, 0, 1, 2], [-2,-1, 0, 1, 2])
plt.xlim(-2,1)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[138]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Total Waiting Time'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Total Waiting Time (min)', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed


plt.xticks([0, 5, 10,15, 20,25, 30, 35,40, 50, 60, 70, 80, 90, 100], [0, 5, 10,15, 20,25, 30,35, 40, 50, 60, 70, 80, 90, 100])
plt.xlim(0,20)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[149]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Reliability'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Reliability', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([-2,-1, 0, 1, 2], [-2,-1, 0, 1, 2])
plt.xlim(-2,2)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[222]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Social_media'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Social_media', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([-2,-1, 0, 1, 2], [-2,-1, 0, 1, 2])
plt.xlim(-1,2)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[223]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Social_influence'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Social_influence', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([-2,-1, 0, 1, 2], [-2,-1, 0, 1, 2])
plt.xlim(-1,2)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[394]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Family_incl'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Family_incl', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([-2,-1, 0, 1, 2], [-2,-1, 0, 1, 2])
plt.xlim(-1.5,1.5)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[225]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Ped_infra'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel(' Ped_infra', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([-2,-1, 0, 1, 2], [-2,-1, 0, 1, 2])
plt.xlim(-1,2)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[398]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'distance_500m'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('distance_500m', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1])
#plt.xlim(0,5)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[ ]:





# In[ ]:





# In[367]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Rapid pass/MRT pass for MRT (Y=1)'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Rapid pass/MRT pass for MRT (Y=1)', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1])
#plt.xlim(0,5)
# Modify Y-axis tick labels in increments of 0.1
y_ticks = np.arange(-0.5, 0.6, 0.1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[148]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Walking_ben'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Walking_ben', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([-2,-1, 0, 1, 2], [-2,-1, 0, 1, 2])
plt.xlim(-1,2)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[152]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)

# Set the figure size and font size
mpl.rcParams['figure.figsize'] = [15, 9]  # Adjust the figure size as needed
mpl.rcParams['font.size'] = 30  # Adjust the font size as needed

var = 'Total Distance for access  trip'

pd_display = PartialDependenceDisplay.from_estimator(rf_classifier, X_train, [var], kind='both',
                                                    percentiles=(0, 1),centered=True, 
                                                    ice_lines_kw={"color": "grey"},
                                                    pd_line_kw={"color": "black", "lw": 4, 'linestyle': '--'})

plt.xlabel('Total Distance for access  trip (Km)', fontsize=30)  # Updated X-axis title
plt.ylabel('Probability of Shift', fontsize=30)  # Adjust the font size as needed
plt.xticks([0, 0.2, 0.4,0.6,0.8,1.0], [0, 0.2, 0.4,0.6,0.8,1.0])
# Set X-axis limit to 10
plt.xlim(0, 1)
plt.yticks(y_ticks)

# Set Y-axis range from 0.6 to -0.6
plt.ylim(-0.5, 0.5)

plt.legend(loc='upper left')
# Set the DPI to 1200 and save the plot to a file
plt.savefig('partial_dependence_plot.png', dpi=500)

# Show the plot
plt.show() 


# In[ ]:




