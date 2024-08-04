#!/usr/bin/env python
# coding: utf-8

# # Business Problem Understanding
# 
# #### Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.
# 
# #### Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings ('ignore')


# In[3]:


df=pd.read_csv("D:\\data science\\LoanData.csv")
df.head()


# In[4]:


df.info()


# In[5]:


# Lets check the column names present in the dataset
df.columns


# ## Data Understanding 

# Loan_ID: Unique Loan ID
# 
# Gender: Male/Female
# 
# Married:Applicant married 
# 
# Dependents:Number of dependents
# 
# Education: Applicant Education
# 
# Self_Employed:Whether the applicant is Self employed
# 
# ApplicantIncome:Applicant income
# 
# CoapplicantIncome:coapplicant income 
# 
# LoanAmount:Loan amount in thousand
#        
# Loan_Amount_Term:Term of Loan in month
# 
# Credit_History:credict history meets guidelines 
# 
# Property_Area: Urban / Semi Urban/Rural
# 
# ## . Loan_Status: Loan approved # Target variable
#       

# In[6]:


df['Loan_ID'].nunique()


# In[7]:


df.drop(columns=['Loan_ID'],inplace=True)


# In[8]:


df['Gender'].unique()


# In[9]:


df['Gender'].value_counts()


# In[10]:


df['Married'].unique()


# In[11]:


df['Married'].value_counts()


# In[12]:


df['Dependents'].unique()


# In[13]:


df['Dependents'].value_counts()


# In[14]:


df['Education'].value_counts()


# In[15]:


df['Self_Employed'].unique()


# In[16]:


df['Self_Employed'].value_counts()


# In[17]:


df['ApplicantIncome'].unique()


# In[18]:


df['CoapplicantIncome'].unique()


# In[19]:


df['LoanAmount'].unique()


# In[20]:


df['Loan_Amount_Term'].unique()


# In[21]:


df['Loan_Amount_Term'].value_counts()


# In[22]:


df['Credit_History'].unique()


# In[23]:


df['Credit_History'] = df['Credit_History'].replace({1:"good",0:"bad"})


# In[24]:


df['Credit_History'].unique()


# In[25]:


df['Credit_History'].value_counts()


# In[26]:


df['Property_Area'].value_counts()


# In[27]:


df['Loan_Status'].unique()


# In[28]:


df['Loan_Status'].value_counts()


# In[29]:


continous = ['ApplicantIncome','CoapplicantIncome','LoanAmount']

discrete_categorical = ['Gender','Married','Education','Self_Employed','Credit_History','Property_Area','Loan_Status']

discrete_count = ['Dependents','Loan_Amount_Term']


# ##  Exploratory Data Analysis (EDA)
# 
# 
# ### for continous Variables

# In[30]:


df [continous].describe()


# In[31]:


plt.rcParams['figure.figsize'] = (18,8)

plt.subplot(1,3,1)
sns.histplot(df['ApplicantIncome'],kde=True)

plt.subplot(1,3,2)
sns.histplot(df['CoapplicantIncome'],kde=True)

plt.subplot(1,3,3)
sns.histplot(df['LoanAmount'],kde=True)

plt.suptitle('Univariate Analysis on Numerical Columns')
plt.show()


# In[32]:


df[continous].skew()


# In[33]:


sns.pairplot(df[continous])
plt.show()


# In[34]:


sns.heatmap(df[continous].corr(),annot=True)
plt.show()


# In[35]:


#lets visualize the outliers using Box plot

plt.subplot(1,3,1)
sns.boxplot(df['ApplicantIncome'])

plt.subplot(1,3,2)
sns.boxplot(df['CoapplicantIncome'])

plt.subplot(1,3,3)
sns.boxplot(df['LoanAmount'])

plt.suptitle('outliers in the df')
plt.show()


# ##  for Discrete Variables

# In[36]:


df [discrete_categorical].describe()


# In[69]:


plt.rcParams["figure.figsize"] = (18, 8)

plt.subplot(2, 3, 1)
sns.countplot(df=df, x="Gender")

plt.subplot(2, 3, 2)
sns.countplot(df=df, x="Married")

plt.subplot(2, 3, 3)
sns.countplot(df=df, x="Self_Employed")

plt.subplot(2, 3, 4)
sns.countplot(df=df, x="Property_Area")

plt.subplot(2, 3, 5)
sns.countplot(df=df, x="Education")

plt.subplot(2, 3, 6)
sns.countplot(df=df, x="Loan_Status")

plt.suptitle("Univariate Analysis on Categorical Columns")
plt.show()


# ##  Data Preparation

# ##  Data Preparation

# In[37]:


df["Income"] = df ["ApplicantIncome"] + df ['CoapplicantIncome']

df.drop(columns=['ApplicantIncome','CoapplicantIncome'],inplace = True)


# ## Modifying the wrong data

# In[38]:


df['Dependents'] = df['Dependents'].replace({'3+':3})


# ## Missing Value Treatment

# In[39]:


#checking no.of missing values
df.isnull() .sum()


# In[40]:


# checking precentage of missing values
df.isnull().sum()/len(df)*100


# In[41]:


df = df.dropna(subset=["Income","LoanAmount","Loan_Amount_Term","Credit_History"])


# In[42]:


# count variable replace with 0
df['Dependents'] = df['Dependents'].fillna(0)


# In[43]:


# categorical variables replace with mode 
df ['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df ['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df ['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])


# In[44]:


df.isnull().sum()


# ## outliers treatment
# 
# 
# ## Encoding

# In[45]:


df['Gender'] = df['Gender'].map({'Male':1,'Female':0}).astype('int')
df['Married'] = df['Married'].map({'Yes':1,'No':0}).astype('int')
df['Education']=df['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
df['Self_Employed']=df['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
df['Property_Area']=df['Property_Area'].map({'Rural':0,'Semiurban':1,'Urban':2}).astype('int')
df['Credit_History']=df['Credit_History'].map({'good':1,'bad':0}).astype('int')
df['Loan_Status']=df['Loan_Status'].map({'Y':1,'N':0}).astype('int')


# ## data type conversion

# In[46]:


df['Dependents'] = df ['Dependents'].astype('int')
df['Loan_Amount_Term'] = df ['Loan_Amount_Term'].astype('int')


# ## transformations

# In[47]:


df[['Income','LoanAmount']].skew()


# In[48]:


# Lets apply boxcox transformation to remove skewness
from scipy.stats import boxcox
df['Income'],a = boxcox(df['Income'])
df['LoanAmount'],c=boxcox(df['LoanAmount'])


# In[49]:


df[['Income','LoanAmount']].skew()


# In[50]:


df['Loan_Amount_Term'] = df['Loan_Amount_Term']/12


# # X&y

# In[51]:


X = df.drop('Loan_Status',axis=1)
y = df['Loan_Status']


# ## identyfy the best random state number 

# In[53]:


Train =  []
Test = []
CV = []

for i in range (0,101):
    from sklearn.model_selection import train_test_split
    X_train , X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=i)

    from sklearn.linear_model import LogisticRegression
    log_default = LogisticRegression()
    log_default.fit(X_train,y_train)

    ypred_train = log_default.predict(X_train)
    ypred_test =  log_default.predict(X_test)

    from sklearn.metrics import accuracy_score
    Train.append(accuracy_score(y_train,ypred_train))
    Test.append(accuracy_score(y_test,ypred_test))
    
    from sklearn.model_selection import cross_val_score
    CV.append(cross_val_score(log_default,X_train,y_train,cv=5,scoring="accuracy").mean())
    
em = pd.DataFrame({"Train":Train,"Test":Test,"CV":CV})   
gm = em[(abs(em['Train']-em['Test'])<=0.5) & (abs(em['Test']-em['CV'])<=0.05)]
rs = gm[gm["CV"]==gm["CV"].max()].index.to_list()[0]
print("best random_state number:",rs)


# ## train-test split

# In[54]:


from sklearn.model_selection import train_test_split
X_train , X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state=70) 


# ##  Machine Learning Modelling & Evaluation
# 

# ## 1.Logistic Regression

# In[55]:


from sklearn.linear_model import LogisticRegression 
log_model = LogisticRegression()
log_model.fit(X_train,y_train)

ypred_train = log_model.predict(X_train)
ypred_test = log_model.predict(X_test)

print("Train Accuracy :",accuracy_score(y_train,ypred_train))
print("Cross validation score:",cross_val_score(log_model,X_train,y_train,cv=5,scoring = "accuracy"))
print("Test Accuracy :",accuracy_score(y_test,ypred_test))


# ##  2.KNN

# In[56]:


from sklearn.neighbors import KNeighborsClassifier

estimator = KNeighborsClassifier()
param_grid = {'n_neighbors':list(range(1,50))}

from sklearn.model_selection import GridSearchCV
knn_grid = GridSearchCV(estimator,param_grid,scoring='accuracy',cv=5)
knn_grid.fit(X_train,y_train)

knn_model = knn_grid.best_estimator_

ypred_train = knn_model.predict(X_train)
ypred_test = knn_model.predict(X_test)

print ("Train Accuracy :",accuracy_score(y_train,ypred_train))
print ("Cross validation score:",cross_val_score(knn_model,X_train,y_train,cv=5,scoring="accuracy"))
print ("Test Accuracy :",accuracy_score(y_test,ypred_test))


# ## 3.Support Vector Machine (SVM)

# In[56]:


from sklearn.svm import SVC

estimator = SVC()
param_grid = {'C':[0.01,0.1,1],'kernel':['linear','rbf','sigmoid','poly']}

from sklearn.model_selection import GridSearchCV
svm_grid = GridSearchCV(estimator,param_grid,scoring='accuracy',cv=5)
svm_grid.fit(X_train,y_train)

svm_model = svm_grid.best_estimator_

ypred_train = svm_model.predict(X_train)
ypred_test = svm_model.predict(X_test)

print ("Train Accuracy:",accuracy_score(y_train,ypred_train))
print ("Cross validation score:",cross_val_score(knn_model,X_train,y_train,cv=5,scoring="accuracy"))
print ("Test Accuracy :",accuracy_score(y_test,ypred_test))


# ## 4. Decision Tree

# In[57]:


from sklearn.tree import DecisionTreeClassifier
estimator = DecisionTreeClassifier(random_state=rs)
param_grid = {"criterion":["gini","entropy"],
             "max_depth":list(range(1,16))}

from sklearn.model_selection import GridSearchCV
dt_grid = GridSearchCV(estimator,param_grid,scoring='accuracy',cv=5)
dt_grid.fit(X_train,y_train)

#identify the best model
dt = dt_grid.best_estimator_

#identify the importance of each feature
dt_fi = dt.feature_importances_

#identify the features where the feature importance is grater than 0
index = [i for i,x in enumerate(dt_fi) if x>0]

#create with best model & with importance features 
X_train_dt = X_train.iloc[:,index]
X_test_dt = X_test.iloc[:,index]

#train with best model & with important features
dt.fit(X_train_dt,y_train)

ypred_train = dt.predict(X_train_dt)
ypred_tes = dt.predict(X_test_dt)

#Evaluate the best model
print ("Train Accuracy:",accuracy_score(y_train,ypred_train))
print ("Cross validation score:",cross_val_score(knn_model,X_train,y_train,cv=5,scoring="accuracy"))
print ("Test Accuracy :",accuracy_score(y_test,ypred_test))


# In[58]:


X_train_dt


# ## 5.Random Forest Classifier

# In[59]:


from sklearn.ensemble import RandomForestClassifier
estimator = RandomForestClassifier(random_state=rs)
param_grid = {'n_estimators':list(range(1,15))}

from sklearn.model_selection import GridSearchCV
rf_grid = GridSearchCV(estimator,param_grid,scoring = 'accuracy',cv=5)
rf_grid.fit(X_train,y_train)

rf = rf_grid.best_estimator_
rf_fi = rf.feature_importances_

index = [i for i,x in enumerate(rf_fi) if x>0]

X_train_rf = X_train.iloc[:,index]
X_test_rf = X_test.iloc[:,index]

rf.fit(X_train_rf,y_train)

ypred_train = rf.predict(X_train_rf)
ypred_test = rf.predict(X_test_rf)

print ("Train Accuracy:",accuracy_score(y_train,ypred_train))
print ("Cross validation score:",cross_val_score(knn_model,X_train,y_train,cv=5,scoring="accuracy"))
print ("Test Accuracy :",accuracy_score(y_test,ypred_test))


# ## 6. AdaBoost Classifier

# In[60]:


from sklearn.ensemble import AdaBoostClassifier
estimator = AdaBoostClassifier (random_state = rs)
param_grid = {'n_estimators': list(range(1,51))}


from sklearn.model_selection import GridSearchCV
ab_grid = GridSearchCV(estimator , param_grid,scoring = "accuracy",cv=5)
ab_grid.fit(X_train,y_train)


ab = ab_grid.best_estimator_
ab_fi = ab.feature_importances_


index =[i for i,x in enumerate (ab_fi) if x>0]


X_train_ab = X_train.iloc[:,index]
X_test_ab = X_test.iloc[:,index]


ab.fit(X_train_ab,y_train)


ypred_train = ab.predict(X_train_ab)
ypred_test = ab.predict(X_test_ab)


print ("Train Accuracy:",accuracy_score(y_train,ypred_train))
print ("Cross validation score:",cross_val_score(knn_model,X_train,y_train,cv=5,scoring="accuracy"))
print ("Test Accuracy :",accuracy_score(y_test,ypred_test))


# ## 7.Gradient Boost Classifier

# In[62]:


from sklearn.ensemble import GradientBoostingClassifier
estimator = GradientBoostingClassifier (random_state = rs)
param_grid = {"n_estimators":list(range(1,10)),"learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.10]}

from sklearn.model_selection import GridSearchCV
gb_grid = GridSearchCV(estimator,param_grid, scoring = "accuracy",cv=5)
gb_grid.fit(X_train,y_train)


gb = gb_grid.best_estimator_
gb_fi = gb.feature_importances_

index =[i for i,x in enumerate(gb_fi) if x>0]

X_train_gb = X_train.iloc[:,index]
X_test_gb = X_test.iloc[:,index]

gb.fit(X_train_gb,y_train)

ypred_train = gb.predict(X_train_gb)
ypred_test = gb.predict(X_test_gb)

print("Train Accuracy:",accuracy_score(y_train,ypred_train))
print("Cross validation score:",cross_val_score(log_model,X_train,y_train,cv=5,scoring="accuracy").mean())
print("Test Accuracy:",accuracy_score(y_test,ypred_test))


# ## 7. XGBoost

# In[62]:


pip install XGBoost


# In[64]:


from xgboost import XGBClassifier
estimator = XGBClassifier(random_state=rs)
param_grid = {"n_estimators":[10,20,40,100],'max_depth':[3,4,5],'gamma':[0,0.15,0.3,0.5,1]}

from sklearn.model_selection import GridSearchCV
xgb_grid = GridSearchCV (estimator,param_grid, scoring = 'accuracy',cv=5)
xgb_grid.fit(X_train,y_train)

xgb = xgb_grid.best_estimator_

xgb_fi = xgb.feature_importances_

index = [i for i,x in enumerate(xgb_fi) if x>0]

X_train_xgb = X_train.iloc[:,index]
X_test_xgb = X_test.iloc[:,index]

xgb.fit(X_train_xgb,y_train)

ypred_train = xgb.predict(X_train_xgb)
ypred_test = xgb.predict(X_test_xgb)

print ("Train Accuracy:",accuracy_score(y_train,ypred_train))
print ("Cross validation score:",cross_val_score(knn_model,X_train,y_train,cv=5,scoring="accuracy"))
print ("Test Accuracy :",accuracy_score(y_test,ypred_test))


# In[ ]:




