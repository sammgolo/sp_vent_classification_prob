
# coding: utf-8

# In[1]:

import pandas as pd
import time
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import collections
from math import sqrt
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn import cross_validation, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve

import seaborn as sns

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (12, 10)


# #### Load data

# In[2]:

train_path = "/Users/sgolomeke/Documents/Miscellaneous/Spring Venture/train_test.csv"
test_path = "/Users/sgolomeke/Documents/Miscellaneous/Spring Venture/holdout 2.csv"

train_data = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# #### Split training data into training (60%) and validation dataset (40%)

# In[76]:

##Split train data into train data and validation set
msk = np.random.rand(len(train_data)) < 0.6

train = train_data[msk]

validation_set = train_data[~msk]


# #### Lets explore the data

# In[4]:

train['target'].unique()


# In[5]:

print(train['target'].value_counts())

print(validation_set['target'].value_counts())


# In[7]:

train['Gender'].unique()


# In[8]:

train['Smoker'].unique()


# In[9]:

train['Emails'].unique()


# In[10]:

train['Neustar Result Code'].unique()


# In[11]:

train['Applicant State/Province'].unique()


# #### Clean Data

# In[77]:

pd.options.mode.chained_assignment = None

#Only one class has been specified. We replace nan with 0. Hence, we classify each row into either a 1 or 0
train.target = train.target.fillna(0)

train['Lead Source'] = train['Lead Source'].fillna(99)
train['Smoker'] = train['Smoker'].fillna('NA')
train['Emails'] = train['Emails'].fillna(99)
train['Gender'] = train['Gender'].fillna('NA')
train['Applicant State/Province'] = train['Applicant State/Province'].fillna('NA')
train['Applicant City'] = train['Applicant City'].fillna('NA')
train['Applicant Zip/Postal Code'] = train['Applicant Zip/Postal Code'].fillna(99)
train['Birthdate'] = train['Birthdate'].fillna('00000.0')
train['Neustar Result Code'] = train['Neustar Result Code'].fillna(99)



validation_set.target = validation_set.target.fillna(0)
validation_set['Lead Source'] = validation_set['Lead Source'].fillna(99)
validation_set['Smoker'] = validation_set['Smoker'].fillna('NA')
validation_set['Emails'] = validation_set['Emails'].fillna(99)
validation_set['Gender'] = validation_set['Gender'].fillna('NA')
validation_set['Applicant State/Province'] = validation_set['Applicant State/Province'].fillna('NA')
validation_set['Applicant City'] = validation_set['Applicant City'].fillna('NA')
validation_set['Applicant Zip/Postal Code'] = validation_set['Applicant Zip/Postal Code'].fillna(99)
validation_set['Birthdate'] = validation_set['Birthdate'].fillna('00000.0')
validation_set['Neustar Result Code'] = validation_set['Neustar Result Code'].fillna(99)


# In[43]:

import calendar, datetime
def dt(u): 
    return datetime.datetime.utcfromtimestamp(u)


# In[78]:

#we split bithdate timestamp into year, month and day
train['Birthdate'] = train['Birthdate'].astype(float)

train['birth_year'] = train['Birthdate'].apply(lambda x: dt(x).year)
train['birth_month'] = train['Birthdate'].apply(lambda x: dt(x).month)
train['birth_day'] = train['Birthdate'].apply(lambda x: dt(x).day)
#train['birth_hour'] = train['Birthdate'].apply(lambda x: dt(x).hour)
#train['birth_minute'] = train['Birthdate'].apply(lambda x: dt(x).minute)
#train['birth_second'] = train['Birthdate'].apply(lambda x: dt(x).second)


validation_set['Birthdate'] = validation_set['Birthdate'].astype(float)

validation_set['birth_year'] = validation_set['Birthdate'].apply(lambda x: dt(x).year)
validation_set['birth_month'] = validation_set['Birthdate'].apply(lambda x: dt(x).month)
validation_set['birth_day'] = validation_set['Birthdate'].apply(lambda x: dt(x).day)
#validation_set['birth_hour'] = validation_set['Birthdate'].apply(lambda x: dt(x).hour)
#validation_set['birth_minute'] = validation_set['Birthdate'].apply(lambda x: dt(x).minute)
#validation_set['birth_second'] = validation_set['Birthdate'].apply(lambda x: dt(x).second)


# In[79]:

train.Smoker = train.Smoker.replace('FALSE','No')
train.Smoker = train.Smoker.replace('TRUE','Yes')
train.Smoker = train.Smoker.replace('N','No')
train.Smoker = train.Smoker.replace('Y','Yes')
train.Smoker = train.Smoker.replace('1','Yes')
train.Smoker = train.Smoker.replace('OE State (Not Required)', '')
train.Smoker = train.Smoker.replace('TZT.Leads.Runtime.Domain.Models.Field','')

#train = train[train.Smoker != '']

validation_set.Smoker = validation_set.Smoker.replace('FALSE','No')
validation_set.Smoker = validation_set.Smoker.replace('TRUE','Yes')
validation_set.Smoker = validation_set.Smoker.replace('N','No')
validation_set.Smoker = validation_set.Smoker.replace('Y','Yes')
validation_set.Smoker = validation_set.Smoker.replace('1','Yes')
validation_set.Smoker = validation_set.Smoker.replace('OE State (Not Required)', '')
validation_set.Smoker = validation_set.Smoker.replace('TZT.Leads.Runtime.Domain.Models.Field','')

#validation_set = validation_set[validation_set.Smoker != '']


# In[80]:

train.Gender = train.Gender.replace('Male', 'M')
train.Gender = train.Gender.replace('Female', 'F')

#train = train[train.Gender != '']

validation_set.Gender = validation_set.Gender.replace('Male', 'M')
validation_set.Gender = validation_set.Gender.replace('Female', 'F')

#validation_set = validation_set[validation_set.Gender != '']


# In[17]:

print(train['Gender'].value_counts())
print ("number of missing Gender values: %f" %(train['Gender'].isnull().values.ravel().sum()))

print(train['Smoker'].value_counts())
print ("number of missing Smoker values: %f" %(train['Smoker'].isnull().values.ravel().sum()))


# In[18]:

print(validation_set['Gender'].value_counts())
print ("number of missing Gender values: %f" %(train['Gender'].isnull().values.ravel().sum()))

print(validation_set['Smoker'].value_counts())
print ("number of missing Smoker values: %f" %(validation_set['Smoker'].isnull().values.ravel().sum()))


# #### Descriptive Statistics of data

# In[19]:

# Statistical description

print(train.describe())


# #### Visualize Clean Data

# #### Categorical Variables

# In[20]:

cat = train[['Smoker','Emails','Gender','Applicant State/Province']]

cols = cat.columns

#Plot count plot for all attributes in a 4x2 grid
n_cols = 2
n_rows = 2
for i in range(n_rows):
    fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(12, 8))
    for j in range(n_cols):
        sns.countplot(x=cols[i*n_cols+j], data=train, ax=ax[j])


# #### Continuous Variables

# In[21]:

sns.violinplot(data=train, y="Lead Source")  
plt.show()


# #### Skewness

# In[22]:

# Skewness of the distribution

print(train.skew())
print(validation_set.skew())

# Values close to 0 show less ske


# #### Clearing Neustar Result Code is skewed, therefore we correct it using the boxcox approach

# In[81]:

from sklearn import preprocessing
from scipy.stats import skew
import numpy as np
from scipy.stats import boxcox

train['Neustar Result Code'] = preprocessing.scale(boxcox(train['Neustar Result Code']+1)[0])
validation_set['Neustar Result Code'] = preprocessing.scale(boxcox(validation_set['Neustar Result Code']+1)[0])


# In[24]:

print(train.skew())
print(validation_set.skew())


# In[25]:

## Factorize to replace nan's with -1

#train['Applicant State/Province'] = train['Applicant State/Province'].factorize()[0]
#train['Applicant City'] = train['Applicant City'].factorize()[0]
#train['birth_year'] = train['birth_year'].factorize()[0]
#train['Applicant Zip/Postal Code'] = train['Applicant Zip/Postal Code'].factorize()[0]

#validation_set['Applicant State/Province'] = validation_set['Applicant State/Province'].factorize()[0]
#validation_set['Applicant City'] = validation_set['Applicant City'].factorize()[0]
#validation_set['birth_year'] = validation_set['birth_year'].factorize()[0]
#validation_set['Applicant Zip/Postal Code'] = validation_set['Applicant Zip/Postal Code'].factorize()[0]


# #### Label encoding

# In[26]:

## Label for label encoding
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[82]:

## Label encode 'Applicant State/Province','Applicant City', 'birth_year'

train_labeled = MultiColumnLabelEncoder(columns = ['Applicant State/Province','Gender', 'Smoker']).fit_transform(train)

validation_set_labeled = MultiColumnLabelEncoder(columns = ['Applicant State/Province','Gender', 'Smoker']).fit_transform(validation_set)


# In[83]:

print(train_labeled.shape)
print(validation_set_labeled.shape)


# #### One hot encode

# In[84]:

## Detach dependent variable from data frame
y_train = train_labeled.target

y_validation_set = validation_set_labeled.target


# In[86]:

num_vars = ['Lead Source','birth_month','birth_day', 'birth_year']#, 'birth_hour', 'birth_minute', 'birth_second']

x_num_train = train_labeled[num_vars].as_matrix()

x_num_validation_set = validation_set_labeled[num_vars].as_matrix()


# In[87]:

x_cat_train = train_labeled.drop(num_vars + ['Unnamed: 0','System ID','Created Date Time','Birthdate','target', 'Applicant Zip/Postal Code', 'Applicant City'], axis = 1)

x_cat_validation_set = validation_set_labeled.drop(num_vars + ['Unnamed: 0','System ID','Created Date Time','Birthdate','target', 'Applicant Zip/Postal Code', 'Applicant City'], axis = 1)


# In[88]:

x_cat_train = x_cat_train.astype(str)
x_cat_validation_set = x_cat_validation_set.astype(str)


# In[89]:

x_cat_train_dict = x_cat_train.to_dict(orient = 'records')

x_cat_validation_set_dict = x_cat_validation_set.to_dict(orient = 'records')


# In[90]:

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
vec1 = DictVectorizer()


# In[91]:

vec_x_cat_train = vec.fit_transform(x_cat_train_dict).toarray()

vec_x_cat_validation_set = vec1.fit_transform(x_cat_validation_set_dict).toarray()


# In[92]:

print(vec_x_cat_train.shape)
print(vec_x_cat_validation_set.shape)


# In[37]:

vec.feature_names_


# In[38]:

vec1.feature_names_


# In[74]:

md_x_cat_train = np.delete(vec_x_cat_train, np.s_[19,42,43,44,46,47,54,64,67,75,76], 1)
md_x_cat_validation_set = np.delete(vec_x_cat_validation_set, np.s_[19,58,61,69,70], 1)


# In[75]:

print(md_x_cat_train.shape)
print(md_x_cat_validation_set.shape)
print(x_num_train.shape)
print(x_num_validation_set.shape)


# In[93]:

x_train = np.hstack((x_num_train, vec_x_cat_train))

x_validation_set = np.hstack((x_num_validation_set, vec_x_cat_validation_set))


# #### Dealing with the unbalanced nature of our dependent variable

# In[94]:

from imblearn.over_sampling import SMOTE

sm = SMOTE(kind='regular')
X_resampled_train, y_resampled_train = sm.fit_sample(x_train, y_train)
X_resampled_validation_set, y_resampled_validation_set = sm.fit_sample(x_validation_set, y_validation_set)


# In[95]:

from collections import Counter

print(Counter(y_resampled_train).keys())
print(Counter(y_resampled_validation_set).values())


# In[96]:

X_resampled_train.shape


# In[97]:

X_resampled_validation_set.shape


# ### Apply various classification machine learning algorithms to data

# In[98]:

ml_best = []


# In[ ]:

from sklearn.svm import SVC   
    
C = 173
gamma = 1.31e-5
shrinking = True

probability = True
verbose = False
    
clf_svc = SVC(C = C, gamma = gamma, shrinking = shrinking, probability = probability, verbose = verbose)
clf_svc.fit(X_resampled_train, y_resampled_train)
    
p_svc = clf_svc.predict_proba(X_resampled_validation_set)
    
svc_auc = AUC(y_resampled_validation_set, p_svc[:,1] )
print ("Support Vector Machine AUC: %f" %(svc_auc))
ml_best.append(['Support Vector Machine AUC', svc_auc])


# In[ ]:

from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier

n_estimators = 10

clf_svc_ensenble = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
clf_svc_ensenble.fit(X_resampled_train, y_resampled_train)

print ("Bagging SVC", end - start, clf.score(X,y))
p_svc_ensemble = clf.predict_proba(X)

svc_ensemble_auc = AUC(y_resampled_validation_set, p_svc_ensemble[:,1] )
print ("Ensemble Support Vector Machine AUC: %f" %(svc_ensemble_auc))
ml_best.append(['Ensemble Support Vector Machine AUC', svc_ensemble_auc])


# In[99]:

from sklearn.ensemble import RandomForestClassifier

n_trees = 100
max_features = int( round( sqrt( x_train.shape[1] ) * 2 ))# try more features at each split
max_features = 'auto'
verbose = 1
n_jobs = 1

clf_RandomForestClassifier = RandomForestClassifier( n_estimators = n_trees, max_features = max_features, verbose = verbose, n_jobs = n_jobs )
clf_RandomForestClassifier.fit(X_resampled_train, y_resampled_train)

p_RandomForestClassifier = clf_RandomForestClassifier.predict_proba(X_resampled_validation_set)

RandomForestClassifier_auc = AUC(y_resampled_validation_set, p_RandomForestClassifier[:,1] )
print ("RandomForestClassifier AUC: %f" %(RandomForestClassifier_auc))
ml_best.append(['RandomForestClassifier AUC', RandomForestClassifier_auc])


# In[100]:

from sklearn.ensemble import ExtraTreesClassifier

clf_extraTrees = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

clf_extraTrees.fit(X_resampled_train, y_resampled_train)

p_extraTrees = clf_extraTrees.predict_proba(X_resampled_validation_set)

extraTrees_auc = AUC(y_resampled_validation_set, p_extraTrees[:,1])
print ("extraTrees AUC: %f" %(extraTrees_auc))
ml_best.append(['extraTrees AUC', extraTrees_auc])


# In[101]:

from sklearn.tree import DecisionTreeClassifier

clf_DecisionTree = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)

clf_DecisionTree.fit(X_resampled_train, y_resampled_train)

p_DecisionTree = clf_DecisionTree.predict_proba(X_resampled_validation_set)

DecisionTree_auc = AUC(y_resampled_validation_set, p_DecisionTree[:,1])
print ("DecisionTree AUC: %f" %(DecisionTree_auc))
ml_best.append(['DecisionTree AUC', DecisionTree_auc])


# In[102]:

from sklearn.ensemble import AdaBoostClassifier

clf_AdaBoost = AdaBoostClassifier(n_estimators=100)

clf_AdaBoost.fit(X_resampled_train, y_resampled_train)

p_AdaBoost = clf_AdaBoost.predict_proba(X_resampled_validation_set)

AdaBoost_auc = AUC(y_resampled_validation_set, p_AdaBoost[:,1])
print ("AdaBoost AUC: %f" %(AdaBoost_auc))
ml_best.append(['AdaBoost AUC', AdaBoost_auc])


# In[103]:

from sklearn.ensemble import GradientBoostingClassifier

clf_GradientBoosting = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

clf_GradientBoosting.fit(X_resampled_train, y_resampled_train)

p_GradientBoosting = clf_GradientBoosting.predict_proba(X_resampled_validation_set)

GradientBoosting_auc = AUC(y_resampled_validation_set, p_GradientBoosting[:,1])
print ("GradientBoosting AUC: %f" %(GradientBoosting_auc))
ml_best.append(['GradientBoosting AUC', GradientBoosting_auc])


# In[104]:

from sklearn.neighbors import KNeighborsClassifier

clf_KNeighbors = KNeighborsClassifier(n_neighbors=7)

clf_KNeighbors.fit(X_resampled_train, y_resampled_train)

p_KNeighbors = clf_KNeighbors.predict_proba(X_resampled_validation_set)

KNeighbors_auc = AUC(y_resampled_validation_set, p_KNeighbors[:,1])
print ("KNeighbors AUC: %f" %(KNeighbors_auc))
ml_best.append(['KNeighbors AUC', KNeighbors_auc])


# In[105]:

from sklearn.naive_bayes import GaussianNB

clf_GaussianNB = GaussianNB()

clf_GaussianNB.fit(X_resampled_train, y_resampled_train)

p_GaussianNB = clf_GaussianNB.predict_proba(X_resampled_validation_set)

GaussianNB_auc = AUC(y_resampled_validation_set, p_GaussianNB[:,1])
print ("GaussianNB AUC: %f" %(GaussianNB_auc))
ml_best.append(['GaussianNB AUC', GaussianNB_auc])


# In[106]:

from sklearn.linear_model import LogisticRegression

clf_LogisticRegression = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                            intercept_scaling=1,class_weight=None, random_state=None, solver='liblinear',
                                            max_iter=100, multi_class='ovr',verbose=0, warm_start=False, n_jobs=1)

clf_LogisticRegression.fit(X_resampled_train, y_resampled_train)

p_LogisticRegression = clf_LogisticRegression.predict_proba(X_resampled_validation_set)

LogisticRegression_auc = AUC(y_resampled_validation_set, p_LogisticRegression[:,1])
print ("LogisticRegression AUC: %f" %(LogisticRegression_auc))
ml_best.append(['LogisticRegression AUC', LogisticRegression_auc])


# In[107]:

from sklearn.naive_bayes import BernoulliNB

clf_BernoulliNB = BernoulliNB()

clf_BernoulliNB.fit(X_resampled_train, y_resampled_train)

p_BernoulliNB = clf_BernoulliNB.predict_proba(X_resampled_validation_set)

BernoulliNB_auc = AUC(y_resampled_validation_set, p_BernoulliNB[:,1])
print ("BernoulliNB AUC: %f" %(BernoulliNB_auc))
ml_best.append(['BernoulliNB AUC', BernoulliNB_auc])


# In[108]:

auc_df = pd.DataFrame(ml_best, columns = ['ml_tool','auc'])


# In[109]:

auc_df


# In[110]:

md = np.array(auc_df.ml_tool)


# In[113]:

x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array(auc_df.auc)
my_xticks = md
plt.xticks(x,my_xticks, size = 'small')
plt.plot(x, y)
plt.show()


# ### Clearly RandomForest perform the best

# In[120]:

#test['Birthdate'] = test['Birthdate'].str.split('/')

import copy
test_tmp = copy.deepcopy(test)
#test_tmp = test.dropna()


# In[119]:

del test_tmp


# In[122]:

test_tmp['birth_month'] = test_tmp['Birthdate'].str.get(0)
test_tmp['birth_day'] = test_tmp['Birthdate'].str.get(1)
test_tmp['birth_year'] = test_tmp['Birthdate'].str.get(2) #.astype(int)


# In[139]:

test_tmp['birth_day'] = test_tmp['birth_day'].fillna('1')
test_tmp['birth_month'] = test_tmp['birth_month'].fillna('1')
test_tmp['birth_year'] = test_tmp['birth_year'].fillna('51').astype(int)


# In[140]:

test_1 = test_tmp[test_tmp['birth_year'] <= 9]
test_2 = test_tmp[(test_tmp['birth_year'] >= 10) & (test_tmp['birth_year'] <= 16)]
test_3 = test_tmp[(test_tmp['birth_year'] > 16)]

test_1['birth_year'] = '200' + test_1['birth_year'].astype(str)
test_2['birth_year'] = '20' + test_2['birth_year'].astype(str)
test_3['birth_year'] = '19' + test_3['birth_year'].astype(str)

test_3['birth_year'] = test_3['birth_year'].astype(str)

test_tmp = pd.concat([test_1, test_2, test_3], ignore_index = False)


# In[144]:

test_tmp['Lead Source'] = test_tmp['Lead Source'].fillna(99)
test_tmp['Smoker'] = test_tmp['Smoker'].fillna('NA')
test_tmp['Emails'] = test_tmp['Emails'].fillna(99)
test_tmp['Gender'] = test_tmp['Gender'].fillna('NA')
test_tmp['Applicant State/Province'] = test_tmp['Applicant State/Province'].fillna('NA')
test_tmp['Applicant City'] = test_tmp['Applicant City'].fillna('NA')
test_tmp['Applicant Zip/Postal Code'] = test_tmp['Applicant Zip/Postal Code'].fillna(99)
test_tmp['Neustar Result Code'] = test_tmp['Neustar Result Code'].fillna(99)


# In[145]:

test_tmp['Smoker'] = test_tmp['Smoker'].replace('FALSE','No')
test_tmp['Smoker'] = test_tmp['Smoker'].replace('TRUE','Yes')
test_tmp['Smoker'] = test_tmp['Smoker'].replace('N','No')
test_tmp['Smoker'] = test_tmp['Smoker'].replace('Y','Yes')
test_tmp['Smoker'] = test_tmp['Smoker'].replace('1','Yes')
test_tmp['Smoker'] = test_tmp['Smoker'].replace('OE State (Not Required)', '')


test_tmp.Gender = test_tmp.Gender.replace('Male', 'M')
test_tmp.Gender = test_tmp.Gender.replace('Female', 'F')


# In[147]:

test_tmp_labeled = MultiColumnLabelEncoder(columns = ['Applicant State/Province','Gender', 'Smoker']).fit_transform(test_tmp)


# In[148]:

x_num_test_tmp = test_tmp_labeled[num_vars].as_matrix()


# In[152]:

x_cat_test_tmp = test_tmp_labeled.drop(num_vars + ['Unnamed: 0','System ID','Created Date Time','Birthdate', 'Applicant Zip/Postal Code', 'Applicant City'], axis = 1)

x_cat_test_tmp = x_cat_test_tmp.astype(str)

x_cat_test_tmp_dict = x_cat_test_tmp.to_dict(orient = 'records')


# In[153]:

vec3 = DictVectorizer()

vec_x_cat_test_tmp = vec3.fit_transform(x_cat_test_tmp_dict).toarray()


# In[167]:

md_x_cat_test_tmp = np.delete(vec_x_cat_test_tmp, np.s_[19,57,60,67,68,69], 1)


# In[173]:

a = np.zeros((34178,11))


# In[174]:

x_test_tmp = np.hstack((x_num_test_tmp, md_x_cat_test_tmp, a))


# In[177]:

x_test_tmp.shape


# In[184]:

test.shape


# In[180]:

final_output = clf_RandomForestClassifier.predict_proba(x_test_tmp)


# In[181]:

final_output_pd = pd.DataFrame(data=final_output, columns=['prob_0','prob_1'])


# In[185]:

final_dataframe = test.join(final_output_pd)


# In[186]:

final_dataframe.head(500)


# In[187]:

final_dataframe.to_csv("/Users/sgolomeke/Documents/Miscellaneous/Spring Venture/holdout2.csv")


# In[ ]:



