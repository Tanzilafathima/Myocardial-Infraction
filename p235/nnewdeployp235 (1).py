#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


st.title("model Deploymemt:bagging classifier")


# In[105]:


st.sidebar.header('User Input Parameters')
AGE = st.sidebar.number_input('insert the AGE')
SEX = st.sidebar.selectbox('SEX',('1','0'))
INF_ANAM = st.sidebar.selectbox('INF_ANAM',('0','1','2','3'))
STENOK_AN = st.sidebar.selectbox("insert the STENOK_AN",('0','1','2','3','4','5','6'))
FK_STENOK = st.sidebar.selectbox("Insert FK_STENOK",('0','1','2','3','4','7'))
IBS_POST = st.sidebar.selectbox('IBS_POST',('0','1','2',))
IBS_NASL = st.sidebar.selectbox('IBS_NASL',('1','0'))
nr04 = st.sidebar.selectbox("Insert the nr04",('0','1'))
D_AD_KBRIG = st.sidebar.number_input('insert D_AD_KBRIG')
S_AD_KBRIG = st.sidebar.number_input('insert S_AD_KBRIG')
S_AD_ORIT = st.sidebar.number_input('insert S_AD_ORIT')
D_AD_ORIT = st.sidebar.number_input('insert D_AD_ORIT')
K_SH_POST = st.sidebar.selectbox("Insert the K_SH_POST",('0','1'))
ant_im = st.sidebar.selectbox("Insert the ant_im",('0','1','2','3','4'))
lat_im = st.sidebar.selectbox("Insert the lat_im",('0','1','2','3','4'))
ritm_ecg_p_07= st.sidebar.selectbox("Insert the ritm_ecg_p_07",('0','1'))
n_r_ecg_p_04= st.sidebar.selectbox("Insert the n_r_ecg_p_04",('0','1'))
n_p_ecg_p_10= st.sidebar.selectbox("Insert the n_p_ecg_p_10",('0','1'))
n_p_ecg_p_12= st.sidebar.selectbox("Insert the n_p_ecg_p_12",('0','1'))
K_BLOOD= st.sidebar.number_input("Insert the K_BLOOD")
Na_BLOOD= st.sidebar.number_input("Insert the Na_BLOOD")
ALT_BLOOD= st.sidebar.number_input("Insert the ALT_BLOOD")
AST_BLOOD= st.sidebar.number_input("Insert the AST_BLOOD")
KFK_BLOOD= st.sidebar.number_input("Insert the KFK_BLOOD")
L_BLOOD= st.sidebar.number_input("Insert the L_BLOOD")
ROE= st.sidebar.number_input("Insert the ROE")
TIME_B_S= st.sidebar.selectbox("Insert the TIME_B_S",('1','2','3','4','5','6','7','8','9'))
R_AB_1_n= st.sidebar.selectbox("Insert the R_AB_1_n",('0','1','2','3'))
R_AB_3_n= st.sidebar.selectbox("Insert the R_AB_3_n",('0','1','2','3'))
NA_KB= st.sidebar.selectbox("Insert the NA_KB",('0','1'))
NOT_NA_KB= st.sidebar.selectbox("Insert the NOT_NA_KB",('0','1'))
NITR_S= st.sidebar.selectbox("Insert the NITR_S",('0','1'))
NA_R_1_n= st.sidebar.selectbox("Insert the NA_R_1_n",('0,''1','2','3','4'))
GEPAR_S_n= st.sidebar.selectbox("Insert the GEPAR_S_n",('0','1'))
RAZRIV= st.sidebar.selectbox("Insert the RAZRIV",('0','1'))
DRESSLER= st.sidebar.selectbox("Insert the DRESSLER",('0','1'))
ZSN= st.sidebar.selectbox("Insert the ZSN",('0','1'))
REC_IM= st.sidebar.selectbox("Insert the REC_IM",('0','1'))
data = {"AGE":AGE,
        'SEX':SEX,
        'INF_ANAM':INF_ANAM,
        'STENOK_AN':STENOK_AN,
        'FK_STENOK':FK_STENOK,
        'IBS_POST':IBS_POST,
        'IBS_NASL':IBS_NASL,
        'nr04':nr04,
        'S_AD_KBRIG':S_AD_KBRIG,
        'D_AD_KBRIG':D_AD_KBRIG,
        'S_AD_ORIT':S_AD_ORIT,
        'D_AD_ORIT':D_AD_ORIT,
        'K_SH_POST':K_SH_POST,
        'SVT_POST':SVT_POST,
        'GT_POST':GT_POST,
        'ant_im':ant_im,
        'lat_im':lat_im,
        'ritm_ecg_p_07':ritm_ecg_p_07,
        'n_r_ecg_p_04':n_r_ecg_p_04,
        'n_p_ecg_p_10':n_p_ecg_p_10,
        'n_p_ecg_p_12':n_p_ecg_p_12,
        'K_BLOOD':K_BLOOD,
        'Na_BLOOD':Na_BLOOD,
        'ALT_BLOOD':ALT_BLOOD,
        'AST_BLOOD':AST_BLOOD,
        'KFK_BLOOD':KFK_BLOOD,
        'L_BLOOD':L_BLOOD,
        'ROE':ROE,
        'TIME_B_S':TIME_B_S,
        'R_AB_1_n':R_AB_1_n,
        'R_AB_3_n':R_AB_3_n,
        'NA_KB':NA_KB,
        'NOT_NA_KB':NOT_NA_KB,
        'NITR_S':NITR_S,
        'NA_R_1_n':NA_R_1_n,
        'GEPAR_S_n':GEPAR_S_n,
        'RAZRIV':RAZRIV,
        'DRESSLER':DRESSLER,
        'ZSN':ZSN,
        'REC_IM':REC_IM,
       }


# In[106]:


features = pd.DataFrame(data,index = [0])
features 
df = ('user_input_features')
st.subheader('User Input parameters')
st.write(df)


# In[107]:


df = pd.read_csv("C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr project\\Myocardial infarction complications.csv")
df


# In[108]:


df.isnull().sum().sum()


# In[109]:


display(round((df.isnull().sum() / (len(df.index)) * 100) , 2).sort_values(ascending = False).head(70).to_frame().rename({0:'%age'}, axis = 1).T.style.background_gradient('magma_r'))
print()
missing = (df.isnull().sum() / (len(df.index)) * 100).to_frame().reset_index().rename({0:'%age'}, axis = 1)


# In[110]:


std=df.std()
std


# In[111]:


df1 = df.fillna(std)
df1


# In[112]:


df1.drop('ID', axis=1, inplace=True)


# In[85]:


df1.columns


# In[86]:


x=df1.iloc[:,0:122]
x
y=df1['LET_IS']


# In[87]:


from sklearn.preprocessing import scale
# Normalizing the numerical data 
df2 = scale(df1)
df2


# In[88]:


# Feature Importance with Decision Trees Classifier
from sklearn.tree import  DecisionTreeClassifier
# feature extraction
model = DecisionTreeClassifier()
model.fit(x, y)
print(model.feature_importances_)


# In[89]:


df3=pd.DataFrame(df2)
df3


# In[90]:


DTreeclassifier_data = pd.DataFrame(model.feature_importances_ , columns=["Score"])
dfcolumns_dtc=pd.DataFrame(df.columns)
dtc_features_rank=pd.concat([dfcolumns_dtc,DTreeclassifier_data],axis=1)
dtc_features_rank.columns=['DTC Features','Score']
dtc_features_rank


# In[91]:


dtc_features_rank['DTC Features'][dtc_features_rank['Score'] > 0.008197]


# In[92]:


DC_Data = df1[['AGE' ,'STENOK_AN' , 'FK_STENOK' ,'IBS_POST' ,'ZSN_A', 'nr_04' , 'S_AD_KBRIG' , 'D_AD_KBRIG',
'S_AD_ORIT' , 'D_AD_ORIT' , 'K_SH_POST' ,'ant_im' , 'lat_im' , 'ritm_ecg_p_07' , 'n_r_ecg_p_04' , 'n_p_ecg_p_10' , 
'n_p_ecg_p_12' ,'K_BLOOD' ,'NA_BLOOD' ,'ALT_BLOOD','AST_BLOOD' ,'L_BLOOD' , 'ROE' , 'TIME_B_S' , 'R_AB_1_n' , 'R_AB_3_n' 
,'NA_KB','NOT_NA_KB', 'NITR_S' , 'NA_R_1_n' , 'GEPAR_S_n' ,'RAZRIV' ,'DRESSLER' ,'ZSN','REC_IM','LET_IS']]
DC_Data


# In[93]:


from sklearn.model_selection  import  train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25, random_state = 0)


# In[94]:


import imblearn
from imblearn.over_sampling import SMOTE
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=100, test_size=0.3)
oversample = SMOTE()
X_train, Y_train = oversample.fit_resample(X_train, Y_train)
from collections import Counter
counter = Counter(Y_train)
print(counter)


# In[95]:


from sklearn.decomposition import PCA
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=100, test_size=0.3)
oversample = SMOTE()
X_train, Y_train = oversample.fit_resample(X_train, Y_train)
pca = PCA(n_components=5)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
pca.fit(X_test)
X_test_pca = pca.transform(X_test)
pcaxtrain=pd.DataFrame(X_train_pca)
pcaxtest=pd.DataFrame(X_test_pca)


# In[96]:


# Bagged Decision Trees for Classification
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
model_bc = BaggingClassifier(base_estimator=cart,n_estimators=10,random_state=8)
results_bc = cross_val_score(model_bc, x_train, y_train, cv=10)
print(results_bc.mean())


# In[97]:


model_bc.fit(x_train , y_train)


# In[98]:


y_pred_bc = model_bc.predict(x_test)
y_pred_bc_train = model_bc.predict(x_train)


# In[99]:


print(accuracy_score(y_test , y_pred_bc))
print(accuracy_score(y_train , y_pred_bc_train))


# In[ ]:





# In[ ]:





# In[ ]:




