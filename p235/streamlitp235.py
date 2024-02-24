#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import streamlit as st
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[92]:


st.title("model Deploymemt:bagging classifier")


# In[93]:


st.sidebar.header("User Input  Parameters")


# In[94]:


def somefun(lengen):
    for length in lengen:
        if not is_blahblah(length): return False


# In[95]:


def user_input_features():
    AGE = st.sidebar.number_input('insert the AGE')
    SEX = st.sidebar.selectbox('SEX',('1','0'))
    INF_ANAM = st.sidebar.selectbox('INF_ANAM',('0','1','2','3'))
    STENOK_AN = st.sidebar.selectbox("insert the STENOK_AN",('0','1','2','3','4','5','6'))
    FK_STENOK = st.sidebar.selectbox("Insert FK_STENOK",('0','1','2','3','4','7'))
    IBS_POST = st.sidebar.selectbox('IBS_POST',('0','1','2',))
    IBS_NASL = st.sidebar.selectbox('IBS_NASL',('1','0'))
    GB = st.sidebar.selectbox("GB",('1','0','2','3'))
    SIM_GIPERT = st.sidebar.selectbox("Insert the SIM_GIPERT",('0','1'))
    DLIT_AG = st.sidebar.selectbox("Insert DLIT_AG",('0','1','2','3','4','5','6','7'))
    HF = st.sidebar.selectbox('HF',('0','1','2','3','4'))
    nr11 = st.sidebar.selectbox('nr11',('1','0'))
    nr01 = st.sidebar.selectbox('nr01',('1','0'))
    nr02 = st.sidebar.selectbox("Insert the nr02",('0','1'))
    nr03 = st.sidebar.selectbox("Insert the nr03",('0','1'))
    nr04 = st.sidebar.selectbox("Insert the nr04",('0','1'))
    nr07 = st.sidebar.selectbox("Insert the nr07",('0','1'))
    nr08 = st.sidebar.selectbox("Insert the nr08",('0','1'))
    np01 = st.sidebar.selectbox("Insert the np01",('0','1'))
    np04 = st.sidebar.selectbox("Insert the np04",('0','1'))
    np05 = st.sidebar.selectbox("Insert the np05",('0','1'))
    np07 = st.sidebar.selectbox("Insert the np07",('0','1'))
    np08 = st.sidebar.selectbox("Insert the np08",('0','1'))
    np09 = st.sidebar.selectbox("Insert the np09",('0','1'))
    np10 = st.sidebar.selectbox("Insert the np10",('0','1'))
    endocr_01 = st.sidebar.selectbox("Insert the endocr_01",('0','1'))
    endocr_02 = st.sidebar.selectbox("Insert the endocr_02",('0','1'))
    endocr_03 = st.sidebar.selectbox("Insert the endocr_03",('0','1'))
    zab_leg_01 = st.sidebar.selectbox("Insert the zab_leg_01",('0','1'))
    zab_leg_02 = st.sidebar.selectbox("Insert the zab_leg_01",('0','1'))
    zab_leg_03 = st.sidebar.selectbox("Insert the zab_leg_01",('0','1'))
    zab_leg_04 = st.sidebar.selectbox("Insert the zab_leg_01",('0','1'))
    zab_leg_06 = st.sidebar.selectbox("Insert the zab_leg_01",('0','1'))
    S_AD_KBRIG = st.sidebar.number_input('insert S_AD_KBRIG')
    D_AD_KBRIG = st.sidebar.number_input('insert D_AD_KBRIG')
    S_AD_ORIT = st.sidebar.number_input('insert S_AD_ORIT')
    D_AD_ORIT = st.sidebar.number_input('insert D_AD_ORIT')
    O_L_POST = st.sidebar.selectbox("Insert the O_L_POST",('0','1'))
    K_SH_POST = st.sidebar.selectbox("Insert the K_SH_POST",('0','1'))
    MP_TP_POST = st.sidebar.selectbox("Insert the MP_TP_POST",('0','1'))
    SVT_POST = st.sidebar.selectbox("Insert the SVT_POST",('0','1'))
    GT_POST = st.sidebar.selectbox("Insert the GT_POST",('0','1'))
    FIB_G_POST = st.sidebar.selectbox("Insert the FIB_G_POST",('0','1'))
    ant_im = st.sidebar.selectbox("Insert the ant_im",('0','1','2','3','4'))
    lat_im = st.sidebar.selectbox("Insert the lat_im",('0','1','2','3','4'))
    inf_im = st.sidebar.selectbox("Insert the inf_im",('0','1','2','3','4'))
    post_im = st.sidebar.selectbox("Insert the post_im",('0','1','2','3','4'))
    IM_PG_P = st.sidebar.selectbox("Insert the IM_PG_P",('0','1'))
    ritm_ecg_p_01 = st.sidebar.selectbox("Insert the ritm_ecg_p_01",('0','1'))
    ritm_ecg_p_02= st.sidebar.selectbox("Insert the ritm_ecg_p_02",('0','1'))
    ritm_ecg_p_04= st.sidebar.selectbox("Insert the ritm_ecg_p_04",('0','1'))
    ritm_ecg_p_06= st.sidebar.selectbox("Insert the ritm_ecg_p_06",('0','1'))
    ritm_ecg_p_07= st.sidebar.selectbox("Insert the ritm_ecg_p_07",('0','1'))
    ritm_ecg_p_08= st.sidebar.selectbox("Insert the ritm_ecg_p_08",('0','1'))
    n_r_ecg_p_01= st.sidebar.selectbox("Insert the n_r_ecg_p_01",('0','1'))
    n_r_ecg_p_02= st.sidebar.selectbox("Insert the n_r_ecg_p_02",('0','1'))
    n_r_ecg_p_03= st.sidebar.selectbox("Insert the n_r_ecg_p_03",('0','1'))
    n_r_ecg_p_04= st.sidebar.selectbox("Insert the n_r_ecg_p_04",('0','1'))
    n_r_ecg_p_05= st.sidebar.selectbox("Insert the n_r_ecg_p_05",('0','1'))
    n_r_ecg_p_06= st.sidebar.selectbox("Insert the n_r_ecg_p_06",('0','1'))
    n_r_ecg_p_08= st.sidebar.selectbox("Insert the n_r_ecg_p_08",('0','1'))
    n_r_ecg_p_09= st.sidebar.selectbox("Insert the n_r_ecg_p_09",('0','1'))
    n_r_ecg_p_10= st.sidebar.selectbox("Insert the n_r_ecg_p_10",('0','1'))                                    
    n_p_ecg_p_01= st.sidebar.selectbox("Insert the n_p_ecg_p_01",('0','1'))                                    
    n_p_ecg_p_03= st.sidebar.selectbox("Insert the n_p_ecg_p_03",('0','1'))                                    
    n_p_ecg_p_04= st.sidebar.selectbox("Insert the n_p_ecg_p_04",('0','1'))                                    
    n_p_ecg_p_05= st.sidebar.selectbox("Insert the n_p_ecg_p_05",('0','1'))                                    
    n_p_ecg_p_06= st.sidebar.selectbox("Insert the n_p_ecg_p_06",('0','1'))                                    
    n_p_ecg_p_07= st.sidebar.selectbox("Insert the n_p_ecg_p_07",('0','1'))                                    
    n_p_ecg_p_08= st.sidebar.selectbox("Insert the n_p_ecg_p_08",('0','1'))                                    
    n_p_ecg_p_09= st.sidebar.selectbox("Insert the n_p_ecg_p_09",('0','1'))                                    
    n_p_ecg_p_10= st.sidebar.selectbox("Insert the n_p_ecg_p_10",('0','1'))                                    
    n_p_ecg_p_11= st.sidebar.selectbox("Insert the n_p_ecg_p_11",('0','1'))
    n_p_ecg_p_12= st.sidebar.selectbox("Insert the n_p_ecg_p_12",('0','1'))                                    
    fibr_ter_01= st.sidebar.selectbox("Insert the fibr_ter_01",('0','1'))                                    
    fibr_ter_02= st.sidebar.selectbox("Insert the fibr_ter_02",('0','1'))                                    
    fibr_ter_03= st.sidebar.selectbox("Insert the fibr_ter_03",('0','1'))                                    
    fibr_ter_05= st.sidebar.selectbox("Insert the fibr_ter_05",('0','1'))                                    
    fibr_ter_06= st.sidebar.selectbox("Insert the fibr_ter_06",('0','1'))                                    
    fibr_ter_07= st.sidebar.selectbox("Insert the fibr_ter_07",('0','1'))                                    
    fibr_ter_08= st.sidebar.selectbox("Insert the fibr_ter_08",('0','1'))                                    
    GIPO_K= st.sidebar.selectbox("Insert the GIPO_K",('0','1'))                                    
    GIPER_Na= st.sidebar.selectbox("Insert the GIPER_Na",('0','1'))                                    
    K_BLOOD= st.sidebar.number_input("Insert the K_BLOOD")
    Na_BLOOD= st.sidebar.number_input("Insert the Na_BLOOD")
    ALT_BLOOD= st.sidebar.number_input("Insert the ALT_BLOOD")
    AST_BLOOD= st.sidebar.number_input("Insert the AST_BLOOD")
    KFK_BLOOD= st.sidebar.number_input("Insert the KFK_BLOOD")
    L_BLOOD= st.sidebar.number_input("Insert the L_BLOOD")
    ROE= st.sidebar.number_input("Insert the ROE")
    TIME_B_S= st.sidebar.selectbox("Insert the TIME_B_S",('1','2','3','4','5','6','7','8','9'))
    R_AB_1_n= st.sidebar.selectbox("Insert the R_AB_1_n",('0','1','2','3'))
    R_AB_2_n= st.sidebar.selectbox("Insert the R_AB_2_n",('0','1','2','3'))
    R_AB_3_n= st.sidebar.selectbox("Insert the R_AB_3_n",('0','1','2','3'))
    NA_KB= st.sidebar.selectbox("Insert the NA_KB",('0','1'))
    NOT_NA_KB= st.sidebar.selectbox("Insert the NOT_NA_KB",('0','1'))
    LID_KB= st.sidebar.selectbox("Insert the LID_KB",('0','1'))
    NITR_S= st.sidebar.selectbox("Insert the NITR_S",('0','1'))
    NA_R_1_n= st.sidebar.selectbox("Insert the NA_R_1_n",('0,''1','2','3','4'))
    NA_R_2_n= st.sidebar.selectbox("Insert the NA_R_2_n",('0,''1','2','3'))
    NA_R_3_n= st.sidebar.selectbox("Insert the NA_R_3_n",('0','1','2'))
    NOT_NA_1_n= st.sidebar.selectbox("Insert the NOT_NA_1_n",('0','1','2','3','4'))
    NOT_NA_2_n= st.sidebar.selectbox("Insert the NOT_NA_2_n",('0','1','2','3'))
    NOT_NA_3_n= st.sidebar.selectbox("Insert the NOT_NA_3_n",('0','1','2'))
    LID_S_n= st.sidebar.selectbox("Insert the LID_S_n",('0','1'))
    B_BLOK_S_n= st.sidebar.selectbox("Insert the B_BLOK_S_n",('0','1'))
    ANT_CA_S_n= st.sidebar.selectbox("Insert the ANT_CA_S_n",('0','1'))
    GEPAR_S_n= st.sidebar.selectbox("Insert the GEPAR_S_n",('0','1'))
    ASP_S_n= st.sidebar.selectbox("Insert the ASP_S_n",('0','1'))
    TIKL_S_n= st.sidebar.selectbox("Insert the TIKL_S_n",('0','1'))
    TRENT_S_n= st.sidebar.selectbox("Insert the TRENT_S_n",('0','1'))
    FIBR_PREDS= st.sidebar.selectbox("Insert the FIBR_PREDS",('0','1'))
    Atrialfibrillation= st.sidebar.selectbox("Insert the Atrial fibrillation",('0','1'))
    PREDS_TAH= st.sidebar.selectbox("Insert the PREDS_TAH",('0','1'))
    JELUD_TAH= st.sidebar.selectbox("Insert the JELUD_TAH",('0','1'))
    FIBR_JELUD= st.sidebar.selectbox("Insert the FIBR_JELUD",('0','1'))
    A_V_BLOK= st.sidebar.selectbox("Insert the A_V_BLOK",('0','1'))
    OTEK_LANC= st.sidebar.selectbox("Insert the OTEK_LANC",('0','1'))    
    RAZRIV= st.sidebar.selectbox("Insert the RAZRIV",('0','1'))
    DRESSLER= st.sidebar.selectbox("Insert the DRESSLER",('0','1'))
    ZSN= st.sidebar.selectbox("Insert the ZSN",('0','1'))
    REC_IM= st.sidebar.selectbox("Insert the REC_IM",('0','1'))
    P_IM_STEN= st.sidebar.selectbox("Insert the P_IM_STEN",('0','1'))
                          
data =      {'SEX':SEX,
            'INF_ANAM':INF_ANAM,
            'STENOK_AN':STENOK_AN,
            'FK_STENOK':FK_STENOK,
            'IBS_POST':IBS_POST,
            'IBS_NASL':IBS_NASL,
            'GB':GB,
            'SIM_GIPERT':SIM_GIPERT,
            'DLIT_AG':DLIT_AG,
            'HF':HF,
            'nr11':nr11,
            'nr01':nr01,
            'nr02':nr02,
            'nr03':nr03,
            'nr04':nr04,
            'nr07':nr07,
            'nr08':nr08,
            'np01':np01,
            'np04':np04,
            'np05':np05,
            'np07':np07,
            'np08':np08,
            'np09':np09,
            'np10':np10,
            'endocr_01':endocr_01,
            'endocr_02':endocr_02,
            'endocr_03':endocr_03,
            'zab_leg_01':zab_leg_01,
            'zab_leg_02':zab_leg_02,
            'zab_leg_03':zab_leg_03,
            'zab_leg_04':zab_leg_04,
            'zab_leg_06':zab_leg_06,
            'S_AD_KBRIG':S_AD_KBRIG,
            'D_AD_KBRIG':D_AD_KBRIG,
            'S_AD_ORIT':S_AD_ORIT,
            'D_AD_ORIT':D_AD_ORIT,
            'O_L_POST':O_L_POST,
            'K_SH_POST':K_SH_POST,
            'MP_TP_POST':MP_TP_POST,
            'SVT_POST':SVT_POST,
            'GT_POST':GT_POST,
            'ant_im':ant_im,
            'lat_im':lat_im,
            'inf_im':inf_im,
            'post_im':post_im,
            'IM_PG_P':IM_PG_P,
            'ritm_ecg_p_01':ritm_ecg_p_01,
            'ritm_ecg_p_02':ritm_ecg_p_02,
            'ritm_ecg_p_04':ritm_ecg_p_04,
            'ritm_ecg_p_06':ritm_ecg_p_06,
            'ritm_ecg_p_07':ritm_ecg_p_07,
            'ritm_ecg_p_08':ritm_ecg_p_08,
            'n_r_ecg_p_01':n_r_ecg_p_01,
            'n_r_ecg_p_02':n_r_ecg_p_02,
            'n_r_ecg_p_03':n_r_ecg_p_03,
            'n_r_ecg_p_04':n_r_ecg_p_04,
            'n_r_ecg_p_05':n_r_ecg_p_05,
            'n_r_ecg_p_06':n_r_ecg_p_06,
            'n_r_ecg_p_08':n_r_ecg_p_08,
            'n_r_ecg_p_09':n_r_ecg_p_09,
            'n_r_ecg_p_10':n_r_ecg_p_10,
            'n_p_ecg_p_01':n_p_ecg_p_01,
            'n_p_ecg_p_03':n_p_ecg_p_03,
            'n_p_ecg_p_04':n_p_ecg_p_04,
            'n_p_ecg_p_05':n_p_ecg_p_05,
            'n_p_ecg_p_06':n_p_ecg_p_06,
            'n_p_ecg_p_07':n_p_ecg_p_07,
            'n_p_ecg_p_08':n_p_ecg_p_08,
            'n_p_ecg_p_09':n_p_ecg_p_09,
            'n_p_ecg_p_10':n_p_ecg_p_10,
            'n_p_ecg_p_11':n_p_ecg_p_11,
            'n_p_ecg_p_12':n_p_ecg_p_12,
            'fibr_ter_01':fibr_ter_01,
            'fibr_ter_02':fibr_ter_02,
            'fibr_ter_03':fibr_ter_03,
            'fibr_ter_05':fibr_ter_05,
            'fibr_ter_06':fibr_ter_06,
            'fibr_ter_07':fibr_ter_07,
            'fibr_ter_08':fibr_ter_08,
            'GIPO_K':GIPO_K,
            'K_BLOOD':K_BLOOD,
            'GIPER_Na':GIPER_Na,
            'Na_BLOOD':Na_BLOOD,
            'ALT_BLOOD':ALT_BLOOD,
            'AST_BLOOD':AST_BLOOD,
            'KFK_BLOOD':KFK_BLOOD,
            'L_BLOOD':L_BLOOD,
            'ROE':ROE,
            'TIME_B_S':TIME_B_S,
            'R_AB_1_n':R_AB_1_n,
            'R_AB_2_n':R_AB_2_n,
            'R_AB_3_n':R_AB_3_n,
            'NA_KB':NA_KB,
            'NOT_NA_KB':NOT_NA_KB,
            'LID_KB':LID_KB,
            'NITR_S':NITR_S,
            'NA_R_1_n':NA_R_1_n,
            'NA_R_2_n':NA_R_2_n,
            'NA_R_3_n':NA_R_3_n,
            'NOT_NA_1_n':NOT_NA_1_n,
            'NOT_NA_2_n':NOT_NA_2_n,
            'NOT_NA_3_n':NOT_NA_3_n,
            'LID_S_n':LID_S_n,
            'B_BLOK_S_n':B_BLOK_S_n,
            'ANT_CA_S_n':ANT_CA_S_n,
            'GEPAR_S_n':GEPAR_S_n,
            'ASP_S_n':ASP_S_n,
            'TIKL_S_n':TIKL_S_n,
            'TRENT_S_n':TRENT_S_n,
            'Atrialfibrillation':Atrialfibrillation,
            'PREDS_TAH':PREDS_TAH,
            'JELUD_TAH':JELUD_TAH,
            'FIBR_JELUD':FIBR_JELUD,
            'A_V_BLOK':A_V_BLOK,
            'OTEK_LANC':OTEK_LANC,
            'DRESSLER':DRESSLER,
            'ZSN':ZSN,
            'REC_IM':REC_IM,
            'P_IM_STEN':P_IM_STEN}


# In[96]:


features = pd.DataFrame(data,index = [0])
features 
df = user_input_features()
st.subheader('User Input parameters')
st.write(df) 


# In[97]:


# Read Data Set
df = pd.read_csv("C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr project\\Myocardial infarction complications.csv")
df


# In[98]:


df.isnull().sum().sum()


# In[99]:


#printing only 70 columns with highest percentage of Null values
display(round((df.isnull().sum() / (len(df.index)) * 100) , 2).sort_values(ascending = False).head(70).to_frame().rename({0:'%age'}, axis = 1).T.style.background_gradient('magma_r'))
print()
missing = (df.isnull().sum() / (len(df.index)) * 100).to_frame().reset_index().rename({0:'%age'}, axis = 1)


# In[100]:


std=df.std()
std


# In[101]:


df1 = df.fillna(std)
df1


# In[102]:


df1.drop('ID', axis=1, inplace=True)


# In[103]:


df1


# In[104]:


df1.columns


# In[105]:


x=df1.iloc[:,0:122]
x


# In[106]:


from sklearn.preprocessing import scale
# Normalizing the numerical data 
df2 = scale(df1)
df2


# In[107]:


# Feature Importance with Decision Trees Classifier
from sklearn.tree import  DecisionTreeClassifier
# feature extraction
model = DecisionTreeClassifier()
model.fit(x, y)
print(model.feature_importances_)


# In[108]:


df3=pd.DataFrame(df2)
df3


# In[109]:


DTreeclassifier_data = pd.DataFrame(model.feature_importances_ , columns=["Score"])
dfcolumns_dtc=pd.DataFrame(df.columns)
dtc_features_rank=pd.concat([dfcolumns_dtc,DTreeclassifier_data],axis=1)
dtc_features_rank.columns=['DTC Features','Score']
dtc_features_rank


# In[110]:


dtc_features_rank['DTC Features'][dtc_features_rank['Score'] > 0.008197]


# In[111]:


DC_Data = df1[['AGE' ,'STENOK_AN' , 'FK_STENOK' ,'IBS_POST' ,'ZSN_A', 'nr_04' , 'S_AD_KBRIG' , 'D_AD_KBRIG',
'S_AD_ORIT' , 'D_AD_ORIT' , 'K_SH_POST' ,'ant_im' , 'lat_im' , 'ritm_ecg_p_07' , 'n_r_ecg_p_04' , 'n_p_ecg_p_10' , 
'n_p_ecg_p_12' ,'K_BLOOD' ,'NA_BLOOD' ,'ALT_BLOOD','AST_BLOOD' ,'L_BLOOD' , 'ROE' , 'TIME_B_S' , 'R_AB_1_n' , 'R_AB_3_n' 
,'NA_KB','NOT_NA_KB', 'NITR_S' , 'NA_R_1_n' , 'GEPAR_S_n' ,'RAZRIV' ,'DRESSLER' ,'ZSN','REC_IM','LET_IS']]
DC_Data


# In[ ]:





# In[112]:


y=df1['LET_IS']


# In[113]:


from sklearn.model_selection  import  train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25, random_state = 0)


# In[114]:


import imblearn
from imblearn.over_sampling import SMOTE
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=100, test_size=0.3)
oversample = SMOTE()
X_train, Y_train = oversample.fit_resample(X_train, Y_train)
from collections import Counter
counter = Counter(Y_train)
print(counter)


# In[115]:


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


# In[116]:


# Bagged Decision Trees for Classification
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
model_bc = BaggingClassifier(base_estimator=cart,n_estimators=10,random_state=8)
results_bc = cross_val_score(model_bc, x_train, y_train, cv=10)
print(results_bc.mean())


# In[117]:


model_bc.fit(x_train , y_train)


# In[118]:


y_pred_bc = model_bc.predict(x_test)
y_pred_bc_train = model_bc.predict(x_train)


# In[119]:


print(accuracy_score(y_test , y_pred_bc))
print(accuracy_score(y_train , y_pred_bc_train))


# In[ ]:




