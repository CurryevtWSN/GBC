#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
import joblib
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Prediction system for gallbladder cancer distant metastasis:a retrospective cohort study based on machine learning')
st.title('Prediction system for gallbladder cancer distant metastasis:a retrospective cohort study based on machine learning')

#%%set varixgbles selection
st.sidebar.markdown('## Varixgbles')

T = st.sidebar.selectbox("T Recode", ('T0','T1','T2','T3','T4','TX'), index = 4)
Age =  st.sidebar.slider("Age (year)", 5,95,value = 65, step = 1)
N = st.sidebar.selectbox("N Recode", ('N0','N1','N2','NX'), index = 2)
Median_household_income_inflation_adj_to_2021 = st.sidebar.selectbox("Median household income inflation(2021)", 
                                                                     ('<$50,000','$50,000-$60,000','$60,000-$70,000','>$70,000'), index = 2)
Rural_Urban_Continuum_Code = st.sidebar.selectbox("Rural Urban Continuum Code", 
                                                  ("<250,000 population","250,000-1 million population",">=1 million population","Others"),index = 1)
Grade_Recode = st.sidebar.selectbox("Grade", ("Well differentiated; Grade I","Moderately differentiated; Grade II",
                                              "Poorly differentiated; Grade III","Undifferentiated; aplastic; Grade IV"), index = 2)

Marital_status = st.sidebar.selectbox("Marital status", ('Single/Unmarried or Domestic Partner',"Married","Divorced/Separated/Widowed"), index = 1)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Shi-Nan Wu, Xiamen University')
#传入数据
map = {'Single/Unmarried or Domestic Partner':1,
       'Married':2,
       'Divorced/Separated/Widowed':3,
       'Well differentiated; Grade I':1, 
       'Moderately differentiated; Grade II':2, 
       "Poorly differentiated; Grade III":3,
       "Undifferentiated; aplastic; Grade IV":4,
       'T0':0,
       'T1':1,
       'T2':2,
       'T3':3,
       'T4':4,
       'TX':5,
       'N0':0,
       'N1':1,
       'N2':2,
       'NX':3,
       '<$50,000':0,
       '$50,000-$60,000':1,
       '$60,000-$70,000':2,
       '>$70,000':3,
       "<250,000 population":1,
       "250,000-1 million population":2,
       ">=1 million population":3,
       "Others":4
}

N =map[N]
T = map[T]
Rural_Urban_Continuum_Code = map[Rural_Urban_Continuum_Code]
Grade_Recode = map[Grade_Recode]
Median_household_income_inflation_adj_to_2021 = map[Median_household_income_inflation_adj_to_2021]
Marital_status = map[Marital_status]

# 数据读取，特征标注
#%%load model
xgb_model = joblib.load(r'D:\厦门大学\合作\刘荣强\胆囊癌SEER\xgb_model.pkl')

#%%load data
hp_train = pd.read_excel(r"D:\厦门大学\合作\刘荣强\胆囊癌SEER\data.xlsx", sheet_name="Sheet1")
features = ["T","Age","N","Median_household_income_inflation_adj_to_2021","Rural_Urban_Continuum_Code",
            "Grade_Recode","Marital_status"]

target = ["M"]
y = np.array(hp_train[target])
sp = 0.5

is_t = (xgb_model.predict_proba(np.array([[T,Age,N,Median_household_income_inflation_adj_to_2021,Rural_Urban_Continuum_Code,
            Grade_Recode,Marital_status]]))[0][1])> sp
prob = (xgb_model.predict_proba(np.array([[T,Age,N,Median_household_income_inflation_adj_to_2021,Rural_Urban_Continuum_Code,
            Grade_Recode,Marital_status]]))[0][1])*1000//1/10
    

if is_t:
    result = 'High Risk Group'
else:
    result = 'Low Risk Group'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Group':
        st.balloons()
    st.markdown('## Probability of High Risk group:  '+str(prob)+'%')
    #%%cbind users data
    col_names = features
    X_last = pd.DataFrame(np.array([[T,Age,N,Median_household_income_inflation_adj_to_2021,Rural_Urban_Continuum_Code,
                Grade_Recode,Marital_status]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0  
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = xgb_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of XGB model')
    fig, ax = plt.subplots(figsize=(12, 6))
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    # link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of XGB model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8))
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of XGB model')
    xgb_prob = xgb_model.predict(X)
    cm = confusion_matrix(y, xgb_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low risk', 'High risk'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of XGB model")
    disp1 = plt.show()
    st.pyplot(disp1)

