import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import warnings
import pickle
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

st.title("Covid Prediction & Classification")
st.write("------------------------------------")

def covid_predict():
    Breathing_Problem=st.selectbox('Breathing Problem', ['0', '1'])
    Fever=st.selectbox('Fever', ['0', '1'])
    Dry_Cough=st.selectbox('Dry Cough', ['0', '1'])
    Sore_throat=st.selectbox('Sore throat', ['0', '1'])
    Running_Nose=st.selectbox('Running Nose', ['0', '1'])
    Asthma=st.selectbox('Asthma', ['0', '1'])
    Chronic_Lung_Disease=st.selectbox('Chronic Lung Disease', ['0', '1'])
    Headache=st.selectbox('Headache', ['0', '1'])
    Heart_Disease=st.selectbox('Heart Disease', ['0', '1'])
    Diabetes=st.selectbox('Diabetes', ['0', '1'])
    Hyper_Tension=st.selectbox('Hyper Tension', ['0', '1'])
    Abroad_travel=st.selectbox('Abroad travel',['0', '1'])
    Contact_with_COVID_Patient=st.selectbox('Contact with COVID Patient', ['0', '1'])
    Attended_Large_Gathering=st.selectbox('Attended Large Gathering', ['0', '1'])
    Visited_Public_Exposed_Places=st.selectbox('Visited Public Exposed Places', ['0', '1'])
    Family_working_in_Public_Exposed=st.selectbox('Family working in Public Exposed Places',['0', '1'])
    Wearing_Masks=st.selectbox('Wearing Masks', ['0', '1'])
    Sanitization_from_Market=st.selectbox('Sanitization from Market',['0', '1'])

    return Breathing_Problem,Fever,Dry_Cough,Sore_throat,Running_Nose,Asthma,Chronic_Lung_Disease,Headache,Heart_Disease,Diabetes,Hyper_Tension,Abroad_travel,Contact_with_COVID_Patient,Attended_Large_Gathering,Visited_Public_Exposed_Places,Family_working_in_Public_Exposed,Wearing_Masks,Sanitization_from_Market

def severity_predict():
    Fever=st.selectbox('Fever', ['0', '1'])
    Tiredness=st.selectbox('Tiredness', ['0', '1'])
    Dry_Cough=st.selectbox('Dry-Cough', ['0', '1'])
    Difficulty_in_Breathing=st.selectbox('Difficulty-in-Breathing', ['0', '1'])
    Sore_Throat=st.selectbox('Sore-Throat', ['0', '1'])
    Pains=st.selectbox('Pains', ['0', '1'])
    Nasal_Congestion=st.selectbox('Nasal-Congestion', ['0', '1'])
    Runny_Nose=st.selectbox('Runny-Nose', ['0', '1'])
    Diarrhea=st.selectbox('Diarrhea', ['0', '1'])
    None_Experiencing=st.selectbox('None_Experiencing', ['0', '1'])

    return Fever,Tiredness,Dry_Cough,Difficulty_in_Breathing,Sore_Throat,Pains,Nasal_Congestion,Runny_Nose,Diarrhea,None_Experiencing

if(st.checkbox('Covid - 19 Prediction')):
    Breathing_Problem,Fever,Dry_Cough,Sore_throat,Running_Nose,Asthma,Chronic_Lung_Disease,Headache,Heart_Disease,Diabetes,Hyper_Tension,Abroad_travel,Contact_with_COVID_Patient,Attended_Large_Gathering,Visited_Public_Exposed_Places,Family_working_in_Public_Exposed,Wearing_Masks,Sanitization_from_Market=covid_predict()
    covid_model=pickle.load(open('/content/covid_prediction.pickle','rb'))
    x=[[Breathing_Problem,Fever,Dry_Cough,Sore_throat,Running_Nose,Asthma,Chronic_Lung_Disease,Headache,Heart_Disease,Diabetes,Hyper_Tension,Abroad_travel,Contact_with_COVID_Patient,Attended_Large_Gathering,Visited_Public_Exposed_Places,Family_working_in_Public_Exposed,Wearing_Masks,Sanitization_from_Market]]
    df=pd.DataFrame(x,index=[1],columns=['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat',
       'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache',
       'Heart Disease', 'Diabetes', 'Hyper Tension', 'Abroad travel',
       'Contact with COVID Patient', 'Attended Large Gathering',
       'Visited Public Exposed Places',
       'Family working in Public Exposed Places', 'Wearing Masks',
       'Sanitization from Market'])
    # df.replace(('Yes', 'No'), (1, 0), inplace=True)
    result=covid_model.predict(x)
    if(result==1):
        st.header('************* You have symptoms of COVID *********')
        st.subheader('*Immediately Perform these tasks*')
        st.subheader('1. Stay home (isolate)')
        st.subheader('2. Contact healthcare provider & Get tested for COVID-19')
        st.subheader('3. If your test is positive, contact your healthcare provider')     
        st.subheader(':::::::::::::::::::::::: Perform these tasks :::::::::::::::::::::::::::::')
        st.write('-> Get tested.\
                 -> Stay away from others, including staying apart from those living in your household\
                 -> Stay home except to get medical care \
                 -> Separate yourself from other people\
                 -> Monitor your symptoms regularly\
                 -> Call ahead before visiting your doctor\
                 -> Wear a well-fitting mask\
                 -> Cover your coughs and sneezes\
                 -> Clean your hands often\
                 -> Avoid sharing personal household items\
                 -> Clean surfaces in your home regularly\
                 -> Take steps to improve ventilation at home')
    else:
        st.header('No Covid predicted. Still be safe. Take Covid related measures.')
    
elif(st.checkbox('Covid - 19 Severity Prediction')):
    Fever,Tiredness,Dry_Cough,Difficulty_in_Breathing,Sore_Throat,Pains,Nasal_Congestion,Runny_Nose,Diarrhea,None_Experiencing=severity_predict()    
    covid_model=pickle.load(open('/content/severity_prediction.pickle','rb',))
    x=[[Fever,Tiredness,Dry_Cough,Difficulty_in_Breathing,Sore_Throat,Pains,Nasal_Congestion,Runny_Nose,Diarrhea,None_Experiencing]]
    df=pd.DataFrame(x,index=[1],columns=['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing','Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea','None_Experiencing'])
    # df.replace(('Yes', 'No'), (1, 0), inplace=True)
    severity_result=covid_model.predict(x)
    if(severity_result==3):
        st.header('************* You have Severe COVID *********')
        st.write('Take measures fast.')
        st.header('************* You have symptoms of COVID *********')
        st.subheader('*Immediately Perform these tasks*')
        st.subheader('1. Stay home (isolate)')
        st.subheader('2. Contact healthcare provider & Get tested for COVID-19')
        st.subheader('3. If your test is positive, contact your healthcare provider')     
        st.subheader(':::::::::::::::::::::::: Perform these tasks :::::::::::::::::::::::::::::')
        st.write('-> Get tested.\
                 -> Stay away from others, including staying apart from those living in your household\
                 -> Stay home except to get medical care \
                 -> Separate yourself from other people\
                 -> Monitor your symptoms regularly\
                 -> Call ahead before visiting your doctor\
                 -> Wear a well-fitting mask\
                 -> Cover your coughs and sneezes\
                 -> Clean your hands often\
                 -> Avoid sharing personal household items\
                 -> Clean surfaces in your home regularly\
                 -> Take steps to improve ventilation at home')
    elif(severity_result==2):
        st.header('You has Moderate Covid. Still be safe. Take Covid related measures.')
        st.write('-> Get tested.\
                 -> Stay away from others, including staying apart from those living in your household\
                 -> Stay home except to get medical care \
                 -> Separate yourself from other people\
                 -> Monitor your symptoms regularly\
                 -> Call ahead before visiting your doctor\
                 -> Wear a well-fitting mask\
                 -> Cover your coughs and sneezes\
                 -> Clean your hands often\
                 -> Avoid sharing personal household items\
                 -> Clean surfaces in your home regularly\
                 -> Take steps to improve ventilation at home')
    elif(severity_result==1):
        st.header('You have Mild Covid. Be safe. Take Covid related measures.')   
        st.write('-> Get tested.\
                 -> Stay away from others, including staying apart from those living in your household\
                 -> Stay home except to get medical care \
                 -> Separate yourself from other people\
                 -> Monitor your symptoms regularly\
                 -> Call ahead before visiting your doctor\
                 -> Wear a well-fitting mask\
                 -> Cover your coughs and sneezes\
                 -> Clean your hands often\
                 -> Avoid sharing personal household items\
                 -> Clean surfaces in your home regularly\
                 -> Take steps to improve ventilation at home')
    else:
        st.header('No Covid predicted. Still be safe. Take Covid related measures.')