import time  # to simulate a real time data, time loop
import random
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import pyodbc
import streamlit as st
from sklearn.preprocessing import LabelEncoder
coder = LabelEncoder()
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
lgbmodel = LGBMClassifier(random_state=42, num_class = 3)
import joblib
##from Feature_engineering import *
import base64

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)


##load model

##model = joblib.load(r"/Users/mac/Desktop/Project/upsale/upsale.joblib")


st.markdown('''
    :blue[CHÀO MỪNG CÁC BẠN ĐẾN VỚI MÔ HÌNH UPSALE]''')


# top-level filters
col1, col2 = st.columns(2)

with col1:
    selected_customer = st.text_area("🗝️ Nhập tuổi khách hàng tại đây:")

with col2:
    EDA = st.button('🟢 Danh sách mô hình gợi ý bán:')
    if EDA :
        st.markdown('''Vui lòng đợi trong quá trình mô hình khởi động ...''')
        # Load data infer
        file_name_columns = '/Users/mac/Desktop/Project/upsale/SalesForCourse_quizz_table.csv'
        train_columns = pd.read_csv(file_name_columns)
        # load model
        file_name_model_trained = '/Users/mac/Desktop/Project/upsale/upsale.joblib'
        #model_trained = joblib.load('/Users/mac/Desktop/Project/upsale/upsale.joblib')
if st.button('🙍‍♂️ Nhấn để Khai thác chân dung khách hàng tại đây: 🔻'):
    # create three columns
    kpi = st.columns(1)

    file_name_columns = '/Users/mac/Desktop/Project/upsale/SalesForCourse_quizz_table.csv'
    train_columns = pd.read_csv(file_name_columns)

    kpi.metric(
        label="Quantity ⏳",
        value=round(train_columns['Quantity']),
    )