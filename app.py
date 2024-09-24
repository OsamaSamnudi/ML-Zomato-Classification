import streamlit as st
import pandas as pd
import numpy as np

# Visualization Package
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.float_format = '{:,.2f}'.format

# Preprocessing Package
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from category_encoders import BinaryEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# Pipeline Package
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Model Deployment Package
import pickle
# import joblib

# Read Data

df_org = pd.read_csv('df_final.csv',index_col=0)
pd.options.display.float_format ='{:,.2f}'.format
st.set_page_config (page_title = 'Zomato' , layout = "wide" , page_icon = 'Z')
st.title("Zomato Classification")

# Sidebar
brief = st.sidebar.checkbox(":red[Brief about Project]")
Planning = st.sidebar.checkbox(":green[About Project]")
About_me = st.sidebar.checkbox(":green[About me]")

if brief:
    st.sidebar.header(":red[Brief about Project]")
    st.sidebar.write("""
    * This project involves analyzing the Zomato dataset to classify restaurants based on user ratings and other key factors. 
    * The primary goal is to predict whether a restaurant is "Good" or "Not Good" based on attributes such as the average rating (rate) and total number of votes (votes).
    * :red[So let us see the insights 👀.]
    """)
# Planning
if Planning :
    st.sidebar.header(":green[About Project]")
    st.sidebar.subheader ('Zomato Classification')
    st.sidebar.write("""
    * This project during my Intership @ Epsilon AI (https://www.epsiloneg.com/). 
    * Data Source:
        - Kaggle : https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants?authuser=0
    """)
    st.sidebar.write("""
    * Data Details:
        * Columns : 25 Features
        * Instance : 41205 Instance

    * Features selected in the deployment model:
         * online_order.
         * book_table.
         * phone.
         * approx_cost.
         * menu_item.
         * listed_type.
         * listed_city.
         * Total_North_Indian.
         * Total_South_Indian.
         * Total_East_Indian.
         * Total_West_Indian.
         * Total_International.
         * Total_Asian.
         * Total_Grill/BBQ/Bar_Food.
         * Total_Fast_Food.
         * Total_Beverages/Desserts.
         * Total_Healthy/Fusion.
         * Total_Bakery.
         * Rest_Fine Dining.
         * Rest_SweetOrBakery.
         * Rest_Drink_Oriented_Establishments.
         * Rest_Specialty_Shops.
         * Rest_Dining_Establishments.
         * Rest_Takeaway_and_Delivery.
    """)
# Aboutme
if About_me :
    st.sidebar.header(":green[About me]")
    st.sidebar.write("""
    - Osama SAAD
    - Certified Data Science - Epsilon AI
    - Infor EAM Master Data and Assets Control Section Head @Ibnsina Pharma
    - LinkedIn: 
        https://www.linkedin.com/in/ossama-ahmed-saad-525785b2
    - Github : 
        https://github.com/OsamaSamnudi
    """)

# Tabs :
# Corrected section for creating tabs
tab1, = st.tabs(['🧠Prediction🤖'])

with tab1:
    with st.container():
        st.header("🧠 Prediction🤖")
        col1, col2, col3 , col4 = st.columns([50, 50, 50 , 50])
        
        # Data Collection
        with col1:
            listed_city = st.selectbox("Select listed_city", ["Select"] + df_org.listed_city.unique().tolist())
            approx_cost = float(st.number_input("approx_cost", 0.0, 10000.0))
            Total_North_Indian = int(st.number_input("North_Indian", 0, 1000))
            Total_South_Indian = int(st.number_input("South_Indian", 0, 1000))
            Total_East_Indian = int(st.number_input("East_Indian", 0, 1000))
            Total_West_Indian = int(st.number_input("West_Indian", 0, 1000))
            Total_International = int(st.number_input("International", 0, 1000))

        with col2 :
            listed_type = st.selectbox("Select listed_type", ["Select"] + df_org.listed_type.unique().tolist())
            Total_Asian = int(st.number_input("Asian", 0, 1000))
            Total_Grill_BBQ_Bar_Food = int(st.number_input("Grill/BBQ/Bar Food", 0, 1000))
            Total_Fast_Food = int(st.number_input("Fast Food", 0, 1000))
            Total_Beverages_Desserts = int(st.number_input("Beverages/Desserts", 0, 1000))
            Total_Healthy_Fusion = int(st.number_input("Healthy/Fusion", 0, 1000))
            Total_Bakery = int(st.number_input("Bakery", 0, 1000))
        with col3:
            online_order = st.radio('online_order', [0, 1], horizontal=True)
            book_table = st.radio('book_table', [0, 1], horizontal=True)
            phone = st.radio('phone', [0, 1], horizontal=True)
            menu_item = st.radio('menu_item', [0, 1], horizontal=True)
            Rest_Fine_Dining = st.radio('Fine Dining', [0, 1], horizontal=True)

                    
        with col4 :
            Rest_SweetOrBakery = st.radio('Sweet Or Bakery', [0, 1], horizontal=True)
            Rest_Drink_Oriented_Establishments = st.radio('Drink Oriented Establishments', [0, 1], horizontal=True)
            Rest_Specialty_Shops = st.radio('Specialty Shops', [0, 1], horizontal=True)
            Rest_Dining_Establishments = st.radio('Dining Establishments', [0, 1], horizontal=True)
            Rest_Takeaway_and_Delivery = st.radio('Takeaway and Delivery', [0, 1], horizontal=True)
        
        with st.container():
            st.write('📌 Your Selected Data:')
            col4, col5, col6 = st.columns([5, 200, 2])
            with col5:
                N_data = pd.DataFrame({
                    'online_order': [online_order],
                    'book_table': [book_table],
                    'phone': [phone],
                    'approx_cost': [approx_cost],
                    'menu_item': [menu_item],
                    'listed_type': [listed_type],
                    'listed_city': [listed_city],
                    'Total_North_Indian': [Total_North_Indian],
                    'Total_South_Indian': [Total_South_Indian],
                    'Total_East_Indian': [Total_East_Indian],
                    'Total_West_Indian': [Total_West_Indian],
                    'Total_International': [Total_International],
                    'Total_Asian': [Total_Asian],
                    'Total_Grill/BBQ/Bar_Food': [Total_Grill_BBQ_Bar_Food],
                    'Total_Fast_Food': [Total_Fast_Food],
                    'Total_Beverages/Desserts': [Total_Beverages_Desserts],
                    'Total_Healthy/Fusion': [Total_Healthy_Fusion],
                    'Total_Bakery': [Total_Bakery],
                    'Rest_Fine Dining': [Rest_Fine_Dining],
                    'Rest_SweetOrBakery': [Rest_SweetOrBakery],
                    'Rest_Drink_Oriented_Establishments': [Rest_Drink_Oriented_Establishments],
                    'Rest_Specialty_Shops': [Rest_Specialty_Shops],
                    'Rest_Dining_Establishments': [Rest_Dining_Establishments],
                    'Rest_Takeaway_and_Delivery': [Rest_Takeaway_and_Delivery]
                })

                st.dataframe(N_data, use_container_width=True , height=100)

            if st.button('Predict'):
                Processor_Model = pickle.load(open('Processor.pkl' , 'rb'))
                xgb_clf_Model = pickle.load(open('xgb_clf_Model.pkl' , 'rb'))
                N_test = xgb_clf_Model.transform(N_data)
                Test_Pred = xgb_clf_Model.predict(N_test)
                if Test_Pred == 1:
                    Result = 'Good'
                    st.balloons()
                    st.markdown(f"""<span style="font-size:larger; color:green">**Patient Result : {Result}**</span>""" , unsafe_allow_html=True)
                else:
                    Result = 'Not Good'
                    st.markdown(f"""<span style="font-size:larger; color:red">**Patient Result : {Result}**</span>""" , unsafe_allow_html=True)
