
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import altair as alt

st.write("""Welcome""")

def main():    
    data = pd.read_csv('C:/Harshad.Ambekar/personal/github/hackathon-intel-auto/dataset/dataset.csv')  
    main_page(data)

def main_page(data):
    file_name = st.selectbox("Select File", list(data['file_name']))
    get_cluster(file_name)

def get_cluster(file_name):
    result = pd.read_csv('C:/Harshad.Ambekar/personal/github/hackathon-intel-auto/dataset/result.csv')      
    df = result[result['file_name']==file_name]
    st.write(df['label'])
    
if __name__ == "__main__":
    main()