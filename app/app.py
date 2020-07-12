
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import altair as alt

st.write("""Auto ML""")

st.sidebar.header('User Input Parameters')

def main():    
    var_uploaded_file = user_input_features()
    df = upload_file(var_uploaded_file)
    data_exploration(df)  
    data_overview(df)
    model_selection(df)
          
    
def data_overview(data):  
    if st.sidebar.checkbox('Overview'):
        st.write(data)

def data_exploration(data):        
    if st.sidebar.checkbox('Data Exploration'):
        chart_type = st.sidebar.radio("Chart Type", ('Line', 'Bar'))
        features = pd.DataFrame(data)        
        st.subheader("Compare Variables")               
        multiselect_variables(data, chart_type)        

def model_selection(data):        
    if st.sidebar.checkbox('Model Building'):
        algo_type = st.sidebar.radio("Algorithm", ('LR', 'RandomForestClassifier'))
        if algo_type == 'RandomForestClassifier':
            mod_RandomForestClassifier(data)


def mod_RandomForestClassifier(data):
    multi_variables_var = st.multiselect("Select Data and Target Variable", list(pd.DataFrame(data)))
    st.write(data[multi_variables_var])    
    if len(multi_variables_var) > 1:
        X = data[multi_variables_var[0]]
        Y = data[multi_variables_var[1]]
        X = X.values.reshape(1, -1)
        Y = Y.values.reshape(1, -1)
        clf = RandomForestClassifier()
        clf.fit(X, Y)
        #prediction = clf.predict(pd.DataFrame(data, index=[0]))
        #prediction_proba = clf.predict_proba(pd.DataFrame(data, index=[0]))

        st.subheader('Class labels and their corresponding index number')
        #st.write(iris.target_names)

        st.subheader('Prediction')
        #st.write(iris.target_names[prediction])
        #st.write(prediction)

        st.subheader('Prediction Probability')
        #st.write(prediction_proba)

        
def multiselect_variables(data, chart_type):    
    multi_variables_var = st.multiselect("Variables", list(pd.DataFrame(data)))
    st.write(data[multi_variables_var])
    
    if chart_type == 'Line':
        if multi_variables_var[0] != "":
            df = data[multi_variables_var[0]]
            df = data.groupby(data[multi_variables_var[0]]).sum().reset_index()

            plot_line_chart("Line Chart", data, multi_variables_var[0], multi_variables_var[0], multi_variables_var[1], multi_variables_var[1], (multi_variables_var[0],multi_variables_var[1]))
    elif chart_type == 'Bar':
        if multi_variables_var[0] != "":
            plot_bar_chart("Bar Chart", data, multi_variables_var[0], multi_variables_var[0], multi_variables_var[1], multi_variables_var[1], (multi_variables_var[0],multi_variables_var[1]))
            

def plot_line_chart(title, df, x, x_title, y, y_title, tooltip, legend_orient="top-left", zero=False, ):
    st.subheader(title) 
    show_legend = st.radio("Show Legent", ('Yes', 'No'))

    if show_legend == 'Yes':
        alt_lc = alt.Chart(df).mark_line(point=True).encode(
                                    x=alt.X(x, axis=alt.Axis(title=x_title)),
                                    y=alt.Y(y, axis=alt.Axis(title=y_title), scale=alt.Scale(zero=zero))
                                    )
    else:  
        alt_lc = alt.Chart(df).mark_line(point=True).encode(
                                    x=alt.X(x, axis=alt.Axis(title=x_title)),
                                    y=alt.Y(y, axis=alt.Axis(title=y_title), scale=alt.Scale(zero=zero))
                                    )

    st.altair_chart(alt_lc, use_container_width=True)
    return
    
def plot_bar_chart(title, df, x, x_title, y, y_title, tooltip, legend_orient="top-left", zero=False, ):
    st.subheader(title)
    alt_lc = alt.Chart(df).mark_bar(point=True).encode(
                                x=alt.X(x, axis=alt.Axis(title=x_title)),
                                y=alt.Y(y, axis=alt.Axis(title=y_title), scale=alt.Scale(zero=zero))
                                #,                                color=alt.Color(x, legend=alt.Legend(orient=legend_orient, fillColor='white'))
                                )
    st.altair_chart(alt_lc, use_container_width=True)
    return

def user_input_features():    
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    return uploaded_file

def upload_file(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)        
        features = pd.DataFrame(data)
        return features
    

def dynamic_selectbox(df):
    #add_selectbox = st.sidebar.selectbox(    "How would you like to be contacted?",    pd.DataFrame(data, index=[0,0]))
    x_axis = st.sidebar.selectbox("Choose a variable for the x-axis", df.columns, index=0)
    if st.sidebar.button('Add', key=1):
        st.write(df[x_axis])
    #y_axis = st.sidebar.selectbox("Choose a variable for the y-axis", df.columns, index=2)
    #st.sidebar.button("Add", key=2)
    #dynamic_slider(df, x_axis, y_axis)

def dynamic_slider(df, x_axis, y_axis):
    x_axis = st.sidebar.slider(x_axis, 4.3, 7.9, 5.4)
    y_axis = st.sidebar.slider(y_axis, 2.0, 4.4, 3.4)
    data = {'x_axis': x_axis,
            'y_axis': y_axis}
    features = pd.DataFrame(data, index=[0])
    return features

def visualize_data(df, x_axis, y_axis):
    print(x_axis)
    print(y_axis)
    graph = alt.Chart(df).mark_circle(size=60).encode(
                        x=x_axis,
                        y=y_axis,
                        color='Origin'
                    ).interactive()

    st.write(graph)
    
if __name__ == "__main__":
    main()


#hist_values = np.histogram(df['Service'], bins=24, range=(0,24))[0]
#st.bar_chart(hist_values)

#st.subheader('Data Frame')
#st.write(df)


#iris = datasets.load_iris()
#X = iris.data
#Y = iris.target

#clf = RandomForestClassifier()
#clf.fit(X, Y)

#prediction = clf.predict(df)
#prediction_proba = clf.predict_proba(df)

#st.subheader('Class labels and their corresponding index number')
#st.write(iris.target_names)

#st.subheader('Prediction')
#st.write(iris.target_names[prediction])
#st.write(prediction)

#st.subheader('Prediction Probability')
#st.write(prediction_proba)
