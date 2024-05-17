import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# add the title 
st.title("data anylysis application")
st.subheader("this is simple data analysis application")

# create a dropdown list to choose a dataset
dataset_options = ['Iris', 'tips', 'titanic','diamond']
selected_dataset = st.selectbox('select a dataset', dataset_options)
# load the selected data set
if selected_dataset == 'Iris':
    df = sns.load_dataset('iris')
elif selected_dataset == 'tips':
    df = sns.load_dataset('tips')
elif selected_dataset == 'titanic':
    df = sns.load_dataset('titanic')
elif selected_dataset == 'diamond':
    df = sns.load_dataset('diamonds')
    
# button to upload custum daraset
uploaded_file = st.file_uploader('upload a custom dataset',type=['csv','xlsx'])
if uploaded_file is not None:

    # process the uploaded file
    df = pd.read_csv(uploaded_file)
    # assuming the uploaded file is in CSV format

 # display the data set 
st.write(df)

# display the number of rows and columns from the selected data set
st.write('number of rows:', df.shape[0])
st.write('number of columns:', df.shape[1])

# display the column names of selected data with their data types
st.write('column names and data types:', df.dtypes)

#print the null values if those are >0
if df.isnull().sum().sum() > 0:
    st.write('null values:', df.isnull().sum().sort_values(ascending=False))
else:
    st.write('no null values')

# display the summary statistics of the selected data
st.write('summary statistics:', df.describe())

# create a pairplot
st.subheader('pairplot')

# select coloumn to be used as hue in pairplot
hue_column = st.selectbox('select a column to be used as hue in pairplot', df.columns)
st.pyplot(sns.pairplot(df, hue=hue_column))

# Create a heatmap
st.subheader('Heatmap')
# select the columns which are numeric and then create a corr_matrix
numeric_columns = df.select_dtypes(include=np.number).columns
corr_matrix = df[numeric_columns].corr()
numeric_columns = df.select_dtypes(include=np.number).columns
corr_matrix = df[numeric_columns].corr()

from plotly import graph_objects as go

# Convert the seaborn heatmap plot to a Plotly figure
heatmap_fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                       x=corr_matrix.columns,
                                       y=corr_matrix.columns,
                                       colorscale='Viridis'))
st.plotly_chart(heatmap_fig)