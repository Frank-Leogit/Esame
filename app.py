import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st 
import io
from io import StringIO
from sklearn.model_selection import train_test_split
import joblib
import xlsxwriter
from sklearn.linear_model import LinearRegression
import os
import warnings



def main():
    warnings.filterwarnings('ignore')
    absolute_path = os.path.dirname(__file__)
    relative_path = "src/lib"
    full_path = os.path.join(absolute_path)

    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
        X = df.drop(columns="Sales")
        y =df["Sales"]
        st.write(X)
        st.write(y)
        #df1 = pd.read_csv(df)
        model = joblib.load('regression_test.pkl')
        #RD = st.number_input("Imposta R&D")
        #AD = st.number_input("Imposta Administration")
        #MK = st.number_input("Imposta Marketing")
        #arr = np.array([[RD,AD,MK]])
        #Sales = newmodel.predict(X)
        #st.write(f"Il Salesto è: {Sales}")
        #st.write(Sales)

        buffer = io.BytesIO()
        # download button 2 to download dataframe as xlsx
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write each dataframe to a different worksheet.
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            # Close the Pandas Excel writer and output the Excel file to the buffer
            writer.save()

            download2 = st.download_button(
                label="Download data as Excel",
                data=buffer,
                file_name='large_df.xlsx',
                mime='application/vnd.ms-excel'
            )
        
if __name__=="__main__":
    main()