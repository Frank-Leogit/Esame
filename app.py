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
        target = st.text_input("Inserisci il nome del target")
        X = df.drop(columns=target)
        y =df[target]
        st.write(X)
        st.write(y)
        size = st.number_input("Inserisci la dimensione del test",0.2,0.3)
        rnd_state = st.number_input("Inserisci il random state",value=80)
        X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=size,
                                               random_state= rnd_state)
        model = LinearRegression()
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        res = pd.DataFrame(data=list(zip(y_pred,y_test)),columns=["predetta","reale"])
        st.write(res)
        lenght = y_pred.shape[0]
        x = np.linspace(0,lenght,lenght)
        figure = plt.figure(figsize=(10,7))
        plt.plot(x, y_test, label=f"{target} reale")
        plt.plot(x, y_pred, label=f"predizione {target}")
        plt.legend(loc=2);
        st.pyplot(figure)
        Y = pd.DataFrame(y_pred,columns=[f"predizione {target}"])


        buffer = io.BytesIO()
        # download button 2 to download dataframe as xlsx
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write each dataframe to a different worksheet.
            Y.to_excel(writer, sheet_name='Sheet1', index=False)
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