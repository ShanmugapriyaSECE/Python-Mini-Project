import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Loan Default Predictor Visualization")

uploaded_file = st.file_uploader("Upload loan data CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Preprocessing
    df['approval_date'] = pd.to_datetime(df['approval_date'])
    df['year'] = df['approval_date'].dt.year
    df['month'] = df['approval_date'].dt.month
    
    # Bar Plot
    st.subheader("Defaults by Credit Score")
    credit_defaults = df[df['default_status'] == 1].groupby('credit_score').size()
    fig1, ax1 = plt.subplots()
    sns.barplot(x=credit_defaults.index, y=credit_defaults.values, ax=ax1)
    st.pyplot(fig1)
    
    # Line Plot
    st.subheader("Defaults Over Time")
    monthly_defaults = df[df['default_status'] == 1].groupby(['year', 'month']).size()
    fig2, ax2 = plt.subplots()
    monthly_defaults.plot(ax=ax2)
    st.pyplot(fig2)

    # Heatmap
    st.subheader("Defaults by Loan Type and Region")
    heatmap_data = df[df['default_status'] == 1].pivot_table(index='loan_type', columns='region', aggfunc='size', fill_value=0)
    fig3, ax3 = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, fmt="d", ax=ax3)
    st.pyplot(fig3)
