import streamlit as st
import os
import glob


st.title('Hello Urs')
st.text(os.getcwd())
st.text(glob.glob(f'{os.getcwd()}/*.png')[0])
st.image(glob.glob(f'{os.getcwd()}/*.png')[0])