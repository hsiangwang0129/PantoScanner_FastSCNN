import streamlit as st
import os
import glob
import cv2

st.title('Hello Urs')
st.text(os.getcwd())
st.text(glob.glob(f'{os.getcwd()}/*.png')[0])
img_array = cv2.imread(glob.glob(f'{os.getcwd()}/*.png')[0])
st.image(img_array)