import streamlit as st
import os
import glob
import cv2

image_emoji = '📷'
model_emoji = '⚙️'
profile_emoji = '📈'
st.title('PantoScan')

tab1, tab2, tab3 = st.tabs([f'{image_emoji} Image', f'{model_emoji} Mask', f'{profile_emoji} Measurement'])

with tab1:
   st.header(f'Source Image')
   img_array = cv2.imread(glob.glob(f'{os.getcwd()}/*.png')[0])
   st.image(img_array)

with tab2:
   st.header(f'Model Output')
   #st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header(f'Profile Height')
   #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


