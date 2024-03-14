import streamlit as st
import os
import glob
import cv2
import numpy as np

image_emoji = '📷'
model_emoji = '⚙️'
profile_emoji = '📈'
st.title('PantoScanner')
st.header('Example')
st.subheader('Thickness measurement of sliding element')

import numpy as np

def generate_data(slope, intercept, num_points):
  """
  Generates data points with a linear degression and a +/- 5% tolerance.

  Args:
      slope: The slope of the linear degression.
      intercept: The y-intercept of the linear degression.
      num_points: The number of data points to generate.

  Returns:
      A numpy array of size (num_points, 1) containing the data points.
  """
  x = np.linspace(0, 1, num_points)  # Creates evenly spaced x-values
  y = slope * x + intercept        # Generates linear function values

  # Add random noise with +/- 5% tolerance
  noise = np.random.uniform(low=-0.05, high=0.05, size=num_points)
  y += noise * y  # Scale noise by original y value for percentage variation

  return y.reshape(-1, 1)  # Reshape to column vector


tab1, tab2, tab3 = st.tabs([f' {image_emoji}  Image', f' {model_emoji}  Mask', f' {profile_emoji}  Measurement'])

with tab1:
   st.header(f'Source Image')
   img_array = cv2.imread(glob.glob(f'{os.getcwd()}/*.png')[0])
   st.image(img_array)

with tab2:
   st.header(f'Model Output')
   #st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header(f'Profile Height')
   # data = np.random.randn(10, 1)
   # Example usage
   # Use 'data' for your chart with linear degression and +/- 5% tolerance

   slope = -2  # Example slope for linear degression
   intercept = 45
   num_points = 20
    
   data = generate_data(slope, intercept, num_points)
   st.line_chart(data)


