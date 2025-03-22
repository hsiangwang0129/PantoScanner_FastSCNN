import streamlit as st
import cv2
import numpy as np
import torch
from fast_scnn.fast_scnn import FastSCNN
from strip_measure_4_0 import prepare_networks_for_measurement, measure_strip
import tempfile
import pandas as pd
import plotly.express as px






# 初始化模型
path_yolo_model = 'app/detection_model.pt'
path_segmentation_model = 'app/segmentation_model.pth'

if 'models' not in st.session_state:
    segmentation_model, yolo_model = prepare_networks_for_measurement(path_yolo_model, path_segmentation_model)
    st.session_state['models'] = {'segmentation': segmentation_model, 'detection': yolo_model}

# 載入圖片
uploaded_file = st.file_uploader("上傳圖片", type=["png", "jpg", "jpeg"])



col1, col2, col3 = st.columns(3)

with col2:
    button_measure = st.button("Measure")
if uploaded_file:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    st.image(image, caption="原始圖片",use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_path = temp_file.name  # 暫存檔案的完整路徑
        cv2.imwrite(temp_path, image)  # 儲存圖片到這個路徑
    if button_measure :
        st.subheader("Running YOLO + FastSCNN...")
        img_with_detections_rgb,mask = measure_strip(temp_path, 
                             st.session_state['models']['detection'],
                             st.session_state['models']['segmentation'], 
                             [], [], [], [],[], 0, (0,0), (0,0), 1408, 3200, 320)

        if mask is not None:
            st.subheader("FastSCNN Segmentation Result")
            mask_np = mask.cpu().numpy()  # 轉成 NumPy，確保在 CPU 上
            unique_values = np.unique(mask_np)
            print(f"Unique values in mask: {unique_values}")
            mask_np = mask_np.squeeze(0)  # 確保 mask 形狀為 (H, W)
            mask_np = (mask_np * 255).astype(np.uint8)  # 轉換成 uint8 格式
            print(mask_np.shape)

            
            edges = cv2.Canny(mask_np, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

            

            # 4️⃣ 計算所有直線的平均斜率
            slopes = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)  # 轉換為角度
                if 70 < angle < 90 or 270 < angle < 290:  # 忽略接近垂直的線
                    continue
                slope = np.tan(theta - np.pi/2)  # 計算斜率
                slopes.append(slope)

            

            k_a = np.mean(slopes)  # 計算平均斜率
            alpha = np.degrees(np.arctan(k_a))  # 旋轉角度

            # 5️⃣ 旋轉影像
            (h, w) = edges.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, alpha, 1.0)
            rotated = cv2.warpAffine(mask_np, rotation_matrix, (w, h))
            # rotated = cv2.warpAffine(edges, rotation_matrix, (w, h))
            rotated_rgb = cv2.warpAffine(np.array(img_with_detections_rgb), rotation_matrix, (w, h))
            
            st.image(mask_np,caption="Segmentation Mask", use_container_width=True)
            st.subheader("First-order derivative & NMS")
            st.image(edges ,caption="Canny Edge Detection", use_container_width=True)

            st.subheader("Angle Correction")
            st.image(rotated ,caption=" image is rotated until the average slope becomes zero", use_container_width=True)

            # 找輪廓
            contours, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rotated_color = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
            # cv2.drawContours(rotated_color,contours[2],-1,(0,0,255),3,lineType=cv2.LINE_AA)
            # st.image(rotated_color ,caption="cv2.drawContours", use_container_width=True)

            

            print("contours len: ",len(contours))
            # print("contours : ",contours)
            max_point_idx = 0
            max_point_num = 0
            for i in range(len(contours)):
                if len(contours[i]) > max_point_num:
                    print("new",len(contours[i]))
                    max_point_idx = i
                    max_point_num = len(contours[i])
            contours = contours[max_point_idx]
            print("max_point_idx",max_point_idx)
                    



            # 把所有輪廓點合併到一個 NumPy 陣列
            all_points = np.vstack([c.reshape(-1, 2) for c in contours])
            print("len of points = ",len(all_points))


            # 提取 x 和 y
            x_vals = all_points[:, 0]  # x 軸座標
            y_vals = all_points[:, 1]  # y 軸座標

            # 建立 DataFrame 以便分組計算
            df = pd.DataFrame({'x': x_vals, 'y': y_vals})
            print("df = ",df.keys())
            # 計算每個 x 座標對應的最小 (y_min) 和最大 (y_max) 值
            thickness_df = df.groupby('x')['y'].agg(['min', 'max'])
            print("thickness_df = ",thickness_df.keys())
            thickness_df['thickness'] = thickness_df['max'] - thickness_df['min']
            
            
            # 轉換為 NumPy 陣列
            x_plot = thickness_df.index.to_numpy()
            thickness = thickness_df['thickness'].to_numpy()

            
            # 📌 5️⃣ Streamlit 顯示厚度曲線
            thickness_df_reset = thickness_df.reset_index()  # 讓 x 軸也變成 DataFrame 的一部分
            filtered_df = thickness_df_reset[(thickness_df_reset['x'] >= 800) & (thickness_df_reset['x'] <= 2400)]
            filtered_df = filtered_df[filtered_df['thickness']>0]
            print(filtered_df.head(50))
            print(filtered_df.shape)
            threshold = 21

            filtered_df['diff'] = filtered_df['thickness'].diff().abs()
            print(filtered_df.head(50))
            
            filtered_df = filtered_df[filtered_df['diff'] < threshold]

            filtered_df = filtered_df.drop(columns=['diff'])
            print(filtered_df.shape)
            

            fig = px.line(filtered_df, x='x', y='thickness', title="Pantograph Thickness Profile",
                        labels={"x": "X Position (pixels)", "thickness": "Thickness (pixels)"},
                        markers=True)
            
            st.plotly_chart(fig, use_container_width=True)
            max_thickness = max(filtered_df['thickness'])
            min_thickness = min(filtered_df['thickness'])
            print(max_thickness)
            print(min_thickness)
            real_thickness = (min_thickness/max_thickness)*20
            st.subheader(f"Max thickness = {max_thickness}")
            st.subheader(f"Min Thickness = {min_thickness}")
            st.subheader(f"Actuall Thickness = {real_thickness}")
            lower_x = filtered_df.loc[filtered_df['thickness']==min_thickness,'x']
            print(lower_x)
            lower_points = df[df['x'].isin(lower_x)]
            print(lower_points)
            print(lower_points.keys())
            lower_points = lower_points.loc[lower_points.groupby('x')['y'].idxmin()]
            # lower_points_copy = df[df.groupby('x')['y'] == df.groupby('x')['y'].max()]
            print(lower_points.groupby('x')['y'])
            for x, y in lower_points.values:
                cv2.circle(rotated_color, (int(x), int(y)), radius=5, color=(255,0,0), thickness=4)
            st.image(rotated_color,caption="Segmentation Mask", use_container_width=True)

            for x, y in lower_points.values:
                cv2.circle(rotated_rgb, (int(x), int(y)), radius=5, color=(255,0,0), thickness=4)
            st.image(rotated_rgb,caption="Segmentation Mask", use_container_width=True)
            


            
