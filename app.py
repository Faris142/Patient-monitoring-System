import streamlit as st
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import joblib
from time import sleep, time
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

model = joblib.load('logistic_regression_model.pkl')
le = joblib.load('label_encoder.pkl')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks_from_frame(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks, results.pose_landmarks
    return None, None

def generate_normal_heartbeat_data(time_data):
    return np.clip(80 + 20 * np.sin(2 * np.pi * 0.5 * time_data) + np.random.uniform(-5, 5, size=time_data.shape), 60, 100)

st.title('Patient and Heartbeat Monitoring System')

col1, col2 = st.columns([2, 1])  

video_placeholder = col1.empty()
heartbeat_chart = col2.empty()
status_box = col2.empty()
condition_report_placeholder = col1.empty()

video = cv2.VideoCapture(1)
if not video.isOpened():
    st.write("Error: Could not access the webcam.")

conditions_count = {label: 0 for label in le.classes_}
conditions_duration = {label: 0 for label in le.classes_}
current_condition = None
condition_start_time = None

num_points = 100
time_data = np.linspace(0, 10, num_points)
heartbeat_data = generate_normal_heartbeat_data(time_data)
data = pd.DataFrame({
    'Time': time_data,
    'Heartbeat': heartbeat_data
})

sns.set_style("darkgrid")
plt.style.use('dark_background')

lower_threshold = 60
upper_threshold = 100

def update_heartbeat_data(data, new_heartbeat):
    new_time = data['Time'].iloc[-1] + (data['Time'].iloc[1] - data['Time'].iloc[0])
    new_data = pd.DataFrame({
        'Time': [new_time],
        'Heartbeat': [new_heartbeat]
    })
    return pd.concat([data, new_data], ignore_index=True).iloc[1:]

def update_status_box(heartbeat):
    color = "#00cc66" if lower_threshold <= heartbeat <= upper_threshold else "#ff4d4d"
    status = f'<div style="background-color:{color};padding:10px;border-radius:5px;color:white;">{heartbeat:.2f} BPM</div>'
    return status

stop_classification = st.empty()

if stop_classification.button("Stop and Generate Report"):
    stop_classification.empty()  
    if current_condition is not None:
        conditions_duration[current_condition] += time() - condition_start_time

    st.write("Generating report...")

    # condition report 
    report_data = []
    for condition, count in conditions_count.items():
        duration = conditions_duration[condition]
        report_data.append({
            'Condition': condition,
            'Count': count,
            'Total Duration (seconds)': round(duration, 2)
        })
    
    report_df = pd.DataFrame(report_data)
    
    st.subheader('Condition Report')
    st.table(report_df)

    condition_names = list(conditions_duration.keys())
    condition_durations = [conditions_duration[condition] for condition in conditions_duration.keys()]

    condition_durations = [0 if np.isnan(duration) else duration for duration in condition_durations]

    colors = sns.color_palette("coolwarm", len(condition_names))

    fig, ax = plt.subplots()
    ax.bar(condition_names, condition_durations, color=colors)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Total Duration (seconds)')
    ax.set_title('Total Duration of Each Condition')
    st.pyplot(fig)

    condition_counts = list(conditions_count.values())

    fig, ax = plt.subplots()
    ax.bar(condition_names, condition_counts, color=colors)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Count')
    ax.set_title('Count of Each Condition')
    st.pyplot(fig)
else:
    # Streamlit app
    while True:
        ok, frame = video.read()
        if not ok:
            st.write("Error: Could not read frame from the webcam.")
            break
        
        frame = cv2.flip(frame, 1)
        landmarks, pose_landmarks = extract_landmarks_from_frame(frame)
        
        if landmarks:
            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks)
            label = le.inverse_transform(prediction)[0]
            
            if label != current_condition:
                if current_condition is not None:
                    conditions_duration[current_condition] += time() - condition_start_time
                current_condition = label
                condition_start_time = time()
            
            conditions_count[label] += 1
            
            mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2
            text_color = (255, 255, 255)  
            cv2.putText(frame, label, (10, 30), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, use_column_width=True)
        
        latest_heartbeat = generate_normal_heartbeat_data(np.array([data['Time'].iloc[-1]]))[0]
        
        data = update_heartbeat_data(data, latest_heartbeat)
        
        plt.figure(figsize=(5, 3))
        ax = sns.lineplot(x='Time', y='Heartbeat', data=data, color='#00FF00', linewidth=2.5)
        
        ax.fill_between(data['Time'], data['Heartbeat'], color='#00FF00', alpha=0.1)
        
        plt.axhspan(60, 100, color='#00FF00', alpha=0.2, label='Normal Range')
        
        plt.ylim(60, 100)
        plt.xlabel('Time (s)', color='white')
        plt.ylabel('Heartbeat (BPM)', color='white')
        plt.title('Live Heartbeat Data', color='white')
        
        ax.tick_params(colors='white')
        
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        heartbeat_chart.pyplot(plt)
        
        status_html = update_status_box(latest_heartbeat)
        status_box.markdown(status_html, unsafe_allow_html=True)
        
        report_data = []
        for condition, count in conditions_count.items():
            duration = conditions_duration[condition]
            if condition == current_condition:
                duration += time() - condition_start_time
            report_data.append({
                'Condition': condition,
                'Count': count,
                'Total Duration (seconds)': round(duration, 2)
            })
        
        report_df = pd.DataFrame(report_data)
        condition_report_placeholder.table(report_df)
        
        sleep(0.1)

    video.release()
    cv2.destroyAllWindows()
