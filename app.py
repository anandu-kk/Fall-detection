import os
import time
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import asyncio
from telegram import Bot
from dotenv import load_dotenv
import tele_send
import multiprocessing
import streamlit as st
import requests

# Function to check internet connectivity
def check_internet():
    try:
        requests.get('https://www.google.com/', timeout=3)
        return True
    except Exception:
        return False

async def send_telegram_message(msg, TELEGRAM_BOT_TOKEN, chat_id):
    await asyncio.sleep(3)
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(chat_id=chat_id, text=msg)

def send_telegram_message_process(msg, TELEGRAM_BOT_TOKEN, TARGET_CHAT_IDS):
    for chat_id in TARGET_CHAT_IDS:
        asyncio.run(send_telegram_message(msg, TELEGRAM_BOT_TOKEN, chat_id))

def start_telegram_message_process(msg, TELEGRAM_BOT_TOKEN, TARGET_CHAT_IDS):
    p = multiprocessing.Process(target=send_telegram_message_process, args=(msg, TELEGRAM_BOT_TOKEN, TARGET_CHAT_IDS))
    p.start()

def run_tele_send():
    tele_send.main()

def start_alerts():
    global process
    process = multiprocessing.Process(target=run_tele_send)
    process.start()

def stop_alerts():
    global process
    process.terminate()
    process.join()
    process=None

def resume_alerts_after_20(alert_sent):
    time.sleep(1200)
    alert_sent.value=False

def run_resume_alerts_after_20(alert_sent):
    restart_process = multiprocessing.Process(target=resume_alerts_after_20, args=(alert_sent,))
    restart_process.start()



if __name__=='__main__':
    
    
    st.set_page_config(page_title="Fall Detection System", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #ff4b4b;'> 🛡️Fall Detection System</h1>", unsafe_allow_html=True)

    st.sidebar.markdown("## 📋 Information")
    st.sidebar.info(
    "1. Click the toggle to start detection\n"
    "2. Use the video file toggle to upload a video\n"
    "3. Alerts will be sent if a fall is detected\n"
    "4. Recovery stops the alert system\n"
    )

    load_dotenv()
    # Get the token from environment
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_IDS')

    TARGET_CHAT_IDS = [int(id_str.strip()) for id_str in TELEGRAM_CHAT_ID.split(',') if id_str.strip()]

    if not TELEGRAM_BOT_TOKEN:
        st.error("TELEGRAM_BOT_TOKEN not found in environment variables.")
        
    elif not TARGET_CHAT_IDS:
        st.error("TELEGRAM_CHAT_IDS not found or empty in environment variables.")

    elif not check_internet():
        st.error("Check internet connection")
        st.stop()

    else:
        start = st.sidebar.toggle("🔍 Start Detection")
        temp_dir = "temp_videos"
        os.makedirs(temp_dir, exist_ok=True)

        if start:
            if st.sidebar.toggle("🎞️ Use video file"):
                uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov",'mkv'], label_visibility="collapsed")
                submitted=st.sidebar.button(label="Upload",disabled=not uploaded_file)
                if submitted:
                    temp_file_to_save = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_to_save, "wb") as outfile:
                        outfile.write(uploaded_file.getbuffer())
                    video_source = temp_file_to_save
                    video_available = True
                else:
                    video_available = False
            else:
                video_source = 0
                video_available = True

        status_placeholder = st.empty()
        angle_placeholder = st.empty()
        

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        curr_frame = 0
        torso_angle_history = deque(maxlen=30)  # Store torso angles for the last 20 frames
        torso_angle_threshold = 50  # Threshold for rapid torso angle change (degrees)
        horizontal_torso_frames = 0  # Counter for frames where the torso is nearly horizontal
        horizontal_torso_threshold = 100  # Number of consecutive frames to confirm lying down
        horizontal_torso_angle_range = (60, 130)  # Range for horizontal torso angles (degrees)
        cooldown_seconds = 10
        last_alert_time = 0
        fall_detected = False
        recovered = True
        alert_sent = multiprocessing.Value('b', False)
        cap = None
        
        if start and video_available:
            try:
                cap = cv2.VideoCapture(video_source)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    frame_placeholder = st.empty()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert frame to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb_frame)
                    curr_frame += 1
                    
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark

                        # Get important keypoints
                        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

                        # Calculate torso angle (shoulders to hips)
                        torso_vector = np.array([left_hip.x - left_shoulder.x, left_hip.y - left_shoulder.y])
                        torso_angle = np.degrees(np.arctan2(torso_vector[1], torso_vector[0]))

                        # Add the current torso angle to the history
                        torso_angle_history.append(torso_angle)

                        # Fall Detection Logic
                        
                        # Check for rapid torso angle change
                        if len(torso_angle_history) == torso_angle_history.maxlen:
                            max_angle_change = max(torso_angle_history) - min(torso_angle_history)
                            if max_angle_change > torso_angle_threshold:  # Rapid torso angle change
                                fall_detected = True
                                


                        # Check if the person is lying down
                        if abs(torso_angle) < horizontal_torso_angle_range[0] or horizontal_torso_angle_range[1] < abs(torso_angle) :
                            horizontal_torso_frames += 1
                        else:
                            horizontal_torso_frames = 0

                        # Confirm fall if lying down for at least 30 frames
                        if horizontal_torso_frames >= horizontal_torso_threshold:
                            fall_detected = True

                        now = time.time()

                        ## Check if the person has recovered from the fall
                        if horizontal_torso_angle_range[0] < abs(torso_angle) < horizontal_torso_angle_range[1]:
                            if not recovered:
                                stop_alerts()
                                status_placeholder.success("Status: Normal")
                                start_telegram_message_process("Recovered from fall! 👍", TELEGRAM_BOT_TOKEN, TARGET_CHAT_IDS)
                                recovered = True
                                alert_sent.value = False
                            fall_detected = False

                        # Draw landmarks
                        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        # Display Fall Detection Alert
                        if fall_detected:
                            status_placeholder.error("Status: Fall Detected")
                            if not alert_sent.value:

                                start_alerts()
                                run_resume_alerts_after_20(alert_sent)
                                alert_sent.value = True
                                recovered = False

                        angle_placeholder.info(f" Torso Angle: {torso_angle:.2f}° ")
                    else:
                        angle_placeholder.info(f"Torso Angle not detected")
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame,channels="RGB")


                cap.release()
                cv2.destroyAllWindows()

            except Exception as e:
                st.error(f"An error occurred: {e}")
                if cap is not None and cap.isOpened():
                    cap.release()
                cv2.destroyAllWindows()
                st.stop()
        
