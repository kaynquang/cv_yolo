import os
import cv2
import mediapipe as mp
import pandas as pd
from collections import defaultdict

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

data_set = defaultdict(list)

# folder chứa các thư mục tư thế yoga
root_folder = 'E:\YogaRandomFr\YogaPoses'  

for idx, label in enumerate(os.listdir(root_folder)):
    folder_path = os.path.join(root_folder, label)
    if not os.path.isdir(folder_path):
        continue

    for image_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            for idd, landmark in enumerate(results.pose_landmarks.landmark):
                data_set[f'X {idd}'].append(landmark.x)
                data_set[f'Y {idd}'].append(landmark.y)
                data_set[f'Z {idd}'].append(landmark.z)
            data_set['Labels'].append(label)
            data_set['Image_Name'].append(image_name)


data_df = pd.DataFrame(data_set)
data_df.to_csv('yoga_keypoints.csv', index=False)
