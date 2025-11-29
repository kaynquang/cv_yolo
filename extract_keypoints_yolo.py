import os
import cv2
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO

# Load YOLO Pose model
model = YOLO('yolov8n-pose.pt')

# YOLO Pose có 17 keypoints (COCO format):
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

data_set = defaultdict(list)

# Folder chứa các thư mục tư thế yoga
root_folder = '/Users/kaynnguyen/Downloads/YogaRandomFr-2/YogaPoses'

for idx, label in enumerate(os.listdir(root_folder)):
    folder_path = os.path.join(root_folder, label)
    if not os.path.isdir(folder_path):
        continue

    print(f"Đang xử lý: {label}")
    
    for image_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Chạy YOLO pose detection
        results = model(img, verbose=False)
        
        if results and len(results) > 0:
            keypoints = results[0].keypoints
            if keypoints is not None and len(keypoints.xy) > 0:
                # Lấy người đầu tiên được detect
                kpts = keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
                
                if len(kpts) == 17:
                    for idd in range(17):
                        # Chuẩn hóa tọa độ về [0, 1]
                        h, w = img.shape[:2]
                        data_set[f'X {idd}'].append(kpts[idd][0] / w)
                        data_set[f'Y {idd}'].append(kpts[idd][1] / h)
                    data_set['Labels'].append(label)
                    data_set['Image_Name'].append(image_name)

print(f"Tổng số mẫu: {len(data_set['Labels'])}")

data_df = pd.DataFrame(data_set)
data_df.to_csv('yoga_keypoints_yolo.csv', index=False)
print("Đã lưu yoga_keypoints_yolo.csv")
