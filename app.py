import streamlit as st
import cv2
import numpy as np
import pickle
import time
import os
import tempfile
from ultralytics import YOLO

from voice import speak  # dùng voice.py của bạn
from PIL import Image
import glob
import random

# ================== CẤU HÌNH TRANG ==================
st.set_page_config(page_title="Yoga Pose Trainer", layout="wide")
st.title("Yoga Pose Trainer")

# ================== CẤU HÌNH TƯ THẾ & GÓC ==================
POSES = ["Downdog", "Plank", "Tree", "Warrior2", "Goddess"]

# Đường dẫn thư mục chứa ảnh mẫu yoga poses
YOGA_POSES_DIR = "/Users/kaynnguyen/Downloads/YogaRandomFr-2/YogaPoses"

POSE_ANGLES = {
    "Downdog": {
        "left_elbow": (170, 20),
        "right_elbow": (170, 20),
        "left_hip": (70, 20),
        "right_hip": (70, 20),
        "left_knee": (170, 20),
        "right_knee": (170, 20),
    },
    "Plank": {
        "left_elbow": (160, 25),
        "right_elbow": (160, 25),
        "left_hip": (170, 20),
        "right_hip": (170, 20),
        "left_knee": (170, 20),
        "right_knee": (170, 20),
    },
    "Tree": {
        "left_knee": (170, 15),
        "left_hip": (170, 20),
    },
    "Warrior2": {
        "left_knee": (95, 25),
        "right_knee": (165, 20),
        "left_shoulder": (85, 25),
        "right_shoulder": (85, 25),
    },
    "Goddess": {
        "left_knee": (95, 25),
        "right_knee": (95, 25),
        "left_hip": (120, 30),
        "right_hip": (120, 30),
    },
}


def lay_hinh_mau_pose(ten_pose):
    """Lấy một ảnh mẫu từ thư mục YogaPoses cho pose tương ứng."""
    pose_folder = os.path.join(YOGA_POSES_DIR, ten_pose)
    if os.path.exists(pose_folder):
        danh_sach_hinh = glob.glob(os.path.join(pose_folder, "*.jpg"))
        if danh_sach_hinh:
            # Lấy ảnh đầu tiên (có thể random nếu muốn)
            return danh_sach_hinh[1]
    return None


def tinh_goc(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    goc = np.abs(radians * 180.0 / np.pi)
    return 360 - goc if goc > 180.0 else goc


def lay_toa_do_yolo(keypoints, idx):
    """Lấy tọa độ từ YOLO keypoints (shape: 17x2 hoặc 17x3)."""
    return [keypoints[idx][0], keypoints[idx][1]]


def tinh_goc_cac_khop_yolo(keypoints):
    """Tính góc các khớp từ YOLO keypoints.
    YOLO indices: 5-left_shoulder, 6-right_shoulder, 7-left_elbow, 8-right_elbow,
    9-left_wrist, 10-right_wrist, 11-left_hip, 12-right_hip,
    13-left_knee, 14-right_knee, 15-left_ankle, 16-right_ankle
    """
    idx = {
        'VAI_TRAI': 5, 'VAI_PHAI': 6,
        'KHUYU_TRAI': 7, 'KHUYU_PHAI': 8,
        'CO_TAY_TRAI': 9, 'CO_TAY_PHAI': 10,
        'HONG_TRAI': 11, 'HONG_PHAI': 12,
        'GOI_TRAI': 13, 'GOI_PHAI': 14,
        'CO_CHAN_TRAI': 15, 'CO_CHAN_PHAI': 16,
    }
    return {
        "left_elbow": tinh_goc(lay_toa_do_yolo(keypoints, idx['VAI_TRAI']),
                               lay_toa_do_yolo(keypoints, idx['KHUYU_TRAI']),
                               lay_toa_do_yolo(keypoints, idx['CO_TAY_TRAI'])),
        "right_elbow": tinh_goc(lay_toa_do_yolo(keypoints, idx['VAI_PHAI']),
                                lay_toa_do_yolo(keypoints, idx['KHUYU_PHAI']),
                                lay_toa_do_yolo(keypoints, idx['CO_TAY_PHAI'])),
        "left_shoulder": tinh_goc(lay_toa_do_yolo(keypoints, idx['HONG_TRAI']),
                                  lay_toa_do_yolo(keypoints, idx['VAI_TRAI']),
                                  lay_toa_do_yolo(keypoints, idx['KHUYU_TRAI'])),
        "right_shoulder": tinh_goc(lay_toa_do_yolo(keypoints, idx['HONG_PHAI']),
                                   lay_toa_do_yolo(keypoints, idx['VAI_PHAI']),
                                   lay_toa_do_yolo(keypoints, idx['KHUYU_PHAI'])),
        "left_hip": tinh_goc(lay_toa_do_yolo(keypoints, idx['VAI_TRAI']),
                             lay_toa_do_yolo(keypoints, idx['HONG_TRAI']),
                             lay_toa_do_yolo(keypoints, idx['GOI_TRAI'])),
        "right_hip": tinh_goc(lay_toa_do_yolo(keypoints, idx['VAI_PHAI']),
                              lay_toa_do_yolo(keypoints, idx['HONG_PHAI']),
                              lay_toa_do_yolo(keypoints, idx['GOI_PHAI'])),
        "left_knee": tinh_goc(lay_toa_do_yolo(keypoints, idx['HONG_TRAI']),
                              lay_toa_do_yolo(keypoints, idx['GOI_TRAI']),
                              lay_toa_do_yolo(keypoints, idx['CO_CHAN_TRAI'])),
        "right_knee": tinh_goc(lay_toa_do_yolo(keypoints, idx['HONG_PHAI']),
                               lay_toa_do_yolo(keypoints, idx['GOI_PHAI']),
                               lay_toa_do_yolo(keypoints, idx['CO_CHAN_PHAI'])),
    }


def danh_gia_tu_the(goc_co_the, ten_tu_the):
    if ten_tu_the not in POSE_ANGLES:
        return 100, []
    tieu_chuan = POSE_ANGLES[ten_tu_the]
    diem_so, can_chinh = [], []
    for khop, (goc_ly_tuong, sai_so) in tieu_chuan.items():
        if khop in goc_co_the:
            chenh_lech = abs(goc_co_the[khop] - goc_ly_tuong)
            if chenh_lech <= sai_so:
                diem_so.append(100)
            elif chenh_lech <= sai_so * 2:
                diem_so.append(80 - (chenh_lech - sai_so))
            else:
                diem_so.append(max(0, 60 - chenh_lech))
                can_chinh.append(khop.replace("_", " "))
    return int(np.mean(diem_so)) if diem_so else 0, can_chinh[:2]


def trich_xuat_keypoints_yolo(keypoints, img_shape):
    """Trích xuất và chuẩn hóa keypoints từ YOLO (17 keypoints, x,y)."""
    h, w = img_shape[:2]
    features = []
    for i in range(17):
        features.append(keypoints[i][0] / w)  # X chuẩn hóa
        features.append(keypoints[i][1] / h)  # Y chuẩn hóa
    return np.array(features).reshape(1, -1)


def ve_skeleton_yolo(frame, keypoints):
    """Vẽ skeleton từ YOLO keypoints lên frame."""
    # Các cặp kết nối skeleton (COCO format)
    connections = [
        (5, 6),   # vai - vai
        (5, 7), (7, 9),   # vai trái - khuỷu - cổ tay
        (6, 8), (8, 10),  # vai phải - khuỷu - cổ tay
        (5, 11), (6, 12), # vai - hông
        (11, 12),  # hông - hông
        (11, 13), (13, 15),  # hông trái - gối - cổ chân
        (12, 14), (14, 16),  # hông phải - gối - cổ chân
        (0, 1), (0, 2),  # mũi - mắt
        (1, 3), (2, 4),  # mắt - tai
    ]
    
    # Vẽ các đường nối
    for (i, j) in connections:
        pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
        pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    # Vẽ các điểm keypoint
    for i in range(17):
        pt = (int(keypoints[i][0]), int(keypoints[i][1]))
        if pt[0] > 0 and pt[1] > 0:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)


# ================== LOAD YOLO & MODEL ==================
@st.cache_resource
def load_yolo_pose():
    yolo_model = YOLO('yolov8n-pose.pt')
    return yolo_model


@st.cache_resource
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


yolo_pose = load_yolo_pose()

# ================== SIDEBAR ==================
st.sidebar.header("Cấu hình")

# Liệt kê .pkl trong models/
models_dir = "models"
model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
if not model_files:
    st.sidebar.error("Không tìm thấy file .pkl nào trong thư mục 'models/'.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Chọn model:", model_files)
model_path = os.path.join(models_dir, selected_model_name)
model = load_model(model_path)

mode = st.sidebar.radio("Chế độ:", ["Webcam realtime", "Upload ảnh", "Upload video"])

target_pose_option = st.sidebar.selectbox(
    "Tư thế để chấm điểm:",
    ["Auto (theo model)"] + POSES
)

hien_skeleton = st.sidebar.checkbox("Hiện skeleton", value=True)
bat_tts = st.sidebar.checkbox("Bật giọng nói (TTS)", value=True)

# ================== LAYOUT ==================
# Chia 2 cột: webcam bên trái, hình mẫu + thông tin bên phải
col_webcam, col_mau = st.columns([1, 1])

with col_webcam:
    st.subheader("Camera")
    video_placeholder = st.empty()

with col_mau:
    st.subheader("Tư thế mẫu")
    hinh_mau_placeholder = st.empty()
    dem_nguoc_placeholder = st.empty()
    info_model = st.empty()
    info_pose = st.empty()
    info_score = st.empty()
    info_time = st.empty()
    info_status = st.empty()

info_model.markdown(f"**Model đang dùng:** `{selected_model_name}`")


def process_video_capture(cap, use_speak=True):
    """Dùng chung cho webcam & video upload."""
    current_pose = None
    last_spoken_pose = None
    hinh_mau_hien_tai = None

    thoi_gian_bat_dau = None
    thoi_gian_giu = 0.0
    last_spoken_second = 0
    good_announced = False
    
    # Biến cho đếm ngược 3,2,1
    dem_nguoc_bat_dau = None
    dem_nguoc_hoan_thanh = False
    last_dem_nguoc = 0

    spoken_feedback_list = []
    last_feedback_time = 0.0
    FEEDBACK_INTERVAL = 5.0  # Tăng lên 5 giây để giảm tiếng ồn
    DIEM_DE_BAT_DAU = 70  # Điểm tối thiểu để bắt đầu đếm ngược

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Không đọc được frame nữa (hết video hoặc lỗi).")
            break

        # frame = cv2.flip(frame, 1)  # nếu muốn như gương (dùng cho webcam)
        
        # YOLO pose detection
        results = yolo_pose(frame, verbose=False)
        
        detected = False
        kpts = None
        
        if results and len(results) > 0:
            keypoints_obj = results[0].keypoints
            if keypoints_obj is not None and len(keypoints_obj.xy) > 0:
                kpts = keypoints_obj.xy[0].cpu().numpy()  # Shape: (17, 2)
                if len(kpts) == 17:
                    detected = True

        if detected and kpts is not None:
            if hien_skeleton:
                ve_skeleton_yolo(frame, kpts)

            features = trich_xuat_keypoints_yolo(kpts, frame.shape)
            predicted_pose = model.predict(features)[0]

            # Chọn tư thế để CHẤM ĐIỂM
            if target_pose_option == "Auto (theo model)":
                pose_for_score = predicted_pose
            else:
                pose_for_score = target_pose_option

            goc_co_the = tinh_goc_cac_khop_yolo(kpts)
            diem, can_chinh = danh_gia_tu_the(goc_co_the, pose_for_score)

            now = time.time()

            if predicted_pose != current_pose:
                current_pose = predicted_pose
                thoi_gian_bat_dau = None
                thoi_gian_giu = 0.0
                last_spoken_second = 0
                good_announced = False
                spoken_feedback_list = []
                last_feedback_time = 0.0
                
                # Reset đếm ngược khi đổi pose
                dem_nguoc_bat_dau = None
                dem_nguoc_hoan_thanh = False
                last_dem_nguoc = 4
                
                # Lấy hình mẫu cho pose mới
                hinh_mau_hien_tai = lay_hinh_mau_pose(predicted_pose)
                
                # Chỉ cập nhật last_spoken_pose, không nói tên tư thế nữa
                last_spoken_pose = current_pose

            # Đếm thời gian giữ tư thế khi điểm đủ cao
            if diem >= DIEM_DE_BAT_DAU:
                # Bắt đầu đếm ngược 3,2,1 nếu chưa
                if dem_nguoc_bat_dau is None:
                    dem_nguoc_bat_dau = now
                    dem_nguoc_hoan_thanh = False
                    last_dem_nguoc = 4  # Để không nói lại số 3
                
                thoi_gian_dem_nguoc = now - dem_nguoc_bat_dau
                
                # Đếm ngược 3, 2, 1
                if not dem_nguoc_hoan_thanh:
                    so_dem = 3 - int(thoi_gian_dem_nguoc)
                    if so_dem >= 1:
                        if so_dem != last_dem_nguoc:
                            if use_speak:
                                speak(str(so_dem))
                            last_dem_nguoc = so_dem
                        dem_nguoc_placeholder.markdown(f"### Đếm ngược: **{so_dem}**")
                    else:
                        # Hoàn thành đếm ngược, bắt đầu timer
                        dem_nguoc_hoan_thanh = True
                        thoi_gian_bat_dau = now
                        dem_nguoc_placeholder.markdown("### **BẮT ĐẦU GIỮ!**")
                        if use_speak:
                            speak("Bắt đầu!")
                else:
                    # Timer chính
                    thoi_gian_giu = now - thoi_gian_bat_dau
                    dem_nguoc_placeholder.empty()
            else:
                # Reset tất cả nếu điểm không đủ
                dem_nguoc_bat_dau = None
                dem_nguoc_hoan_thanh = False
                last_dem_nguoc = 0
                thoi_gian_bat_dau = None
                thoi_gian_giu = 0.0
                last_spoken_second = 0
                good_announced = False
                spoken_feedback_list = []
                last_feedback_time = now
                dem_nguoc_placeholder.markdown("### Vào tư thế đúng để đếm ngược")

            current_sec = int(thoi_gian_giu)
            current_min = current_sec // 60

            # Thông báo các mốc thời gian quan trọng (phút)
            MOC_THOI_GIAN = [1, 5, 10, 15, 20, 30, 45, 60]  # phút
            if use_speak and dem_nguoc_hoan_thanh:
                if current_min in MOC_THOI_GIAN and current_min != last_spoken_second:
                    speak(f"Bạn đã giữ được {current_min} phút!")
                    last_spoken_second = current_min
                elif current_sec == 30 and last_spoken_second == 0:
                    # Thông báo 30 giây đầu tiên
                    speak("Tốt lắm! 30 giây!")
                    last_spoken_second = -1  # Đánh dấu đã nói 30s

            # Feedback chỉnh khớp mỗi 3s
            if use_speak and can_chinh and (now - last_feedback_time >= FEEDBACK_INTERVAL):
                for khop in can_chinh:
                    if khop not in spoken_feedback_list:
                        speak(f"Điều chỉnh {khop}")
                        spoken_feedback_list.append(khop)
                        last_feedback_time = now
                        break
            if not can_chinh:
                spoken_feedback_list = []

            # Vẽ overlay
            cv2.putText(frame, f"{predicted_pose} ({pose_for_score}) - {diem}%", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Hold: {thoi_gian_giu:.1f}s", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if dem_nguoc_hoan_thanh and current_sec >= 10:
                cv2.putText(frame, "GOOD!", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Hiển thị hình mẫu pose
            if hinh_mau_hien_tai and os.path.exists(hinh_mau_hien_tai):
                try:
                    img_mau = Image.open(hinh_mau_hien_tai)
                    hinh_mau_placeholder.image(img_mau, caption=f"Tư thế mẫu: {predicted_pose}", use_container_width=True)
                except Exception as e:
                    hinh_mau_placeholder.warning(f"Không load được hình mẫu: {e}")
            else:
                hinh_mau_placeholder.info(f"Chưa có hình mẫu cho: {predicted_pose}")

            # Cập nhật info
            info_pose.markdown(f"**Model detect:** `{predicted_pose}`")
            info_score.markdown(f"**Chấm theo tư thế:** `{pose_for_score}` → `{diem}%`")
            if dem_nguoc_hoan_thanh:
                info_time.markdown(f"**Thời gian giữ:** `{thoi_gian_giu:.1f} s`")
            else:
                info_time.markdown("**Thời gian giữ:** _Chờ đếm ngược..._")
            if can_chinh:
                info_status.markdown("Cần chỉnh: " + ", ".join(can_chinh))
            else:
                info_status.markdown("Tư thế khá tốt!")

        else:
            # Không detect pose
            current_pose = None
            hinh_mau_hien_tai = None
            thoi_gian_bat_dau = None
            thoi_gian_giu = 0.0
            last_spoken_second = 0
            good_announced = False
            dem_nguoc_bat_dau = None
            dem_nguoc_hoan_thanh = False
            last_dem_nguoc = 0
            spoken_feedback_list = []
            last_feedback_time = time.time()
            info_pose.markdown("**Tư thế hiện tại:** _Chưa nhận diện_")
            info_status.markdown("Không nhận diện được người trong khung hình.")
            hinh_mau_placeholder.info("Đang chờ nhận diện tư thế...")
            dem_nguoc_placeholder.empty()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        time.sleep(0.02)


# ================== MODE 1: WEBCAM REALTIME ==================
if mode == "Webcam realtime":
    if st.button("Bắt đầu luyện tập với webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Không mở được webcam. Kiểm tra lại camera.")
        else:
            st.warning("Đang chạy webcam... Dừng app hoặc đóng tab để thoát.")
            process_video_capture(cap, use_speak=bat_tts)
            cap.release()
            st.success("Đã dừng webcam.")

# ================== MODE 2: UPLOAD ẢNH ==================
elif mode == "Upload ảnh":
    uploaded_file = st.file_uploader("Upload ảnh (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Không đọc được ảnh.")
        else:
            # YOLO pose detection
            results = yolo_pose(img, verbose=False)
            
            detected = False
            kpts = None
            
            if results and len(results) > 0:
                keypoints_obj = results[0].keypoints
                if keypoints_obj is not None and len(keypoints_obj.xy) > 0:
                    kpts = keypoints_obj.xy[0].cpu().numpy()
                    if len(kpts) == 17:
                        detected = True

            if detected and kpts is not None:
                if hien_skeleton:
                    ve_skeleton_yolo(img, kpts)

                features = trich_xuat_keypoints_yolo(kpts, img.shape)
                predicted_pose = model.predict(features)[0]

                if target_pose_option == "Auto (theo model)":
                    pose_for_score = predicted_pose
                else:
                    pose_for_score = target_pose_option

                goc_co_the = tinh_goc_cac_khop_yolo(kpts)
                diem, can_chinh = danh_gia_tu_the(goc_co_the, pose_for_score)

                cv2.putText(img, f"Predict: {predicted_pose}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(img, f"Score ({pose_for_score}): {diem}%", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                y0 = 110
                for i, khop in enumerate(can_chinh):
                    cv2.putText(img, f"Fix: {khop}", (20, y0 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                info_pose.markdown(f"**Model detect:** `{predicted_pose}`")
                info_score.markdown(f"**Chấm theo tư thế:** `{pose_for_score}` → `{diem}%`")
                if can_chinh:
                    info_status.markdown("Cần chỉnh: " + ", ".join(can_chinh))
                else:
                    info_status.markdown("Tư thế khá tốt!")

            else:
                info_pose.markdown("**Tư thế hiện tại:** _Chưa nhận diện pose_")
                info_status.markdown("Không tìm thấy landmarks trên ảnh.")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video_placeholder.image(img_rgb, channels="RGB", use_container_width=True)
    else:
        st.info("Hãy upload một tấm ảnh yoga để hệ thống nhận diện và chấm điểm.")

# ================== MODE 3: UPLOAD VIDEO ==================
elif mode == "Upload video":
    uploaded_video = st.file_uploader("Upload video (mp4, avi, mov, mkv)",
                                      type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        # Lưu video ra file tạm
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.flush()

        if st.button("Chạy video"):
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Không mở được video.")
            else:
                st.warning("Đang phân tích video...")
                # Có thể tắt speak nếu không muốn nó đọc quá nhiều khi xem lại video:
                process_video_capture(cap, use_speak=bat_tts)
                cap.release()
                st.success("Đã xử lý xong video.")
    else:
        st.info("Hãy upload một video yoga để hệ thống phân tích.")
