import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

from voice import speak  # file voice.py của bạn


# ================== LOAD MODEL & MEDIAPIPE ==================
model = pickle.load(open('models/yoga_rf_model.pkl', 'rb'))

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================== CẤU HÌNH TƯ THẾ & GÓC ==================

POSES = ["Downdog", "Plank", "Tree", "Warrior2", "Goddess"]

# Góc lý tưởng cho từng tư thế: {khớp: (góc_lý_tưởng, sai_số_cho_phép)}
POSE_ANGLES = {
    "Downdog": {  # Tư thế chó úp mặt
        "left_elbow": (170, 20),
        "right_elbow": (170, 20),
        "left_hip": (70, 20),
        "right_hip": (70, 20),
        "left_knee": (170, 20),
        "right_knee": (170, 20),
    },
    "Plank": {  # Tư thế plank
        "left_elbow": (160, 25),
        "right_elbow": (160, 25),
        "left_hip": (170, 20),
        "right_hip": (170, 20),
        "left_knee": (170, 20),
        "right_knee": (170, 20),
    },
    "Tree": {  # Tư thế cây
        "left_knee": (170, 15),
        "left_hip": (170, 20),
    },
    "Warrior2": {  # Tư thế chiến binh 2
        "left_knee": (95, 25),
        "right_knee": (165, 20),
        "left_shoulder": (85, 25),
        "right_shoulder": (85, 25),
    },
    "Goddess": {  # Tư thế nữ thần
        "left_knee": (95, 25),
        "right_knee": (95, 25),
        "left_hip": (120, 30),
        "right_hip": (120, 30),
    },
}


def tinh_goc(a, b, c):
    """Tính góc tại điểm b (a-b-c) tính theo độ."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    goc = np.abs(radians * 180.0 / np.pi)
    return 360 - goc if goc > 180.0 else goc


def lay_toa_do(landmarks, idx):
    """Lấy tọa độ (x, y) của 1 landmark."""
    lm = landmarks.landmark[idx]
    return [lm.x, lm.y]


def tinh_goc_cac_khop(landmarks):
    """Tính góc của các khớp quan trọng."""
    idx = {
        'VAI_TRAI': 11, 'VAI_PHAI': 12,
        'KHUYU_TRAI': 13, 'KHUYU_PHAI': 14,
        'CO_TAY_TRAI': 15, 'CO_TAY_PHAI': 16,
        'HONG_TRAI': 23, 'HONG_PHAI': 24,
        'GOI_TRAI': 25, 'GOI_PHAI': 26,
        'CO_CHAN_TRAI': 27, 'CO_CHAN_PHAI': 28,
    }
    return {
        "left_elbow": tinh_goc(lay_toa_do(landmarks, idx['VAI_TRAI']),
                               lay_toa_do(landmarks, idx['KHUYU_TRAI']),
                               lay_toa_do(landmarks, idx['CO_TAY_TRAI'])),
        "right_elbow": tinh_goc(lay_toa_do(landmarks, idx['VAI_PHAI']),
                                lay_toa_do(landmarks, idx['KHUYU_PHAI']),
                                lay_toa_do(landmarks, idx['CO_TAY_PHAI'])),
        "left_shoulder": tinh_goc(lay_toa_do(landmarks, idx['HONG_TRAI']),
                                  lay_toa_do(landmarks, idx['VAI_TRAI']),
                                  lay_toa_do(landmarks, idx['KHUYU_TRAI'])),
        "right_shoulder": tinh_goc(lay_toa_do(landmarks, idx['HONG_PHAI']),
                                   lay_toa_do(landmarks, idx['VAI_PHAI']),
                                   lay_toa_do(landmarks, idx['KHUYU_PHAI'])),
        "left_hip": tinh_goc(lay_toa_do(landmarks, idx['VAI_TRAI']),
                             lay_toa_do(landmarks, idx['HONG_TRAI']),
                             lay_toa_do(landmarks, idx['GOI_TRAI'])),
        "right_hip": tinh_goc(lay_toa_do(landmarks, idx['VAI_PHAI']),
                              lay_toa_do(landmarks, idx['HONG_PHAI']),
                              lay_toa_do(landmarks, idx['GOI_PHAI'])),
        "left_knee": tinh_goc(lay_toa_do(landmarks, idx['HONG_TRAI']),
                              lay_toa_do(landmarks, idx['GOI_TRAI']),
                              lay_toa_do(landmarks, idx['CO_CHAN_TRAI'])),
        "right_knee": tinh_goc(lay_toa_do(landmarks, idx['HONG_PHAI']),
                               lay_toa_do(landmarks, idx['GOI_PHAI']),
                               lay_toa_do(landmarks, idx['CO_CHAN_PHAI'])),
    }


def danh_gia_tu_the(goc_co_the, ten_tu_the):
    """
    Đánh giá tư thế yoga dựa trên góc các khớp.
    Trả về: (điểm trung bình %, danh sách khớp cần chỉnh).
    """
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
                # lệch hơn nhưng vẫn tạm ổn
                diem_so.append(80 - (chenh_lech - sai_so))
            else:
                diem_so.append(max(0, 60 - chenh_lech))
                can_chinh.append(khop.replace("_", " "))

    return int(np.mean(diem_so)) if diem_so else 0, can_chinh[:2]


def trich_xuat_keypoints(landmarks):
    """Trích xuất (x, y, z) của tất cả landmarks thành vector."""
    keypoints = []
    for lm in landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])
    return np.array(keypoints).reshape(1, -1)


def main():
    # ================== VIDEO INPUT ==================
    # Webcam:
    cap = cv2.VideoCapture(0)

    # Video file demo:
    #cap = cv2.VideoCapture(r'E:\YogaRandomFr\test\downdog.mp4')

    current_pose = None
    last_spoken_pose = None

    thoi_gian_bat_dau = None   # mốc thời gian bắt đầu giữ đúng tư thế (điểm đủ cao)
    thoi_gian_giu = 0.0
    last_spoken_second = 0
    good_announced = False     # để chỉ nói "Giỏi quá bạn ơi!" một lần

    # cho feedback chỉnh khớp
    spoken_feedback_list = []  # danh sách các khớp đã nói lỗi
    last_feedback_time = 0.0
    FEEDBACK_INTERVAL = 3.0    # mỗi 3 giây mới cho 1 feedback mới

    print("Nhấn 'q' để thoát")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Nếu dùng webcam, có thể lật hình:
        # frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)

            # Dự đoán tư thế
            keypoints = trich_xuat_keypoints(results.pose_landmarks)
            predicted_pose = model.predict(keypoints)[0]

            # Tính góc & chấm điểm
            goc_co_the = tinh_goc_cac_khop(results.pose_landmarks)
            diem, can_chinh = danh_gia_tu_the(goc_co_the, predicted_pose)

            now = time.time()

            # Nếu đổi tư thế → reset đếm / thời gian / good / feedback
            if predicted_pose != current_pose:
                current_pose = predicted_pose
                thoi_gian_bat_dau = None
                thoi_gian_giu = 0.0
                last_spoken_second = 0
                good_announced = False
                spoken_feedback_list = []
                last_feedback_time = 0.0

                if current_pose != last_spoken_pose:
                    speak(current_pose)
                    last_spoken_pose = current_pose

            # Đếm thời gian giữ tư thế khi điểm đủ cao (>=70%)
            if diem >= 70:
                if thoi_gian_bat_dau is None:
                    thoi_gian_bat_dau = now
                thoi_gian_giu = now - thoi_gian_bat_dau
            else:
                thoi_gian_bat_dau = None
                thoi_gian_giu = 0.0
                last_spoken_second = 0
                good_announced = False
                spoken_feedback_list = []
                last_feedback_time = now

            # Đếm bằng giây: đọc 1,2,...,10 (mỗi giây 1 lần)
            current_sec = int(thoi_gian_giu)

            if 1 <= current_sec <= 10 and current_sec != last_spoken_second:
                speak(str(current_sec))
                last_spoken_second = current_sec

            # Khi đạt >=10s chạy GOOD
            if current_sec >= 10 and not good_announced:
                speak("Giỏi quá bạn ơi!")
                good_announced = True

            # ========= FEEDBACK CHỈNH KHỚP =========
            # Mỗi 3s, đọc tối đa 1 lỗi mới (nếu có)
            if can_chinh and (now - last_feedback_time >= FEEDBACK_INTERVAL):
                for khop in can_chinh:
                    if khop not in spoken_feedback_list:
                        speak(f"Điều chỉnh {khop}")
                        spoken_feedback_list.append(khop)
                        last_feedback_time = now
                        break  # chỉ đọc 1 lỗi rồi thôi

            if not can_chinh:
                # nếu tư thế đã chuẩn, reset list để lần sau có lỗi khác sẽ đọc lại
                spoken_feedback_list = []

            # ========= HIỂN THỊ LÊN FRAME =========
            # Tên tư thế + điểm
            cv2.putText(frame, f"{current_pose} - {diem}%", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Thời gian giữ
            cv2.putText(frame, f"Hold: {thoi_gian_giu:.1f}s", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # GOOD! khi đủ 10s
            if current_sec >= 10:
                cv2.putText(frame, "GOOD!", (50, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # In log ra console (debug)
            print(f"Tư thế: {current_pose} | Điểm: {diem}% | Giữ: {thoi_gian_giu:.1f}s")

        else:
            # Không detect được pose → reset
            current_pose = None
            thoi_gian_bat_dau = None
            thoi_gian_giu = 0.0
            last_spoken_second = 0
            good_announced = False
            spoken_feedback_list = []
            last_feedback_time = time.time()

        cv2.imshow("Yoga Pose Trainer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("\nĐã thoát chương trình.")


if __name__ == "__main__":
    main()
