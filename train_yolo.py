import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load dữ liệu
df = pd.read_csv('yoga_keypoints_yolo.csv')

# Tách features và labels
X = df.drop(columns=['Labels', 'Image_Name'])
y = df['Labels']

print(f"Tổng số mẫu: {len(y)}")
print(f"Các lớp: {y.unique()}")
print(f"Số features: {X.shape[1]}")

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Train Random Forest
print("\nĐang train Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Đánh giá
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Lưu model
model_path = 'models/yoga_rf_model_yolo.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)

print(f"\nĐã lưu model: {model_path}")
