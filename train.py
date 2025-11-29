import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('yoga_keypoints.csv')

X = data.drop(['Labels', 'Image_Name'], axis=1).values
y = data['Labels'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


with open('models/yoga_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved model to models/yoga_rf_model.pkl")
