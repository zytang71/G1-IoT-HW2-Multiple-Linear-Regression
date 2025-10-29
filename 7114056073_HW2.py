import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE

# --------資料載入與前處理--------

# 讀取資料集
file_path = 'data/StudentPerformanceFactors.csv'
data = pd.read_csv(file_path)

# 檢查資料的前幾行
print(data.head())


# 轉換類別資料為數值資料 (Label Encoding)
label_encoder = LabelEncoder()
data['Parental_Involvement'] = label_encoder.fit_transform(data['Parental_Involvement'])
data['Access_to_Resources'] = label_encoder.fit_transform(data['Access_to_Resources'])
data['Extracurricular_Activities'] = label_encoder.fit_transform(data['Extracurricular_Activities'])
data['Motivation_Level'] = label_encoder.fit_transform(data['Motivation_Level'])
data['Internet_Access'] = label_encoder.fit_transform(data['Internet_Access'])
data['Family_Income'] = label_encoder.fit_transform(data['Family_Income'])
data['Teacher_Quality'] = label_encoder.fit_transform(data['Teacher_Quality'])
data['School_Type'] = label_encoder.fit_transform(data['School_Type'])
data['Peer_Influence'] = label_encoder.fit_transform(data['Peer_Influence'])
data['Learning_Disabilities'] = label_encoder.fit_transform(data['Learning_Disabilities'])
data['Parental_Education_Level'] = label_encoder.fit_transform(data['Parental_Education_Level'])
data['Distance_from_Home'] = label_encoder.fit_transform(data['Distance_from_Home'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])


# 檢查是否有缺失值
print(data.isnull().sum())

# 確定是否有做完Label Encoding


print(data.head())

# --------資料切分--------

# 定義特徵（X）和目標變數（y）
X = data.drop('Exam_Score', axis=1)
y = data['Exam_Score']

# 切分資料集為訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 顯示訓練集和測試集的大小
print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")

# --------建立線性回歸模型並訓練--------

model = LinearRegression()

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 計算MSE與R2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2: {r2}")

# --------特徵選擇--------

# 使用RFE進行特徵選擇
selector = RFE(model, n_features_to_select=5)
selector = selector.fit(X_train, y_train)

# 顯示選擇的特徵
print(f"Selected features: {X.columns[selector.support_]}")

# --------預測結果與信賴區間繪圖--------

# 預測
y_pred = model.predict(X_test)

# 繪製預測結果與真實值
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Prediction vs True Values")
plt.show()