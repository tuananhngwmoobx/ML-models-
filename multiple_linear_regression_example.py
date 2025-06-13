import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
np.random.seed(42)
n_samples = 100

# Tạo các biến độc lập
X1 = np.random.normal(0, 1, n_samples)  # Diện tích nhà
X2 = np.random.normal(0, 1, n_samples)  # Số phòng ngủ
X3 = np.random.normal(0, 1, n_samples)  # Khoảng cách đến trung tâm

# Tạo biến phụ thuộc (giá nhà)
y = 2*X1 + 1.5*X2 - 0.5*X3 + np.random.normal(0, 0.5, n_samples)

# Tạo DataFrame
data = pd.DataFrame({
    'DienTich': X1,
    'SoPhongNgu': X2,
    'KhoangCach': X3,
    'GiaNha': y
})

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = data[['DienTich', 'SoPhongNgu', 'KhoangCach']]
y = data['GiaNha']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo và huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Hệ số hồi quy:")
print(f"β₀ (intercept): {model.intercept_:.4f}")
print(f"β₁ (DienTich): {model.coef_[0]:.4f}")
print(f"β₂ (SoPhongNgu): {model.coef_[1]:.4f}")
print(f"β₃ (KhoangCach): {model.coef_[2]:.4f}")
print(f"\nR-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# Vẽ đồ thị so sánh giá trị thực tế và dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Giá nhà thực tế')
plt.ylabel('Giá nhà dự đoán')
plt.title('So sánh giá trị thực tế và dự đoán')
plt.tight_layout()
plt.show() 