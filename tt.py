import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

# =============================
# تنظیمات اپ
# =============================
st.set_page_config(page_title="Modeling App", layout="wide")
st.title("📊 Modeling App (SVR, MLP, Ensemble)")

# =============================
# آپلود فایل اکسل
# =============================
uploaded_file = st.file_uploader("یک فایل اکسل شامل داده‌های آموزشی آپلود کنید:", type=["xlsx"])

if uploaded_file is not None:
    dataset = pd.read_excel(uploaded_file)
    dataset = dataset.drop(dataset.columns[0], axis=1)

    X = dataset.iloc[:, 0:18].values
    y = dataset.iloc[:, 18:19].values

    # فقط داده‌های آموزش (۸۰٪) با random_state ثابت
    x_train, _, y_train, _ = train_test_split(X, y, train_size=0.8, random_state=21)

    # نرمال‌سازی
    scaler_x = StandardScaler().fit(x_train)
    x_train_scaled = scaler_x.transform(x_train)
    x_total_scaled = scaler_x.transform(X)

    scaler_y = StandardScaler().fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)
    y_total_scaled = scaler_y.transform(y)

    # =============================
    # آموزش مدل‌ها فقط با train
    # =============================
    Svr = SVR(kernel='rbf', C=15, gamma=0.003164, epsilon=0.10)
    Svr.fit(x_train_scaled, y_train_scaled.ravel())

    mlp = MLPRegressor(hidden_layer_sizes=(19,), activation='logistic',
                       solver='adam', alpha=0.0006, learning_rate_init=0.0005,
                       max_iter=875, early_stopping=True, random_state=22)
    mlp.fit(x_train_scaled, y_train_scaled.ravel())

    # =============================
    # گرفتن ورودی ۱۸ متغیر از کاربر
    # =============================
    st.subheader("🔢 ورود مقادیر 18 متغیر")
    feature_names = [
        "Water Bodies", "lawn", "Flower", "Plants and Shurbs",
        "Trees", "Sky View", "Soft Landscape", "Sitting Equipments",
        "Trees Density", "Natural Stone", "Elements and Sculpture",
        "Informal Designing", "Vegetation Diversity", "Color Diversity",
        "Fewer Buildings", "Shade Roof", "Less Floor, Paths & Stairs",
        "Environmental Equipments"
    ]

    user_input = []
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        value = cols[i % 3].number_input(f"{feat}", value=0.0, step=0.1)
        user_input.append(value)

    if st.button("🔮 Predict"):
        x_input = np.array(user_input).reshape(1, -1)
        x_input_scaled = scaler_x.transform(x_input)

        # پیش‌بینی‌ها
        y_pred_svr = scaler_y.inverse_transform(Svr.predict(x_input_scaled).reshape(-1, 1))[0, 0]
        y_pred_mlp = scaler_y.inverse_transform(mlp.predict(x_input_scaled).reshape(-1, 1))[0, 0]
        y_pred_ensemble = (y_pred_svr + y_pred_mlp) / 2

        st.subheader("📌 Prediction Results")
        st.write(f"**SVR Prediction:** {y_pred_svr:.3f}")
        st.write(f"**MLP Prediction:** {y_pred_mlp:.3f}")
        st.write(f"**SVR & MLP Ensemble Prediction:** {y_pred_ensemble:.3f}")

        # =============================
        # نمایش نمودارها
        # =============================

        # ۱) R² مقایسه‌ای

