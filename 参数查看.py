import joblib

# 加载模型或缩放器
model = joblib.load('gaofushuai_model.pkl')
# scaler = joblib.load('scaler.pkl')

print("✅ 模型类型:", type(model))
# print("✅ 缩放器类型:", type(scaler))

# 如果想看模型的参数：
print("\n--- 模型参数 ---")
print(model.get_params())

# 如果想看模型的权重（系数）：
print("\n--- 模型权重 (coefficients) ---")
print(model.coef_)

print("\n--- 模型偏置 (intercept) ---")
print(model.intercept_)
