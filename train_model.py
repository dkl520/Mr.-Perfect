import joblib
from sklearn.linear_model import SGDRegressor   # 导入随机梯度下降回归器
from sklearn.preprocessing import StandardScaler # 导入标准化工具
import os

def train_model(df, model_filename="gaofushuai_model.pkl", scaler_filename="scaler.pkl"):
    """训练模型并保存"""

    print("\n开始训练模型...")  # 提示用户训练开始

    # -----------------------------
    # 选择用于训练的特征列
    # Height_Score, Wealth_Score, Looks_Score
    # -----------------------------
    features = ['Height_Score', 'Wealth_Score', 'Looks_Score']

    # -----------------------------
    # 目标列，也就是我们要预测的分数
    # GaoFuShuai_Score
    # -----------------------------
    target = 'GaoFuShuai_Score'

    # -----------------------------
    # 准备训练数据
    # X 是特征矩阵（DataFrame）
    # y 是目标向量
    # -----------------------------
    X = df[features]
    y = df[target]

    # -----------------------------
    # 标准化特征
    # 使用 StandardScaler 将每个特征缩放为均值为0，标准差为1
    # 对梯度下降类算法是最佳实践
    # -----------------------------
   # 对数据进行level 化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # fit_transform: 拟合并转换数据

    # -----------------------------
    # 初始化随机梯度下降回归器
    # max_iter=1000: 最大迭代次数
    # tol=1e-4: 收敛阈值
    # random_state=42: 随机种子，保证可复现
    # eta0=0.01: 初始学习率
    # -----------------------------
    model = SGDRegressor(max_iter=1000, tol=1e-4, random_state=42, eta0=0.01)

    # -----------------------------
    # 训练模型
    # 用标准化后的特征 X_scaled 和目标 y 拟合模型
    # -----------------------------
    model.fit(X_scaled, y)

    # -----------------------------
    # 保存训练好的模型和缩放器
    # model_filename: 模型文件名 (.pkl)
    # scaler_filename: 缩放器文件名 (.pkl)
    # 使用 joblib 保存，比 pickle 更高效，适合 sklearn 模型
    # -----------------------------
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)

    # 提示保存成功
    print(f"✅ 模型已保存为 '{model_filename}'，缩放器已保存为 '{scaler_filename}'")

    # 返回模型和缩放器对象，方便后续调用
    return model, scaler
