import numpy as np
import pandas as pd
import joblib
from utils import normalize_height, normalize_wealth

class Scorer:
    """封装模型加载与预测逻辑"""

    def __init__(self, model_path='gaofushuai_model.pkl', scaler_path='scaler.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        except FileNotFoundError:
            print("❌ 错误: 未找到已训练的模型或缩放器文件。请先运行训练流程。")
            self.model = None
            self.scaler = None

    def calculate_score(self, height_cm, wealth_cny, looks_score):
        """输入原始值计算高富帅分数"""
        if not self.model or not self.scaler:
            return None

        # 1. 标准化输入
        height_s = normalize_height(height_cm)
        wealth_s = normalize_wealth(wealth_cny)
        looks_s = np.clip(looks_score, 0, 100)

        # 2. 用 DataFrame 构造特征向量（带列名，避免警告）
        input_features = pd.DataFrame(
            [[height_s, wealth_s, looks_s]],
            columns=['Height_Score', 'Wealth_Score', 'Looks_Score']
        )

        # 3. 使用已训练的缩放器进行缩放
        input_scaled = self.scaler.transform(input_features)

        # 4. 使用模型进行预测
        predicted_score = self.model.predict(input_scaled)

        # return  predicted_score
        # 5. 返回结果，并确保在0-100范围内
        # print(predicted_score)
        return np.clip(predicted_score, 0, 100)[0]
