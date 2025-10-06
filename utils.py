import numpy as np

def normalize_height(height_cm):
    """将原始身高(cm)标准化到0-100分"""
    score = (height_cm - 150) / (185 - 150) * 100
    return np.clip(score, 0, 100)


def normalize_wealth(wealth_cny):
    """将原始财富(CNY)标准化到0-100分"""
    score = wealth_cny / 100_000_000 * 100
    return np.clip(score, 0, 100)


def calculate_ground_truth_score(height_score, wealth_score, looks_score):
    """根据预设权重计算“事实标准”分数"""
    weights = {'height': 0.25, 'wealth': 0.45, 'looks': 0.30}
    return (weights['height'] * height_score +
            weights['wealth'] * wealth_score +
            weights['looks'] * looks_score)



