import pandas as pd  # 导入 pandas，用于数据处理和DataFrame操作
import numpy as np  # 导入 numpy，用于数值计算和生成随机数据
from utils import normalize_height, normalize_wealth, calculate_ground_truth_score


# 从 utils.py 导入工具函数：身高标准化、财富标准化、计算最终分数

def generate_dataset(num_samples=1000, filename="GaoFuShai_Dataset.xlsx"):
    """生成并保存合成数据集"""

    print(f"正在生成 {num_samples} 条合成数据...")  # 打印提示信息

    np.random.seed(42)  # 设置随机种子，保证每次生成的数据一致（可复现）

    # -----------------------------
    # 生成身高数据，服从正态分布
    # 均值175，标准差7，生成 num_samples 条数据
    # 再用 np.clip 限制范围在150~200 cm之间
    # -----------------------------
    heights = np.clip(np.random.normal(175, 7, num_samples), 150, 200)

    # -----------------------------
    # 生成财富数据，服从对数正态分布，模拟财富分布不均
    # mean=15, sigma=2
    # 限制最大财富为3亿 CNY
    # -----------------------------
    wealth = np.clip(np.random.lognormal(mean=15, sigma=2, size=num_samples), 0, 300_000_000)

    # -----------------------------
    # 生成颜值数据，服从正态分布
    # 均值60，标准差15，限制在0~100分之间
    # -----------------------------
    looks = np.clip(np.random.normal(60, 15, num_samples), 0, 100)

    # -----------------------------
    # 将原始数据组成一个 DataFrame
    # 每一列分别为身高、财富、颜值
    # -----------------------------
    df = pd.DataFrame({
        'Raw_Height_cm': heights,
        'Raw_Wealth_CNY': wealth,
        'Raw_Looks_0_100': looks
    })

    # -----------------------------
    # 对身高进行标准化，映射到0~100分
    # 使用 utils.py 中的 normalize_height 函数
    # -----------------------------
    df['Height_Score'] = normalize_height(df['Raw_Height_cm'])

    # -----------------------------
    # 对财富进行标准化，映射到0~100分
    # 使用 utils.py 中的 normalize_wealth 函数
    # -----------------------------
    df['Wealth_Score'] = normalize_wealth(df['Raw_Wealth_CNY'])

    # -----------------------------
    # 颜值分数直接使用原始生成的分数
    # -----------------------------
    df['Looks_Score'] = df['Raw_Looks_0_100']

    # -----------------------------
    # 计算最终“高富帅分数”，作为标签
    # 根据预设权重计算身高、财富、颜值的综合分
    # 调用 utils.py 中的 calculate_ground_truth_score 函数
    # -----------------------------
    df['GaoFuShuai_Score'] = calculate_ground_truth_score(
        df['Height_Score'], df['Wealth_Score'], df['Looks_Score']
    )

    # -----------------------------
    # 将生成的数据集保存为 Excel 文件
    # index=False 表示不保存行索引
    # -----------------------------
    df.to_excel(filename, index=False)

    print(f"✅ 数据集已成功保存至 '{filename}'")  # 提示保存成功
    return df  # 返回生成的 DataFrame 对象
