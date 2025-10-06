import numpy as np
import matplotlib


from scipy.interpolate import make_interp_spline
from scorer import Scorer
from utils import normalize_height, normalize_wealth

matplotlib.use('TkAgg')  # 必须在 import
import matplotlib.pyplot as plt

# (请确保你的 scorer.py 和 utils.py 在同一目录下, 或者使用下面的简单模拟)
# class Scorer:
#     def calculate_score(self, h, w, l): return (normalize_height(h) + normalize_wealth(w) + np.clip(l, 0, 100)) / 3
# def normalize_height(h): return np.clip((h - 160) / 20 * 100, 0, 100)
# def normalize_wealth(w): return np.clip(np.log10(w + 1) / np.log10(100000000) * 100, 0, 100)


def plot_candidate_score_cool(height_cm, wealth_cny, looks_score):
    """
    绘制候选人分数的平滑面积图 (Cooler Version)
    """
    # 1️⃣ 初始化和计算分数
    scorer = Scorer()
    height_s = normalize_height(height_cm)
    wealth_s = normalize_wealth(wealth_cny)
    looks_s = np.clip(looks_score, 0, 100)
    total_score = scorer.calculate_score(height_cm, wealth_cny, looks_score)

    labels = ['Height', 'Wealth', 'Looks', 'Total Score']
    values = np.array([height_s, wealth_s, looks_s, total_score])

    # 2️⃣ 创建平滑曲线
    x = np.arange(len(labels))
    x_smooth = np.linspace(x.min(), x.max(), 300)  # 创建更密集的x轴用于平滑
    spl = make_interp_spline(x, values, k=3)  # k=3 表示三次样条插值
    y_smooth = spl(x_smooth)

    # 3️⃣ 开始绘图
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制平滑曲线
    ax.plot(x_smooth, y_smooth, color='#ff6347', linewidth=3)

    # 填充面积并制造渐变效果
    # 注意：matplotlib本身不支持直接的渐变填充，但可以通过这种方式模拟
    ax.fill_between(x_smooth, y_smooth, color='#ff6347', alpha=0.3)
    ax.fill_between(x_smooth, y_smooth * 0.6, color='#ff6347', alpha=0.3)  # 叠加一层模拟渐变
    ax.fill_between(x_smooth, y_smooth * 0.3, color='#ff6347', alpha=0.3)

    # 标记原始数据点
    ax.scatter(x, values, color='white', s=100, zorder=5, ec='#ff6347', lw=2)

    # 在数据点上显示数值
    for i, txt in enumerate(values):
        ax.text(i, values[i] + 4, f'{txt:.1f}', ha='center', color='white', fontsize=12)

    # 4️⃣ 美化图表
    ax.set_ylim(0, 120)
    ax.set_ylabel("Score (0-100)", fontsize=14, color='gray')
    ax.set_title("📈 Candidate Score Flow 📈", fontsize=18, color='white', weight='bold')
    ax.text(0.5, 1.05, f"Height: {height_cm}cm, Wealth: {wealth_cny:,} CNY, Looks: {looks_score}",
            horizontalalignment='center', fontsize=12, color='lightgray', transform=ax.transAxes)

    # 设置x轴
    plt.xticks(x, labels, fontsize=12, color='white')

    # 隐藏边框和调整网格
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(False)  # 通常在面积图中隐藏x轴网格

    # 调整背景色
    fig.patch.set_facecolor('#2e3440')
    ax.set_facecolor('#2e3440')

    plt.tight_layout()
    plt.show()

# --- 示例调用 ---
# plot_candidate_score_area(height_cm=185, wealth_cny=5000000, looks_score=85)