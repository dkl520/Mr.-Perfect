

# 高富帅分数测评机器 (README.md)

## 项目简介

本项目实现了一个基于人工智能的“高富帅”分数评估引擎。它使用一个机器学习模型（Scikit-learn 的 `SGDRegressor`）来学习一个预定义的评分逻辑，该逻辑基于身高、财富和颜值三个维度对个体进行打分。

项目首先会生成一个包含1000条记录的合成数据集，然后利用该数据集训练模型，并最终提供一个简单的接口，用于对新的输入数据进行评分预测。

## 主要功能

  * **数据生成**: 自动创建一个名为 `GaoFuShuai_Dataset.xlsx` 的 Excel 文件，其中包含符合特定统计分布的模拟数据。
  * **模型训练**: 使用生成的数据集训练一个随机梯度下降回归（SGD Regressor）模型。
  * **模型持久化**: 将训练好的模型 (`gaofushuai_model.pkl`) 和特征缩放器 (`scaler.pkl`) 保存到本地，以便未来直接使用，无需重复训练。
  * **分数预测**: 提供一个 `Scorer` 类，可以轻松加载已保存的模型，并根据输入的原始身高（厘米）、财富（人民币）和颜值（0-100分）来预测最终的“高富帅”分数。

## 环境要求与安装

本项目基于 Python 3 编写，需要安装以下第三方库：

  * pandas
  * numpy
  * scikit-learn
  * openpyxl (用于读写 Excel 文件)
  * joblib (用于保存和加载模型)

你可以通过以下命令使用 pip 一次性安装所有依赖：

```bash
pip install pandas numpy scikit-learn openpyxl joblib
```

## 运行流程

1.  **保存代码**: 将项目报告第四节中提供的完整 Python 代码保存为一个文件，例如 `scorer.py`。

2.  **执行脚本**: 在你的终端或命令行中，导航到 `scorer.py` 文件所在的目录，然后运行以下命令：

    ```bash
    python main.py
    ```

3.  **首次运行**:

      * 脚本会自动检测当前目录下是否存在 `GaoFuShuai_Dataset.xlsx` 文件。如果不存在，它将生成该数据集。
      * 接着，脚本会加载数据集，训练机器学习模型。
      * 训练完成后，它会将模型保存为 `gaofushuai_model.pkl`，并将用于数据标准化的缩放器保存为 `scaler.pkl`。
      * 最后，脚本会演示几个预测示例，将结果打印在终端上。

4.  **后续运行**:

      * 如果再次运行 `python scorer.py`，脚本会发现数据集和模型文件已经存在，因此会跳过数据生成和训练步骤，直接加载已保存的模型进行预测演示。这大大加快了执行速度。

## 文件说明

当您运行脚本后，项目目录下会生成以下几个关键文件：

  * `scorer.py`: 你的主程序文件，包含了数据生成、模型训练和预测的所有逻辑。
  * `GaoFuShuai_Dataset.xlsx`: 自动生成的Excel格式训练数据集，包含1000条样本数据。
  * `gaofushuai_model.pkl`: 经过训练并序列化保存的机器学习模型文件。
  * `scaler.pkl`: 用于对输入特征进行标准化的缩放器对象文件。~~A~~
~~~~
-----

# "Mr. Perfect" Score Evaluator (README.md)

## Project Overview

This project implements an AI-based evaluation engine for a "Mr. Perfect" (GaoFuShuai) score. It utilizes a machine learning model (`SGDRegressor` from Scikit-learn) to learn a predefined scoring logic based on three key attributes: height, wealth, and physical appearance (looks).

The project first generates a synthetic dataset of 1000 records, then uses this dataset to train the model, and finally provides a simple interface to predict scores for new inputs.

## Features

  * **Dataset Generation**: Automatically creates an Excel file named `GaoFuShuai_Dataset.xlsx` containing simulated data that follows specific statistical distributions.
  * **Model Training**: Trains a Stochastic Gradient Descent (SGD) Regressor model using the generated dataset.
  * **Model Persistence**: Saves the trained model (`gaofushuai_model.pkl`) and the feature scaler (`scaler.pkl`) to disk, allowing for instant use in the future without retraining.
  * **Score Prediction**: Provides a `Scorer` class that easily loads the saved model and predicts a final "Mr. Perfect" score based on raw inputs for height (cm), wealth (CNY), and looks (0-100).

## Requirements and Installation

This project is written in Python 3 and requires the following third-party libraries:

  * pandas
  * numpy
  * scikit-learn
  * openpyxl (for reading/writing Excel files)
  * joblib (for saving/loading the model)

You can install all dependencies at once using pip with the following command:

```bash
pip install pandas numpy scikit-learn openpyxl joblib
```

## How to Run

1.  **Save the Code**: Save the complete Python code provided in Section 4 of the project report into a file, for example, `scorer.py`.

2.  **Execute the Script**: Open your terminal or command prompt, navigate to the directory where you saved `scorer.py`, and run the following command:

    ```bash
    python main.py
    ```

3.  **First Run**:

      * The script will check if `GaoFuShuai_Dataset.xlsx` exists in the current directory. If not, it will generate it.
      * Next, the script will load the dataset and train the machine learning model.
      * Upon completion of training, it will save the model as `gaofushuai_model.pkl` and the data scaler as `scaler.pkl`.
      * Finally, it will demonstrate several prediction examples by printing the results to the console.

4.  **Subsequent Runs**:

      * If you run `python scorer.py` again, the script will detect that the dataset and model files already exist. It will skip the generation and training steps and proceed directly to loading the saved model for the prediction demonstration. This makes subsequent executions much faster.

## File Descriptions

After running the script, the following key files will be present in your project directory:

  * `scorer.py`: Your main program file, containing all logic for data generation, model training, and prediction.
  * `GaoFuShuai_Dataset.xlsx`: The auto-generated training dataset in Excel format, containing 1000 data samples.
  * `gaofushuai_model.pkl`: The serialized, trained machine learning model file.
  * `scaler.pkl`: The scaler object file used to standardize input features.