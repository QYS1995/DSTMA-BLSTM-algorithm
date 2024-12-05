import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# 设置GPU内存按需分配，避免占用全部显存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 定义自定义层 DynamicSharedTaskSpecificAttention（共享与任务专属的多头注意力机制）
class DynamicSharedTaskSpecificAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, num_tasks, **kwargs):
        super(DynamicSharedTaskSpecificAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.num_tasks = num_tasks
        # 定义共享的多头注意力层
        self.shared_attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        # 定义每个任务专属的多头注意力层
        self.task_specific_attention = [tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
                                        for _ in range(self.num_tasks)]
        # 定义动态权重的全连接层，使用softmax激活
        self.dynamic_weight_dense = tf.keras.layers.Dense(self.num_tasks, activation='softmax')

    def call(self, inputs):
        shared_output = self.shared_attention(inputs, inputs)
        task_specific_outputs = [attention_layer(inputs, inputs) for attention_layer in self.task_specific_attention]
        dynamic_weights = self.dynamic_weight_dense(shared_output)
        dynamic_weights = tf.expand_dims(dynamic_weights, axis=-1)
        final_outputs = [dynamic_weights[:, :, i, :] * task_output for i, task_output in enumerate(task_specific_outputs)]
        return final_outputs

# 读取数据
file_path = 'Data_matrix-1.txt'  # 数据文件路径
# 定义列名（污染物和气象相关特征）
columns = ['NO(ppb)', 'NO2(ppb)', 'CO2(ppm)', 'CO(ppm)', 'CH4(ppm)', 'Pressure', 'Temperature',
           'Humidity', 'WindSpeed', 'WindDirections', 'TrafficCounts', 'DieselCounts', 'GasselCounts']

# 使用 pandas 读取数据文件
data_matrix = pd.read_csv(file_path, delimiter=',')
# 设置列名
data_matrix.columns = columns

# 数据预处理：使用 MinMaxScaler 进行归一化，将数据缩放到 [0, 1] 区间
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_matrix.values)

# 定义函数来创建输入和输出序列，用于 LSTM 模型的预测
def create_sequences(data, seq_length, input_columns, target_column):
    X, y = [], []  # X: 输入序列, y: 输出序列
    for i in range(len(data) - seq_length):
        # 收集输入序列
        X.append(data[i:i + seq_length, input_columns])
        # 收集输出目标
        y.append(data[i + seq_length, target_column])
    return np.array(X), np.array(y)

# 设置输入序列的长度
seq_length = 10
# 定义输入特征的列索引
input_columns = [5, 6, 7, 8, 9, 10, 11, 12]  # 对应气象和交通相关特征
# 定义输出目标（污染物）的列索引
target_column = [0, 1, 2, 3, 4]  # 对应污染物列

# 生成输入和输出序列
X, y = create_sequences(scaled_data, seq_length, input_columns, target_column)

# 加载预训练的最佳模型
checkpoint_path = 'best_model_checkpoint.h5'  # 最佳模型的路径
# 加载模型，并指定自定义层 DynamicSharedTaskSpecificAttention
model = tf.keras.models.load_model(
    checkpoint_path,
    custom_objects={'DynamicSharedTaskSpecificAttention': DynamicSharedTaskSpecificAttention}
)

# 使用训练好的模型对所有数据进行预测
y_pred = model.predict(X)

# 反归一化预测值和真实值，转换回原始的污染物单位
y_inv, y_pred_inv = [], []  # 存储反归一化后的预测值和真实值
for i in range(len(target_column)):
    # 选取 y 和 y_pred 中当前污染物的列
    y_column = y[:, i].reshape(-1, 1)
    y_pred_column = y_pred[i].reshape(-1, 1)

    # 反归一化，将预测值和真实值恢复到原始的尺度
    y_inv.append(scaler.inverse_transform(np.hstack([np.zeros((y_column.shape[0], scaled_data.shape[1] - 1)), y_column]))[:, -1])
    y_pred_inv.append(scaler.inverse_transform(np.hstack([np.zeros((y_pred_column.shape[0], scaled_data.shape[1] - 1)), y_pred_column]))[:, -1])

# 导出每个污染物的预测值与真实值对比数据，并进行残差分析
for i in range(len(target_column)):
    pollutant_name = columns[target_column[i]]  # 污染物名称
    true_values = y_inv[i]  # 反归一化后的真实值
    predicted_values = y_pred_inv[i]  # 反归一化后的预测值
    residuals = true_values - predicted_values  # 残差：真实值与预测值之差

    # 保存每个污染物的预测值与真实值的对比数据和残差数据为 CSV 文件
    # comparison_data = pd.DataFrame({
    #     'Time Steps': range(1, len(true_values) + 1),
    #     f'True {pollutant_name}': true_values,
    #     f'Predicted {pollutant_name}': predicted_values,
    #     f'Residuals {pollutant_name}': residuals
    # })
    # comparison_data.to_csv(f'{pollutant_name}_predictions_residuals.csv', index=False)

    # 绘制预测值与真实值对比图
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label=f'True {pollutant_name}')
    plt.plot(predicted_values, label=f'Predicted {pollutant_name}')
    plt.title(f'{pollutant_name} - Predicted vs Actual')  # 图标题
    plt.xlabel('Time Steps')  # 横轴标签
    plt.ylabel(f'{pollutant_name} Concentration')  # 纵轴标签
    plt.legend()  # 显示图例
    plt.show()

    # 绘制残差图
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title(f'{pollutant_name} - Residuals')  # 图标题
    plt.xlabel('Time Steps')  # 横轴标签
    plt.ylabel('Residuals')  # 纵轴标签
    plt.show()

# 清理内存，释放资源
gc.collect()
tf.keras.backend.clear_session()
