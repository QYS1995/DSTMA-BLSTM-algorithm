import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        self.shared_attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.task_specific_attention = [tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
                                        for _ in range(self.num_tasks)]
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
columns = ['NO(ppb)', 'NO2(ppb)', 'CO2(ppm)', 'CO(ppm)', 'CH4(ppm)', 'Pressure', 'Temperature',
           'Humidity', 'WindSpeed', 'WindDirections', 'TrafficCounts', 'DieselCounts', 'GasselCounts']

# 使用 pandas 读取数据文件
data_matrix = pd.read_csv(file_path, delimiter=',')
data_matrix.columns = columns

# 定义输入特征和输出目标列索引
input_columns = [5, 6, 7, 8, 9, 10, 11, 12]  # 对应气象和交通相关特征
target_column = [0, 1, 2, 3, 4]  # 对应污染物列

# 分别对输入特征和输出目标进行归一化
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

# 分别对输入和输出进行归一化
scaled_inputs = input_scaler.fit_transform(data_matrix.iloc[:, input_columns].values)
scaled_outputs = output_scaler.fit_transform(data_matrix.iloc[:, target_column].values)

# 将归一化后的输入和输出组合在一起
scaled_data = np.hstack([scaled_inputs, scaled_outputs])

# 定义函数来创建输入和输出序列，用于 LSTM 模型的训练
def create_sequences(data, seq_length, input_columns, target_column):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, input_columns])
        y.append(data[i + seq_length, target_column])
    return np.array(X), np.array(y)

# 设置输入序列的长度
# 使用前连续seq_length个时间步的数据作为输入。模型会基于这seq_length个时间步的数据来预测下一个时间步的目标值
seq_length = 10

# 生成输入和输出序列
X, y = create_sequences(scaled_data, seq_length, input_columns, target_column)

# 将数据集分为训练集和测试集，80% 用于训练，20% 用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 加载预训练的最佳模型
checkpoint_path = 'best_model_checkpoint.h5'  # 最佳模型的路径
model = tf.keras.models.load_model(checkpoint_path, custom_objects={'DynamicSharedTaskSpecificAttention': DynamicSharedTaskSpecificAttention})

# 定义 EarlyStopping 回调函数
early_stopping = EarlyStopping(
    monitor='val_loss',         # 监控验证集的损失
    patience=80,                # 如果验证集损失在80 个 epoch 内没有改善，停止训练
    mode='min',                 # 选择最小化验证集损失
    restore_best_weights=True   # 在训练结束后恢复最优模型的权重
)

# 继续使用 TensorBoard 来监控训练过程
tensorboard_callback = TensorBoard(log_dir='./logs')

# 训练模型
history = model.fit(
    X_train,
    [y_train[:, i] for i in range(len(target_column))],
    epochs=150,                  # 训练 350 个 epochs
    batch_size=150,              # 批次大小为 50
    validation_data=(X_test, [y_test[:, i] for i in range(len(target_column))]),
    callbacks=[early_stopping, tensorboard_callback]
)

# 使用训练好的模型对测试数据进行预测
y_pred = model.predict(X_test)

# 反归一化预测值和真实值，转换回原始的污染物单位
y_test_inv, y_pred_inv = [], []
for i in range(len(target_column)):
    y_test_column = y_test[:, i].reshape(-1, 1)
    y_pred_column = y_pred[i].reshape(-1, 1)

    # 使用 output_scaler 的 min_ 和 max_ 参数进行反归一化
    min_val = output_scaler.data_min_[i]
    max_val = output_scaler.data_max_[i]

    # 反归一化公式: 原始值 = 归一化值 * (X_max - X_min) + X_min
    y_test_inv_column = y_test_column * (max_val - min_val) + min_val
    y_pred_inv_column = y_pred_column * (max_val - min_val) + min_val

    # 将反归一化的二维数组变为一维数组并存储
    y_test_inv.append(y_test_inv_column.flatten())
    y_pred_inv.append(y_pred_inv_column.flatten())

# 1. 绘制整体的训练和验证损失曲线
loss_data = pd.DataFrame({
    'Epoch': range(1, len(history.history['loss']) + 1),
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
})
loss_data.to_csv('training_validation_loss.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 2. 导出每个污染物的预测值与真实值对比数据，并进行残差分析
# 创建一个列表来保存全局误差指标
error_metrics = []

# 计算并保存每个污染物的误差指标、残差分析结果，并导出到一个 CSV 文件
for i in range(len(target_column)):
    pollutant_name = columns[target_column[i]]  # 获取污染物名称
    true_values = y_test_inv[i]  # 反归一化后的真实值
    predicted_values = y_pred_inv[i]  # 反归一化后的预测值
    residuals = true_values - predicted_values  # 计算残差

    # 计算全局误差指标（基于整个预测集）
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    # **将误差指标添加到 error_metrics 列表中**
    error_metrics.append([pollutant_name, mse, rmse, mae, r2])

    # 将残差和预测结果保存为 CSV
    comparison_data = pd.DataFrame({
        'Time Steps': range(1, len(true_values) + 1),
        f'True {pollutant_name}': true_values,
        f'Predicted {pollutant_name}': predicted_values,
        f'Residuals {pollutant_name}': residuals
    })
    comparison_data.to_csv(f'{pollutant_name}_predictions_residuals.csv', index=False)

    # 绘制预测值与真实值对比图
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label=f'True {pollutant_name}')
    plt.plot(predicted_values, label=f'Predicted {pollutant_name}')
    plt.title(f'{pollutant_name} - Predicted vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{pollutant_name} Concentration')
    plt.legend()
    plt.show()

    # 绘制残差直方图
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'{pollutant_name} Residuals Histogram')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

    # 保存残差直方图数据为 CSV 文件
    hist_data = pd.DataFrame({
        'Bins': bins[:-1],  # bins 中的最后一个值是右边界，不需要
        'Frequency': counts
    })
    hist_data.to_csv(f'{pollutant_name}_residuals_histogram.csv', index=False)

# **将全局误差指标保存为一个单独的 CSV 文件**
error_metrics_df = pd.DataFrame(error_metrics, columns=['Pollutant', 'MSE', 'RMSE', 'MAE', 'R²'])
error_metrics_df.to_csv('error_metrics_analysis.csv', index=False)

# 3. 特征重要性分析：通过 permutation importance 方法评估输入特征的重要性
class KerasRegressorWrapper:
    """
    将 Keras 模型包装为一个可用于特征重要性计算的回归器
    """
    def __init__(self, model, seq_length, n_features, output_index):
        self.model = model
        self.seq_length = seq_length
        self.n_features = n_features
        self.output_index = output_index

    def fit(self, X, y):
        # 拟合模型（未真正使用，但方法需要存在）
        self.model.fit(X.reshape((X.shape[0], self.seq_length, self.n_features)), y, epochs=1, batch_size=32)
        return self

    def predict(self, X):
        # 预测输出，并返回指定任务的预测值
        X_reshaped = X.reshape((X.shape[0], 1, self.n_features))
        predictions = self.model.predict(X_reshaped)
        return predictions[self.output_index].flatten()

# 计算并可视化每个污染物的特征重要性
all_importances = []  # 存储所有污染物的特征重要性

for target_index in range(len(target_column)):
    # 包装模型，使其适合用于特征重要性计算
    model_wrapper = KerasRegressorWrapper(model, seq_length, X.shape[2], output_index=target_index)
    # 仅使用最后一个时间步的特征进行计算
    X_test_last_step = X_test[:, -1, :]

    # 使用 permutation importance 计算特征重要性
    results = permutation_importance(
        model_wrapper, X_test_last_step, y_test[:, target_index],
        n_repeats=10, random_state=42, scoring='neg_mean_squared_error'
    )

    # 保存特征重要性结果为 CSV 文件
    all_importances.append(results.importances_mean)
    feature_names = [columns[i] for i in input_columns]
    df = pd.DataFrame({'Feature': feature_names, 'Importance': results.importances_mean})
    df.to_csv(f'feature_importances_for_{columns[target_index]}.csv', index=False)

# 计算每个污染物的平均特征重要性
all_importances = np.array(all_importances)
mean_importances = np.mean(all_importances, axis=0)

# 保存整体平均特征重要性为 CSV 文件
df_avg = pd.DataFrame({'Feature': feature_names, 'Average Importance': mean_importances})
df_avg.to_csv('overall_average_feature_importances.csv', index=False)

# 可视化每个污染物的特征重要性
for target_index in range(len(target_column)):
    feature_names = [columns[i] for i in input_columns]
    importances = all_importances[target_index]
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Mean Decrease in Accuracy")  # 横轴标签
    plt.title(f"Feature Importances for {columns[target_column[target_index]]}")  # 图标题
    plt.show()

# 可视化整体的平均特征重要性
plt.figure(figsize=(10, 6))
indices = np.argsort(mean_importances)
plt.barh(range(len(indices)), mean_importances[indices], align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Mean Decrease in Accuracy")  # 横轴标签
plt.title("Overall Average Feature Importances")  # 图标题
plt.show()

# 清理内存，释放资源
gc.collect()
tf.keras.backend.clear_session()
