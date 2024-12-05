import os
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Lambda, Bidirectional
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras_tuner import HyperModel, BayesianOptimization
import gc

# 关闭TensorFlow的oneDNN优化，确保兼容性
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 忽略警告信息
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 读取数据
file_path = 'Data_matrix-1.txt'  # 数据文件路径
# 如果文件未找到，抛出异常提醒用户
try:
    data_matrix = pd.read_csv(file_path, delimiter=',')
except FileNotFoundError:
    raise Exception(f"文件 {file_path} 未找到，请检查路径是否正确。")

# 定义数据集的列名
columns = ['NO(ppb)', 'NO2(ppb)', 'CO2(ppm)', 'CO(ppm)', 'CH4(ppm)', 'Pressure', 'Temperature',
           'Humidity', 'WindSpeed', 'WindDirections', 'TrafficCounts', 'DieselCounts', 'GasselCounts']

# 设置数据框的列名
data_matrix.columns = columns

# 分别对输入特征和输出目标进行特征缩放，将输入特征和输出目标的特征值分别缩放到[0, 1]的范围内

# 对输入特征进行归一化
input_columns = [5, 6, 7, 8, 9, 10, 11, 12]  # 气象和交通相关特征
input_scaler = MinMaxScaler()
scaled_inputs = input_scaler.fit_transform(data_matrix.iloc[:, input_columns].values)

# 对目标污染物浓度进行归一化
target_column = [0, 1, 2, 3, 4]  # 污染物相关特征
output_scaler = MinMaxScaler()
scaled_outputs = output_scaler.fit_transform(data_matrix.iloc[:, target_column].values)

# 将归一化后的输入特征和输出目标合并
scaled_data = np.hstack([scaled_inputs, scaled_outputs])

# 定义一个函数，用于创建输入序列和输出目标
def create_sequences(data, seq_length, input_columns, target_column):
    X, y = [], []  # X用于存储输入序列，y用于存储对应的输出目标
    for i in range(len(data) - seq_length):
        # 输入序列是连续的seq_length时间步
        X.append(data[i:i + seq_length, input_columns])
        # 输出是下一时间步的目标值
        y.append(data[i + seq_length, target_column])
    return np.array(X), np.array(y)

# 设置输入序列长度为10
# 使用前连续seq_length个时间步的数据作为输入。模型会基于这seq_length个时间步的数据来预测下一个时间步的目标值
seq_length = 10

# 调用create_sequences函数，生成模型的输入X和输出y
X, y = create_sequences(scaled_data, seq_length, input_columns, target_column)

# 将数据集划分为训练集和测试集，80%用于训练，20%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 自定义动态共享与专属多头注意力机制
class DynamicSharedTaskSpecificAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, num_tasks, **kwargs):
        # 初始化共享和任务专属多头注意力层
        super(DynamicSharedTaskSpecificAttention, self).__init__(**kwargs)
        self.num_heads = num_heads  # 多头注意力的头数
        self.key_dim = key_dim  # key的维度
        self.num_tasks = num_tasks  # 任务数量
        # 定义共享多头注意力层
        self.shared_attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        # 定义每个任务专属的多头注意力层
        self.task_specific_attention = [tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
                                        for _ in range(self.num_tasks)]
        # 定义动态权重的全连接层，并使用softmax激活函数
        self.dynamic_weight_dense = Dense(self.num_tasks, activation='softmax')

    def call(self, inputs):
        # 计算共享注意力层的输出
        shared_output = self.shared_attention(inputs, inputs)
        # 计算每个任务专属的注意力输出
        task_specific_outputs = [attention_layer(inputs, inputs) for attention_layer in self.task_specific_attention]
        # 计算动态权重，并对其扩展维度
        dynamic_weights = self.dynamic_weight_dense(shared_output)
        dynamic_weights = tf.expand_dims(dynamic_weights, axis=-1)
        # 计算最终输出，将动态权重与任务专属输出进行加权
        final_outputs = [dynamic_weights[:, :, i, :] * task_output for i, task_output in enumerate(task_specific_outputs)]
        return final_outputs

# 定义超参数优化模型类，继承自HyperModel
class MyHyperModel(HyperModel):
    def build(self, hp):
        # 输入层，输入序列的形状为 (seq_length, n_features)
        input_layer = Input(shape=(seq_length, X.shape[2]))

        # 第一层双向LSTM层，单元数量在32到1024之间可调
        lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=1024, step=32)
        lstm_out = Bidirectional(LSTM(units=lstm_units_1, return_sequences=True))(input_layer)
        # 添加丢弃层，丢弃率在0.0到0.5之间可调
        lstm_out = Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1))(lstm_out)

        # 第二层双向LSTM层，单元数量在32到1024之间可调
        lstm_units_2 = hp.Int('lstm_units_2', min_value=32, max_value=1024, step=32)
        lstm_out = Bidirectional(LSTM(units=lstm_units_2, return_sequences=True))(lstm_out)
        # 添加丢弃层
        lstm_out = Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1))(lstm_out)

        # 定义共享与专属多头注意力机制
        num_heads = hp.Int('num_heads', min_value=1, max_value=16, step=1)
        attention_layer = DynamicSharedTaskSpecificAttention(num_heads=num_heads, key_dim=X.shape[2], num_tasks=len(target_column))
        # 获取每个任务的输出
        attention_outputs = attention_layer(lstm_out)

        # 输出层，每个任务输出一个值
        outputs = [Dense(1)(Lambda(lambda x: x[:, -1, :])(attention_output)) for attention_output in attention_outputs]

        # 构建模型
        model = Model(inputs=input_layer, outputs=outputs)
        # 设置学习率，范围在1e-4到1e-2之间
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        # 编译模型，使用Adam优化器和均方误差损失函数
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        return model

# 定义超参数优化器 (BayesianOptimization)  搜索轮数 max_trials* executions_per_trial
hypermodel = MyHyperModel()
tuner = BayesianOptimization(
    hypermodel,                   # 传入自定义的超参数模型
    objective='val_loss',          # 优化的目标是验证集损失
    max_trials=30,                  # 最多进行2次不同的超参数组合搜索
    executions_per_trial=2,        # 每个超参数组合执行两次，取平均值
    directory='my_dir',            # 保存调优过程中结果的目录
    project_name='hyperparam_tuning_lstm',  # 项目名称，用于区分不同的超参数搜索项目
    overwrite=True                 # 如果目录已存在，则覆盖原有内容
)

# 定义保存最佳模型的文件路径
checkpoint_path = 'best_model_checkpoint.h5'

# 添加 ModelCheckpoint 回调，在超参数搜索过程中保存验证损失最小的模型
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,    # 只保存验证损失最小的模型
    monitor='val_loss',     # 监控验证损失
    mode='min',             # 验证损失减小时保存模型
    verbose=1,              # 打印提示信息
    save_weights_only=False # 保存整个模型，而不是仅保存权重
)

# 运行超参数搜索
tuner.search(X_train, [y_train[:, i] for i in range(len(target_column))],
             epochs=150,        # 每次搜索时训练10个epoch
             validation_data=(X_test, [y_test[:, i] for i in range(len(target_column))]),
             callbacks=[checkpoint_callback])  # 添加checkpoint回调

# 获取最佳超参数
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

# 清理内存，防止内存泄漏
gc.collect()
tf.keras.backend.clear_session()

# 使用最佳超参数重新构建模型
model = hypermodel.build(best_hp)

# 使用最佳模型保存路径保存模型
model.save(checkpoint_path)
