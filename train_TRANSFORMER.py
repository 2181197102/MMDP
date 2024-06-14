import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification, AdamWeightDecay
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import tensorflow as tf

# 关闭oneDNN自定义操作
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 检查 GPU 是否可用
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 加载数据
data = pd.read_csv('./traindata.csv', low_memory=False)

# 数据预处理
data = data.dropna()

# 处理包含分隔符的特征，将其拆分为两个特征
def split_feature(df, column):
    split_cols = df[column].str.split('/', expand=True)
    df[column + '_part1'] = pd.to_numeric(split_cols[0], errors='coerce')
    df[column + '_part2'] = pd.to_numeric(split_cols[1], errors='coerce')
    df = df.drop(column, axis=1)
    return df

# 处理特征
data = split_feature(data, 'cDNA_position')
data = split_feature(data, 'CDS_position')
data = split_feature(data, 'Protein_position')

# 选择需要编码的特征
categorical_features = ['CHROM', 'Ensembl-Gene-ID', 'Ensembl-Protein-ID', 'Ensembl-Transcript-ID', 'Uniprot-Accession',
                        'REF-Nuc', 'ALT-Nuc', 'Amino_acids', 'Codons', 'Consequence', 'IMPACT', 'DOMAINS', 'ClinVar_preferred_disease_name_in_CLNDISDB']
label_encoders = {}

# 对每个分类特征进行Label Encoding
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature].astype(str))
    label_encoders[feature] = le

# 分离特征和目标变量
X = data.drop('True Label', axis=1)
y = data['True Label']

# 将特征转为字符串
X_str = X.astype(str).agg(' '.join, axis=1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_str, y, test_size=0.2, random_state=42)

# 加载本地的BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

# 对文本进行tokenization和编码
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

# 将数据转换为TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
)).batch(16).prefetch(tf.data.AUTOTUNE)

# 加载本地的BERT模型
model = TFBertForSequenceClassification.from_pretrained('./bert-base-uncased', num_labels=2)

# 编译模型
optimizer = AdamWeightDecay(learning_rate=3e-5, weight_decay_rate=0.01)  # 使用 transformers 的优化器
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # 使用 TensorFlow 的损失函数
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 设置早停
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 手动添加进度条
epochs = 3
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    with tqdm(total=len(train_dataset)) as pbar:
        for step, (batch_data, batch_labels) in enumerate(train_dataset):
            model.train_on_batch(batch_data, batch_labels)
            pbar.update(1)
        val_loss, val_accuracy = model.evaluate(test_dataset)
        print(f'Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')

# 评估模型
loss, accuracy = model.evaluate(test_dataset)
print(f'Test Accuracy: {accuracy:.2f}')

# 绘制训练过程中的损失和准确率曲线
import matplotlib.pyplot as plt

# 假设保存了训练过程中的损失和准确率，可以使用 matplotlib 绘制这些数据
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.show()
