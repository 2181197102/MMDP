import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 关闭oneDNN自定义操作
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建模型
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 设置早停
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# 绘制训练过程中的损失和准确率曲线
import matplotlib.pyplot as plt

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
