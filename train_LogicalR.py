import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 读取特征重要性文件
feature_importance_df = pd.read_csv('RF_feature_importance.csv')

# 选择最重要的前20个特征（根据具体需求调整数量）
top_features = feature_importance_df['Feature'].head(20).tolist()
print("Top features selected for optimization:", top_features)


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
X = data[top_features]
y = data['True Label']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建逻辑回归模型
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# 评估模型
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
