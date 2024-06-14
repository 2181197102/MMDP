import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

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

# 构建随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 评估模型
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# 计算并打印特征重要性
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# 打印特征重要性
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {X.columns[indices[f]]} ({importances[indices[f]]})")

# 将特征重要性保存到文件
feature_importance_df = pd.DataFrame({
    'Feature': X.columns[indices],
    'Importance': importances[indices]
})
feature_importance_df.to_csv('RF_feature_importance.csv', index=False)

# 绘制特征重要性
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
