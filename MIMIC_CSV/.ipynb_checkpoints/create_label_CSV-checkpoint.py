### 1.获得特定标签集合的清洗后CSV
import pandas as pd

# 输入文件路径
input_file_path = '/Users/post/Desktop/mimic-cxr-2.0.0-negbio_all.csv'
# 输出文件路径
output_file_path = '/Users/post/Downloads/mimic-cxr-2.0.0-negbio_all_10label.csv'

# 需要保留和清洗的列
columns_to_keep = [
    'subject_id',
    'study_id',
    'Atelectasis',
    'Consolidation',
    'Cardiomegaly',
    'Edema',
    'Lung Lesion',
    'Lung Opacity',
    'Pleural Effusion',
    'Pneumonia',
    'Pneumothorax'
]

# --- 数据处理逻辑 ---
try:
    # 1. 读取CSV文件，只加载需要的列
    print(f"正在读取文件: {input_file_path}...")
    df = pd.read_csv(input_file_path, usecols=columns_to_keep)
    print(f"原始数据加载完成。行数: {len(df)}")

    # 2. 确定用于清洗的列（即除了 subject_id 和 study_id 的所有标签列）
    # 标签列通常是以 float 形式存储的，所以我们指定 dtype=float 来进行精确比较
    label_columns = columns_to_keep[2:]

    # 3. 执行数据清洗：删除标签列中任何一个包含 -1 的行
    # (df[label_columns] == -1.0) 创建一个布尔DataFrame，标记所有 -1.0 的位置
    # .any(axis=1) 检查每行是否存在至少一个 True (-1.0)
    # ~ 取反，保留没有 -1.0 的行
    rows_to_keep = ~(df[label_columns] == -1.0).any(axis=1)
    df_cleaned = df[rows_to_keep].copy() # 使用 .copy() 避免 SettingWithCopyWarning

    print(f"数据清洗完成。已删除 {len(df) - len(df_cleaned)} 行包含 -1.0 的数据。")
    print(f"剩余行数: {len(df_cleaned)}")

    # 4. 将清理后的数据保存到新的CSV文件
    df_cleaned.to_csv(output_file_path, index=False)

    print(f"\n✅ 文件已成功处理。清理后的数据保存在：{output_file_path}")

except FileNotFoundError:
    print(f"错误：找不到文件 {input_file_path}，请检查路径是否正确。")
except ValueError as e:
    # 捕获可能由于列名不存在导致的错误
    print(f"错误：读取文件时发生错误，可能是列名配置不正确。详细信息: {e}")
except Exception as e:
    print(f"发生未知错误: {e}")

### 2.加入normal列，生成最终的文件
import pandas as pd
import numpy as np

# 1. 定义文件路径
input_file = "/Users/post/Downloads/mimic-cxr-2.0.0-negbio_all_10label.csv"
output_file = "/Users/post/Downloads/processed_data_checked_all.csv"


def process_mimic_cxr_data(input_path, output_path):
    """
    对MIMIC-CXR子集数据执行指定处理步骤。
    """
    try:
        # 1. 读取数据集
        df = pd.read_csv(input_path)
        print(f"原始数据集加载成功，形状: {df.shape}")

        # 2. 删除 'subject_id' 列
        if 'subject_id' in df.columns:
            df = df.drop(columns=['subject_id'])
            print("已删除 'subject_id' 列。")
        else:
            print("警告: 数据集中不存在 'subject_id' 列。跳过删除。")

        # 3. 确定需要处理的列 (除了 'study_id')
        label_columns = [col for col in df.columns if col != 'study_id']
        print(f"需要处理的标签列数量: {len(label_columns)}")

        # 4. 对标签列进行处理

        # 4a. 将 NAN 值填充为 0
        df[label_columns] = df[label_columns].fillna(0)
        print("NAN 值已填充为 0。")

        # 4b. 规范化: 保持 0.0 为 0, 1.0 为 1 (此步骤在 NAN 填充后，如果数据已经是 0 或 1，则无需额外转换)
        # 鉴于 MIMIC-CXR 的标签通常是 0, 1, 或 NaN，fillna(0) 已经确保所有值是 0 或 1。
        # 我们可以将这些列的数据类型统一为整数 (int)
        df[label_columns] = df[label_columns].astype(int)
        print("标签列数据类型已转换为整数 (0 或 1)。")

        # 5. 增加 'Normal' 列
        # 规则: 检查所有标签列是否全为 0。

        # 使用 .all(axis=1) 检查每一行在所有标签列中是否都为 0 (即 (df[label_columns] == 0).all(axis=1) )
        df['Normal'] = (df[label_columns] == 0).all(axis=1).astype(int)
        print("已增加 'Normal' 列。")

        # 检查 'Normal' 列的分布 (可选)
        print(f"新数据集形状: {df.shape}")
        print(f"'Normal' 列分布:\n{df['Normal'].value_counts()}")

        # 6. 存储最终数据集
        df.to_csv(output_path, index=False)
        print(f"最终数据集已成功保存到: {output_path}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_path}。请检查路径是否正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


# 执行函数
process_mimic_cxr_data(input_file, output_file)