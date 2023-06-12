import pandas as pd

# 创建示例 DataFrame A 和 B
A = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}, index=[0, 1, 2])
B = pd.DataFrame({'col1': [7, 8, 9], 'col2': [10, 11, 12]}, index=[0, 1, 2])

# 合并 A 和 B
C = pd.merge(A, B, left_index=True, right_index=True)

print(C)
