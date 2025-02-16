import random

# 读取测试集文件
with open('dataset/test_original.txt', 'r') as file:
    lines = file.readlines()

# 检查每一行是否有换行符，没有则手动添加
lines = [line if line.endswith('\n') else line + '\n' for line in lines]

# 将每两行组合成一个条目
data = [lines[i:i + 2] for i in range(0, len(lines), 2)]

# 确定子集的数量和每个子集的大小
num_subsets = 5
subset_ratio = 0.8
subset_size = int(subset_ratio * len(data))

# 如果 subset_size 不是偶数，调整为偶数
if subset_size % 2 != 0:
    subset_size -= 1

# 存储子集
subsets = []

for i in range(num_subsets):
    # 随机选择不重复的条目索引，确保每次选择的条目数量为偶数
    random_indices = random.sample(range(len(data)), subset_size)
    subset = [data[index] for index in random_indices]

    # 计算子集的行数
    total_lines = sum(len(entry) for entry in subset)

    # 打印每个子集的行数信息，便于排查问题
    print(f"子集 {i + 1} 的条目数: {len(subset)}, 总行数: {total_lines}")

    subsets.append(subset)

# 保存子集到文件
for i, subset in enumerate(subsets):
    with open(f'G:/论文代码/ETFC-main/对比实验/测试用的训练集/a_{i + 1}.txt', 'w') as f:
        for entry in subset:
            f.writelines(entry)  # 写入条目中的每一行
    print(f"子集 {i + 1} 已保存到相应的 .txt 文件中。")
