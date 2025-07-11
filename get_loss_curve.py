import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('/opt/data/private/czw/207/BDINR-master/outputs/code_dev/BDNeRV_RC/train/2025-04-16_03-25-35/epoch-results.csv', header=[0, 1])
# 检查列名
print("DataFrame:", df.columns)
# 提取训练和验证损失
train_loss = df[('loss', 'train')]
valid_loss = df[('loss', 'valid')]

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Loss', linewidth=2)
plt.plot(valid_loss, label='Validation Loss', linewidth=2)

# 添加标题和标签
plt.title('Training and Validation Loss Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# 保存图像为PNG文件
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')

# 关闭图像（避免显示）
plt.close()