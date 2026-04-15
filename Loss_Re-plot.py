import pandas as pd
from Tools import plot_losses


data_path = 'training/DefensiveDataProcessed/trajectory_sce3_cond.npy'  # 轨迹数据集路径，需为numpy数组 (num_samples, seq_len, dim)
model_name = data_path.split('/')[-1]
model_name = model_name.split('.')[0]
model_name = model_name.replace("trajectory_", "", 1)
epochs = 3000
latent_dim = 8  # 潜在空间维度（VAE编码器输出的潜在向量维度）
loss_save_path = 'training/loss/vae_offset_' + model_name + '_ld' + str(latent_dim) + '_epoch' + str(epochs) + '_loss1.png'

loss_history = {'total_loss': [], 'recon_loss': [], 'kld_loss': [], 'start_loss': [], 'time_loss': []}
csv_file_path = r'training\loss\vae_offset_sce3_cond_ld8_epoch3000_loss1.csv'  # CSV文件路径
try:
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 将各列数据转换为列表存入字典
    for key in loss_history.keys():
        loss_history[key] = df[key].tolist()

    print("数据读取成功！")
    # 可选：打印前5条数据验证
    print("recon_loss前5条数据:", loss_history['recon_loss'][:5])
except FileNotFoundError:
    print(f"错误：未找到文件 {csv_file_path}")
except KeyError as e:
    print(f"错误：CSV文件中缺少列 {e}，请检查表头是否正确")
except Exception as e:
    print(f"未知错误：{e}")

# 绘制损失曲线
plot_losses(loss_history, epochs, loss_save_path)
