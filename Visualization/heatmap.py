import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# 假设输出是 [B, 1, H, W] 的张量
# The output of the last layer of the network is used as input
input = torch.randn(8, 1, 352, 352)
output = torch.sigmoid(input)  
save_dir = 'heatmaps'
os.makedirs(save_dir, exist_ok=True)

# 遍历 batch 内的每一张图
for i in range(output.shape[0]):
    # 取出第 i 张图并转为 numpy 格式
    heatmap = output[i, 0].detach().cpu().numpy()
    
    # 归一化到 0-255
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    # 使用 OpenCV 的 JET 颜色映射生成热力图
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 可视化
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    plt.imshow(heatmap_color)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"heatmap_{i+1}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"{output.shape[0]} heatmaps have been successfully saved to the folder: {save_dir}")

