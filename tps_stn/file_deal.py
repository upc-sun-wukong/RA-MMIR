import torch
import numpy as np

pt = torch.load('E:\\daily_Log\\0618\\RA-MMIR\\base_files\\superpoint_v1.pth')
#L:\\CODE\\matching\\SuperGlue\\SuperGlue_training-main\\tps_stn\\mnist_data\\MNIST\\processed\\training.pt
pt_COCO = torch.load("E:\\daily_Log\\0717\\last_att_emau_coco_feature.pt")
print(pt)

# r1 = 0.9
# x = np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (4 - 1))
#
# print(x)