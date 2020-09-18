import torch
import numpy as np

a = torch.ones((1, 1, 3, 3), dtype=torch.float32)
b = torch.ones((1, 1, 3, 3), dtype=torch.float32)
#
# zeros = torch.zeros(a.shape[0], a.shape[1], a.shape[2], 1)
# a = torch.cat([a, zeros], dim=3)
a = a.unsqueeze(0)
print(a.shape)