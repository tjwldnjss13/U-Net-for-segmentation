import numpy as np
import torch

a = torch.LongTensor([1, 2, 0, 4])
b = torch.LongTensor([0, 1, 1, 0])

a = (a > 0)
b = (b > 0)

print(a)
print(b)
print(len(a.__and__(b).nonzero()))