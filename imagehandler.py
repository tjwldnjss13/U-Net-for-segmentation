import torch
import torchvision.transforms as transforms

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a.resize_((2, 2))
print(a)