import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def test_transforms():
    to_tensor = transforms.ToTensor()
    sample_image = np.array([[0, 128, 255],
                            [64, 192, 32]], dtype=np.uint8)
    # ... if the numpy.ndarray has dtype = np.uint8. In the other cases, tensors are returned without scaling.

    tensor_image = to_tensor(sample_image)
    print(tensor_image)
    print(tensor_image.permute(1, 2, 0))
    plt.imshow(tensor_image.permute(1, 2, 0))
    plt.show()