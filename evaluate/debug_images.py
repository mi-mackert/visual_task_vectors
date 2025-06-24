import matplotlib.pyplot as plt
from multitask_dataloader import DatasetNYU, DatasetPASCAL  # or wherever your DatasetNYU is
from nyu_dataloader import NYUDepthV2Dataset
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

padding = 1
image_transform = transforms.Compose(
    [transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
     transforms.ToTensor()])
mask_transform = [transforms.Compose(
    [transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
     transforms.ToTensor()]), transforms.Compose(
    [transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
     transforms.ToTensor()])]

# Instantiate the dataset
dataset = NYUDepthV2Dataset(type='test', image_transform=image_transform, mask_transform=mask_transform, task=0)


# Visualize a few samples
for i in range(5):
    batch = dataset[i]  # image: Tensor(C, H, W), mask: Tensor(H, W)
    grid_list = batch['grid']
    #grid_tensor = torch.cat(grid_list, dim=1)

    # Convert to PIL and show
    plt.imshow(TF.to_pil_image(grid_list))
    plt.title(f"Query: {batch['query_name']} | Support: {batch['support_name']}")
    plt.axis('off')
    plt.savefig(f"/content/drive/MyDrive/BachelorArbeit/vtv_output/nyu-depth-fixed/debug_images/debug_output_{i:04d}.png")


    # print(f"Sample {i}:")
    # print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
    # print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
    # print(f"  Unique mask classes: {torch.unique(mask)}")

    # Convert image to numpy for display
    # image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    # image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # normalize for display

    # plt.figure(figsize=(10, 4))

    # plt.subplot(1, 2, 1)
    # plt.imshow(image_np)
    # plt.title('Image')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(mask.cpu().numpy(), cmap='tab20')  # categorical colormap
    # plt.title('Segmentation Mask')
    # plt.axis('off')

    # plt.show()