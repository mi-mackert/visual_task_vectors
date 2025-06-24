import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

class NYUDepthV2Dataset(Dataset):
    def __init__(self, root_dir="/content/drive/MyDrive/BachelorArbeit/Datasets_VAT/nyu-depth-v2", type: str = "train", task=None, image_transform=None, padding: bool = 1, mask_transform = None, query_support_list = None,):
        self.task = task
        self.root_dir = root_dir
        self.type = type
        self.image_transform = image_transform
        self.padding = padding
        self.mask_transform = mask_transform
        self.query_support_pairs = query_support_list

        self.image_dir = os.path.join(root_dir, '%s/images/data' % (type))
        self.label_dir = os.path.join(root_dir, '%s/masks/data' % (type))
        self.depth_dir = os.path.join(root_dir, '%s/depth/data' % (type))

        self.filenames = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.query_support_pairs is not None:
            # Use query and support names from the JSON file
            pair = self.query_support_pairs[idx % len(self.query_support_pairs)]
            img_name = pair['query_name']
            support_name = pair['support_name']
        else:
            img_name = self.filenames[idx]
            support_name = self.get_support(idx)

        img_path = os.path.join(self.image_dir, img_name)
        support_path = os.path.join(self.image_dir, support_name)
        image = Image.open(img_path)
        support_image = Image.open(support_path)

        if self.task == 1:
            label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.png'))
            label = Image.open(label_path)
            support_label_path = os.path.join(self.label_dir, support_name.replace('.jpg', '.png'))
            support_mask = Image.open(support_label_path)
            grid = self.segmentation_grid(support_image, support_mask, image, label)

        if self.task == 0:
            depth_path = os.path.join(self.depth_dir, img_name)
            depth = self.load_depth(depth_path)
            support_depth_path = os.path.join(self.depth_dir, support_name)
            support_depth = self.load_depth(support_depth_path)
            grid = self.depth_grid(image, support_image, depth, support_depth)

        if self.task == None:
            grid = []
            depth_path = os.path.join(self.depth_dir, img_name)
            depth = self.load_depth(depth_path)
            support_depth_path = os.path.join(self.depth_dir, support_name)
            support_depth = self.load_depth(support_depth_path)
            grid.append(self.depth_grid(image, support_image, depth, support_depth))
            
        batch = {'query_name': img_name, 'support_name': support_name, 'grid': grid}
        return batch
    
    def segmentation_grid(self, support_img, support_mask, query_img, query_mask):
        if self.image_transform:
            query_img = self.image_transform(query_img)
            support_img = self.image_transform(support_img)
        if self.mask_transform:
            query_mask = self.mask_transform[0](query_mask)
            support_mask = self.mask_transform[0](support_mask)
        
        grid = self.create_grid_from_images_segmentation(support_img, support_mask, query_img, query_mask, flip=False)
        return grid
    

    def get_support(self, idx):
        query_name = self.filenames[idx]
        while True:
            support_id = np.random.choice(self.filenames)
            if query_name != support_id:
                break
        return support_id

    def create_grid_from_images_segmentation(self, support_img, support_mask, query_img, query_mask, flip: bool = False):
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding, 2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        if flip:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas
    

    def load_depth(self, path):
        depth = Image.open(path).convert('I')
        depth_np = np.array(depth).astype(np.float32)
        return torch.from_numpy(depth_np).unsqueeze(0)
    

    """def depth_grid(self, query_img, support_img, query_mask, support_mask):
        if self.mask_transform:
          query_mask = self.mask_transform[0](query_mask)
          support_mask = self.mask_transform[0](support_mask)
        if self.image_transform:
          query_img = self.image_transform(query_img)
          support_img = self.image_transform(support_img)
        grid = self.create_grid_from_depth_estimation(support_img, support_mask, query_img, query_mask)
        
        return grid"""

    ''' def create_grid_from_depth_estimation(self, support_img, support_mask, query_img, query_mask, flip: bool = False):
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding, 2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        if flip:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas '''

    def depth_grid(self, query_img, support_img, query_depth, support_depth):
        # Apply transforms to RGB images if defined
        if self.image_transform:
            query_img = self.image_transform(query_img)
            support_img = self.image_transform(support_img)
    
        # Convert depth to 3-channel normalized display for VQGAN input
        query_depth = self.normalize_depth_for_display(query_depth)
        support_depth = self.normalize_depth_for_display(support_depth)

        grid = self.create_grid_from_depth_estimation(support_img, support_depth, query_img, query_depth)
        return grid

    def create_grid_from_depth_estimation(self, support_img, support_depth, query_img, query_depth, flip: bool = False):
        if self.mask_transform:
          query_depth = self.resize_tensor(query_depth)
          support_depth = self.resize_tensor(support_depth)
        C = support_img.shape[0]  # should be 3
        H = 2 * support_img.shape[1] + 2 * self.padding
        W = 2 * support_img.shape[2] + 2 * self.padding
        canvas = torch.ones((C, H, W))

        # Top-left: support image
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img

        if flip:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_depth
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_depth
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_depth
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_depth

        return canvas

    def normalize_depth_for_display(self, depth_image):
        if isinstance(depth_image, Image.Image):
            depth_tensor = transforms.ToTensor()(depth_image)
        else:
            depth_tensor = depth_image

        d = depth_tensor.clone()
        d -= d.min()
        d /= (d.max() + 1e-8)
        return d.repeat(3, 1, 1)  # [3, H, W]

    def resize_tensor(self, tensor, size=(111, 111)):
      """Resize a [C, H, W] tensor to (size, size) using bilinear interpolation."""
      return F.interpolate(tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)