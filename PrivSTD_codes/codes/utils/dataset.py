import numpy as np
import random
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import torch.nn as nn
from datetime import datetime, timezone




def read_dataset(config):
    dataset_path = './data/' + config['datasets']['name'] + '.npy'
    db = np.load(dataset_path, allow_pickle=True)

    dims = db.shape[1]

    db = db.reshape(-1, dims)
    # the bounds of the data
    min_vals = np.min(db, axis=0)
    max_vals = np.max(db, axis=0)
    n = db.shape[0]

    return db, min_vals, max_vals, n


def get_counts(config, db, min_vals, max_vals, grid, time_grid):
    _bins = []
    dims = db.shape[1]

    for i in range(dims):
        step = (max_vals[i] - min_vals[i]) / (grid if i != dims - 1 else time_grid)
        boundaries = np.arange(min_vals[i], max_vals[i] + step, step)
        _bins.append(boundaries)

    counts, _ = np.histogramdd(db, bins=_bins)
    counts = counts.astype(int)

    test_samples = db.copy()
    np.random.shuffle(test_samples)
    test_samples = test_samples[:config['test']['test_size']]

    # convert to (i,j, k)
    _bins = []
    for i in range(dims):
        step = (max_vals[i] - min_vals[i]) / (grid if i != dims - 1 else time_grid)
        boundaries = np.arange(min_vals[i], max_vals[i], step)
        _bins.append(boundaries)
    test_samples = tuple(
        np.searchsorted(_bins[i], test_samples[:, i], side='right') - 1 for i in range(dims)
    )
    test_samples = np.array(test_samples).T

    return counts, test_samples


def pad_data(x, window_size):
    # x: (H, W)
    h_old, w_old = x.shape
    # if h_old % window_size == 0 and w_old % window_size == 0:
    #     return x
    multiplier = max(h_old // window_size + 1, w_old // window_size + 1)
    h_pad = multiplier * window_size - h_old
    w_pad = multiplier * window_size - w_old
    x = torch.cat([x, torch.flip(x, [0])], dim=0)[:h_pad + h_old, :]
    x = torch.cat([x, torch.flip(x, [1])], dim=1)[:, :w_pad + w_old]

    return x


class MVDataset(Dataset):
    def __init__(self, data, img_size, is_train=True):
        super(MVDataset, self).__init__()
        self.data = data
        self.is_train = is_train
        self.img_size = img_size

    def get_patch(self, img):
        # img: (C, H, W)
        H, W = img.shape[-2:]
        if self.img_size >= H:
            return img
        # randomly select the top-left corner of the patch
        top_h = np.random.randint(0, H - self.img_size)
        top_w = np.random.randint(0, W - self.img_size)
        patch = img[:, top_h:top_h + self.img_size, top_w:top_w + self.img_size]
        return patch

    def get_all_pathes(self, img):
        # img: (C, H, W)
        H, W = img.shape[-2:]
        if self.img_size >= H:
            return img
        # return all patches
        patches = []
        for top_h in range(0, H - self.img_size, self.img_size):
            for top_w in range(0, W - self.img_size, self.img_size):
                patch = img[:, top_h:top_h + self.img_size, top_w:top_w + self.img_size]
                patches.append(torch.from_numpy(patch))
        return patches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # self.data: (N, C, H, W)
        img = self.data[idx]  # (C, H, W)

        if self.is_train:
            # patches = self.get_all_pathes(img)
            # # return all patches with [B, C, H, W]
            # return torch.stack(patches).float()
            patch = self.get_patch(img)
            patch = torch.from_numpy(patch).float()

            # C, H, W => B, C, H, W
            # if len(patch.shape) < 4:
            #     patch = patch.unsqueeze(0)
            return patch.float()
        else:
            img = torch.from_numpy(img).float()
            # if len(img.shape) < 4:
            #     img = img.unsqueeze(0)

            return img.float()

def pad_data_two(data, window_size):
    # data: torch.Tensor，可以是 [H,W] (转为 [1,1,H,W])、[C,H,W] (转为 [1,C,H,W]) 或 [B,C,H,W]
    if len(data.shape) == 2:  # [H,W] -> [1,1,H,W]
        data = data.unsqueeze(0).unsqueeze(0)
    elif len(data.shape) == 3:  # [C,H,W] -> [1,C,H,W]
        data = data.unsqueeze(0)
    # 现在 data 是 [B,C,H,W]
    B, C, H, W = data.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    padded = nn.functional.pad(data, (0, pad_w, 0, pad_h), mode='reflect')
    return padded  # 始终返回 [B,C,H_p,W_p]，不 squeeze

class MVDataset_two(Dataset):
    def __init__(self, data, img_size, is_train=True, apply_transform=True, seed=None):
        super(MVDataset_two, self).__init__()
        self.data = data
        self.is_train = is_train
        self.img_size = img_size
        self.apply_transform = apply_transform and is_train  # 只在训练时可选应用
        self.seed = seed  # 用于同步随机种子（确保 noisy 和 var 使用相同随机）

        # 定义数据增强变换（几何：flip/rotate，帮助粗网格泛化；强度小以防破坏结构）
        if self.apply_transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90, fill=0),  # 90度旋转，适合网格数据
            ])
        else:
            self.transform = None

    def get_patch(self, img):
        # img: (C, H, W)
        H, W = img.shape[-2:]
        if self.img_size >= H:
            return img
        # randomly select the top-left corner of the patch
        top_h = np.random.randint(0, H - self.img_size)
        top_w = np.random.randint(0, W - self.img_size)
        patch = img[:, top_h:top_h + self.img_size, top_w:top_w + self.img_size]
        return patch

    def get_all_pathes(self, img):
        # img: (C, H, W)
        H, W = img.shape[-2:]
        if self.img_size >= H:
            return img
        # return all patches
        patches = []
        for top_h in range(0, H - self.img_size, self.img_size):
            for top_w in range(0, W - self.img_size, self.img_size):
                patch = img[:, top_h:top_h + self.img_size, top_w:top_w + self.img_size]
                patches.append(torch.from_numpy(patch))
        return patches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 设置随机种子以同步 noisy 和 var 的随机（e.g., patch位置/transform）
        if self.seed is not None:
            np.random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)

        # self.data: (N, C, H, W)
        img = self.data[idx]  # (C, H, W)；当前 C=1

        if self.is_train:
            # patches = self.get_all_pathes(img)
            # # return all patches with [B, C, H, W]
            # return torch.stack(patches).float()
            patch = self.get_patch(img)
            patch = torch.from_numpy(patch).float()

            # 应用变换（如果启用）
            if self.transform:
                # 变换应用于 [C,H,W]，但 torchvision 期望 [C,H,W]
                patch = self.transform(patch)

            # C, H, W => B, C, H, W （注释掉，原代码有但不需，因为 DataLoader 会 batch）
            # if len(patch.shape) < 4:
            #     patch = patch.unsqueeze(0)
            return patch.float()
        else:
            img = torch.from_numpy(img).float()
            # if len(img.shape) < 4:
            #     img = img.unsqueeze(0)

            return img.float()