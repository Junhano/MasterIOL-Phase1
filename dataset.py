from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import torchvision.transforms as T

class IOLClassfierDataset(Dataset):

    def __init__(self, imagenames , labels, image_path = '../../../cropped_image/', mask_path = '../../../segmentation/cropped_mask/', transform = None):
        self.imagenames = imagenames
        self.labels = labels
        self.image_path = image_path
        self.transform = transform
        self.mask_path = mask_path
        self.state_cache_dict = dict()

    def __len__(self):
       return len(self.imagenames)


    def __getitem__(self, index):
        image_path = self.image_path + self.imagenames[index]
        img = read_image(image_path).float()

        if image_path in self.state_cache_dict:
            mean, std = self.state_cache_dict[image_path][0], self.state_cache_dict[image_path][1]
        else:
            mean, std = img.mean(), img.std()
            self.state_cache_dict[image_path] = [mean, std]

        img = T.Normalize(mean, std)(img)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[index]
