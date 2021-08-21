import glob
import random
import os
import os.path
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):

    def __init__(self, root, size, mode='train'):
        self.size = size
        self.transform = transforms.Compose(None)
        self.mode = mode
        # two list,each contains the path of all pictures in the selected folder
        self.files_A = glob.glob(os.path.join(root, '%sA' % mode) + '/*.*')
        self.files_B = glob.glob(os.path.join(root, '%sB' % mode) + '/*.*')
        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)
        if self.mode == 'train':
            # 256 to 286
            self.transform = transforms.Compose(
                [transforms.Resize(int(self.size * 1.12), Image.BICUBIC),
                 transforms.RandomCrop(self.size),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        elif self.mode == 'test':
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        elif self.mode == 'in':
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256,256), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )

    def __getitem__(self, index):
        if self.mode == 'train':
            item_A = self.transform(Image.open(self.files_A[index % self.len_A]))
            item_B = self.transform(Image.open(self.files_B[random.randint(0, self.len_B - 1)]))
            return {'A': item_A, 'B': item_B}

        elif self.mode in ['test', 'in']:
            if self.len_A != 0:
                filename_A = self.files_A[index % self.len_A]
                item_A = self.transform(Image.open(filename_A))
                filename_A = os.path.basename(filename_A)
                items_A = (item_A, filename_A)
            else:
                items_A = ([], ".")

            if self.len_B != 0:
                filename_B = self.files_B[index % self.len_B]
                item_B = self.transform(Image.open(filename_B))
                filename_B = os.path.basename(filename_B)
                items_B = (item_B, filename_B)
            else:
                items_B = ([], ".")

            return {'A': items_A, 'B': items_B}

    def __len__(self):
        return max(self.len_A, self.len_B)

    def A_isNotEmpty(self):
        return self.len_A != 0

    def B_isNotEmpty(self):
        return self.len_B != 0
