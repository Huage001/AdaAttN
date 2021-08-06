import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch.backends.cudnn as cudnn
from PIL import ImageFile

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


class UnalignedDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = opt.content_path
        self.dir_B = opt.style_path
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)

    def __getitem__(self, index):
        if self.opt.isTrain:
            index_A = index
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_A = index // self.B_size
            index_B = index % self.B_size
        A_path = self.A_paths[index_A]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform_A(A_img)
        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path).convert('RGB')
        B = self.transform_B(B_img)

        name_A = os.path.basename(A_path)
        name_B = os.path.basename(B_path)
        name = name_B[:name_B.rfind('.')] + '_' + name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]

        result = {'c': A, 's': B, 'name': name}

        return result

    def __len__(self):
        if self.opt.isTrain:
            return self.A_size
        else:
            return min(self.A_size * self.B_size, self.opt.num_test)