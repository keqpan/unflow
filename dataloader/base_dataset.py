import os
import glob
import numpy as np
import imageio
import albumentations as A
import torch.utils.data as data
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
    
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.scale = self.opt.max_distance / 2
        self.IMG_EXTENSIONS = []
        self.transforms = [A.Resize(height=self.opt.load_size_h, width=self.opt.load_size_w, interpolation=4, p=4)]
        self.dir_A = os.path.join(self.root, self.opt.phase + 'A')
        self.dir_B = os.path.join(self.root, self.opt.phase + 'B')
    
    def add_extensions(self, ext_list):
        self.IMG_EXTENSIONS.extend(ext_list)    
    
    def is_image_files(self, files):
        for f in files:
            assert any(f.endswith(extension) for extension in self.IMG_EXTENSIONS), 'not implemented file extntion type {}'.format(f.split('.')[1])
        
    def get_paths(self, dir, reverse=False):
        files = []
        assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)
        files = sorted(glob.glob(os.path.join(dir, '**/*.*'), recursive=True), reverse=reverse)
        return files[:min(self.opt.max_dataset_size, len(files))]
    
    def get_name(self, file_path):
        img_n = os.path.basename(file_path).split('.')[0]
        return img_n
    
    def normalize_img(self, img):
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                if img.shape[2] > 3:
                    img = img[:,:,:3]
                img = img.astype(np.float32)
#                 img = (img-mean_i)/std_i
                img = img / 127.5 - 1.0
                return img
#             elif img.dtype == np.float32:
#                 if img.shape[2] > 3:
#                     img = img[:,:,:3]
#                 img = img / 127.5 - 1.0
#                 return img
            else:
                print(img.dtype)
                raise AssertionError('Img datatype')
        else:
            raise AssertionError('Img filetype')
    
    def normalize_depth(self, depth):
        if isinstance(depth, np.ndarray):
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32)
                depth = np.where(depth>self.opt.max_distance, self.opt.max_distance, depth)
                depth = depth / self.scale - 1
                return depth
            else:
                print(depth.dtype)
                raise AssertionError('Depth datatype')
        else:
            raise AssertionError('Depth filetype')
    
    def read_data(self, path):
        return imageio.imread(path)
    
    @abstractmethod
    def __len__(self):
        return 0
    @abstractmethod
    def __getitem__(self, index):
        pass




