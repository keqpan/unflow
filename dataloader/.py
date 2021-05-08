from dataloader.base_dataset import BaseDataset
import random
import os
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
class SemiCycleDataset(BaseDataset):
    @staticmethod
    def mod_options(parser, is_train):
#         parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
#         parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  
        return parser
    
    def trasform(self, depth, img=None):
        depth = TF.to_tensor(depth)
        if img is not None:
            img = TF.to_tensor(img)
        return depth, img
    
    def __init__(self, opt):
        super().__init__(opt)
        self.dir_A = os.path.join(self.root, self.opt.phase + 'A')
        self.dir_B = os.path.join(self.root, self.opt.phase + 'B')
        self.dir_A_img = os.path.join(self.dir_A, 'img')
        self.dir_A_depth = os.path.join(self.dir_A, 'depth')
        self.dir_B_img = os.path.join(self.dir_B, 'img')
        self.dir_B_depth = os.path.join(self.dir_B, 'depth')
        
        self.A_imgs = sorted(self.get_paths(self.dir_A_img))
        self.A_depths = sorted(self.get_paths(self.dir_A_depth))
        self.B_imgs = sorted(self.get_paths(self.dir_B_img))
        self.B_depths = sorted(self.get_paths(self.dir_B_depth))
        assert (len(self.A_imgs) == len(self.A_depths)), 'not pair img depth'
        assert (len(self.B_imgs) == len(self.B_depths)), 'not pair img depth'
        
        self.A_size = len(self.A_imgs)
        self.B_size = len(self.B_depths)
                              

        
    def __getitem__(self, index):
        A_img_path = self.A_imgs[index]
        A_depth_path = self.A_depths[index]
        index_B = random.randint(0, self.B_size - 1)
        B_img_path = self.B_imgs[index_B]
        B_depth_path = self.B_depths[index_B]
        try:
            A_img = Image.open(A_img_path)
            A_depth = Image.open(A_depth_path)
            B_img = Image.open(B_img_path)
            B_depth = Image.open(B_depth_path)
        except:
            pass
        A_depth, A_img = self.trasform(A_depth, A_img)
        B_depth, B_img = self.trasform(B_depth, B_img)
        return {'A_depth': A_depth, 'A_img': A_img, 'B_depth': B_depth, 'B_img': B_img, 
                'A_depth_path': A_depth_path, 'A_img_path': A_img_path, 'B_depth_path': B_depth_path, 'B_img_path': B_img_path}
    def __len__(self):
        return max(self.A_size, self.B_size)

