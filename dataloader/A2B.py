from dataloader.base_dataset import BaseDataset
import numpy as np
import torch
import os
import imageio
import albumentations as A
import cv2
import random

class A2BDataset(BaseDataset):
    
    @staticmethod 
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--max_distance', type=float, default=8000.0, help='all depth bigger will seted to this value')
        return parser
    
    def __init__(self, opt):
        super().__init__(opt)
        self.img_dir = os.path.join(self.root, self.opt.phase + 'B', 'img')
        self.depth_dir = os.path.join(self.root, self.opt.phase + 'B', 'depth')
        self.imgs = self.get_paths(self.img_dir)
        self.depths = self.get_paths(self.depth_dir)
        assert len(self.imgs) == len(self.depths), '#imgs should be = #detphs'
        self.size = len(self.imgs)
        self.scale = self.opt.max_distance / 2
        
    def __getitem__(self, index):
        img_path = self.imgs[index]
        depth_path = self.depths[index]
        index_adv = random.randint(0, self.size - 1)
        adv_depth_path = self.depths[index_adv]
        img_name_ = os.path.basename(img_path)
        depth_name_ = os.path.basename(depth_path)
        img_name = os.path.splitext(img_name_)[0]
        depth_name = os.path.splitext(depth_name_)[0]
        assert img_name == depth_name, 'not pair img depth'
        assert self.is_image_file(img_name_), 'not implemented file extention'
        assert self.is_image_file(depth_name_), 'not implemented file extention'
        try:
            img = imageio.imread(img_path)
            depth = imageio.imread(depth_path)
            adv_depth = imageio.imread(adv_depth_path)
        except:
            pass
        img, depth, bad_depth, adv_depth = self.transform(img, depth, adv_depth)
        return {'Image': img, 'Depth': depth, 'Bad_depth': bad_depth, 'Adv_depth': adv_depth}
    
    def __len__(self):
        return self.size
    
    def apply_transformer(self, transformation, img, depth, bad_depth, adv_depth):
        target = {
            'image':'image',
            'depth':'image',
            'bad_depth':'image', 
            'adv_depth' :'image'
        }
        res = A.Compose(transformation, p=1, additional_targets=target)(image=img, depth=depth, bad_depth=bad_depth, adv_depth=adv_depth)
        return res
    
    def transform(self, img, depth, adv_depth):
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        
        size = depth.shape
        new_size = (int(size[1] / 4), int(size[0] / 4))
        res = cv2.resize(depth, dsize=new_size, interpolation=cv2.INTER_NEAREST)
        noise = np.random.uniform(low=0, high=50, size=res.shape)
        res = res + noise
        bad_depth = cv2.resize(res, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        
        depth = depth.astype(np.float32)
        depth = np.where(depth>self.opt.max_distance, self.opt.max_distance, depth)
        depth = (depth - self.scale) / self.scale
        
        adv_depth = adv_depth.astype(np.float32)
        adv_depth = np.where(adv_depth>self.opt.max_distance, self.opt.max_distance, adv_depth)
        adv_depth = (adv_depth - self.scale) / self.scale
        
        bad_depth = bad_depth.astype(np.float32)
        bad_depth = np.where(bad_depth>self.opt.max_distance, self.opt.max_distance,bad_depth)
        bad_depth = (bad_depth - self.scale) / self.scale
        
        transform_list = []
        transform_list.append(A.Resize(height=self.opt.load_size_h, width=self.opt.load_size_w, interpolation=4, p=1))
        if self.opt.isTrain:
            transform_list.append(A.RandomCrop(height=self.opt.crop_size, width=self.opt.crop_size, p=1))
            transform_list.append(A.HorizontalFlip(p=0.5))
            transform_list.append(A.VerticalFlip(p=0.5))
        transformed = self.apply_transformer(transform_list, img, depth, bad_depth, adv_depth)
        img = np.clip(transformed['image'], -1, 1)
        img = torch.from_numpy(img).permute(2, 0, 1)
        depth = np.clip(transformed['depth'], -1, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)
        bad_depth = np.clip(transformed['bad_depth'], -1, 1)
        bad_depth = torch.from_numpy(bad_depth).unsqueeze(0)
        adv_depth = np.clip(transformed['adv_depth'], -1, 1)
        adv_depth = torch.from_numpy(adv_depth).unsqueeze(0)
        return img, depth, bad_depth, adv_depth


# In[ ]:




