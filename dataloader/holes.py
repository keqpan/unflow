from dataloader.base_dataset import BaseDataset
import numpy as np
import torch
import os
import imageio
import albumentations as A
class HolesDataset(BaseDataset):
    @staticmethod 
    def modify_commandline_options(parser, is_train):
        if is_train:
            pass
        return parser
    def __init__(self, opt):
        super().__init__(opt)
        self.img_dir = os.path.join(self.root,self.opt.phase + 'A', 'img')
        self.depth_dir = os.path.join(self.root, self.opt.phase + 'A', 'depth')
        self.imgs = self.get_paths(self.img_dir)
        self.depths = self.get_paths(self.depth_dir)
        assert len(self.imgs) == len(self.depths), '#imgs should be = #detphs'
        self.size = len(self.imgs)
        
    def __getitem__(self, index):
        img_path = self.imgs[index]
        depth_path = self.depths[index]
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
        except:
            pass
        mask = depth == 0
        img, mask = self.transform(img, mask)
        return {'Image': img, 'GT_H': mask}
    
    def __len__(self):
        return self.size
    
    def apply_transformer(self, transformation, img, mask):
        target = {
            'image':'image',
            'mask':'mask'
        }
        res = A.Compose(transformation, p=1, additional_targets=target)(image=img, mask=mask)
        return res
    
    def transform(self, img, mask):
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        mask = mask.astype(np.float32)
        transform_list = []
        transform_list.append(A.Resize(height=self.opt.load_size_h, width=self.opt.load_size_w, interpolation=4, p=1))
        if self.opt.isTrain:
            transform_list.append(A.RandomCrop(height=self.opt.crop_size, width=self.opt.crop_size, p=1))
            transform_list.append(A.HorizontalFlip(p=0.5))
            transform_list.append(A.VerticalFlip(p=0.5))
        transformed = self.apply_transformer(transform_list, img, mask)
        img = np.clip(transformed['image'], -1, 1)
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = transformed['mask']
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

