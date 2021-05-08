import numpy as np
import matplotlib
import imageio
import matplotlib.pyplot as plt
import torch
from utils import util
import os
import logging

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.logger = logging.getLogger()
        
        
    def plot_validation(self, names, SAVE_PATH, domain='A', norm=False):
        if domain == 'A':
            scenes = ['scene0264_01_1600', 'scene0265_01_400', 'scene0262_00_1450', 'scene0088_01_500', 'scene0000_00_200']
        else:
            scenes = ['1066_13', '1137_7', '1203_3', '1273_6', '153_15']
        fig, axes = plt.subplots(nrows=len(scenes), ncols=len(names)+1, figsize=(200,200))
        fig.subplots_adjust(hspace=0.01, wspace=0.01)
        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
        for i, scene in enumerate(scenes):
            if norm:
                axes[i, 0].imshow(util.get_normal(imageio.imread(os.path.join(SAVE_PATH, names[0], 'val', domain, scene+'_depth.png'))),cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance)
            else:
                axes[i, 0].imshow(imageio.imread(os.path.join(SAVE_PATH, names[0], 'val', domain, scene+'_depth.png')),cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance)
            for j , name in enumerate(names):
                if norm:
                    axes[i,j+1].imshow(util.get_normal(imageio.imread(os.path.join(SAVE_PATH, name, 'val', domain, scene+'_depth_fake.png'))))
                else:
                    axes[i,j+1].imshow(imageio.imread(os.path.join(SAVE_PATH, name, 'val', domain, scene+'_depth_fake.png')),cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance)
    
    def plot_crop(self, img_dict, v_min, v_max, h_min, h_max):
        A_depth = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)
        B_depth_fake = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)
        n_col = 2
        n_row = 1
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(30, 30))
        for ax in axes.flatten():
            ax.axis('off')
        axes[0].set_title('Real Depth')
        axes[1].set_title('R-S Depth')
        axes[0].imshow(A_depth[0][v_min:v_max,h_min:h_max],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance/1000)
        axes[1].imshow(B_depth_fake[0][v_min:v_max,h_min:h_max],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance/1000)
        
    def plot_batch_(self, batch):
        A_imgs = util.tensor2im(batch['Image'], self.opt, isDepth=False)
        B_depth = util.tensor2im(batch['Depth'], self.opt, isDepth=True)
        A_depth = util.tensor2im(batch['Bad_depth'], self.opt, isDepth=True)
        Adv_depth = util.tensor2im(batch['Adv_depth'], self.opt, isDepth=True)

        n_col = 4
        n_row = 1
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(50, 50))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        for ax in axes.flatten():
            ax.axis('off')
        for i in range(1):
            axes[0].imshow(A_imgs[i])
            axes[1].imshow(A_depth[i], cmap=plt.get_cmap('RdYlBu'), vmin=-1, vmax = 1)
            axes[2].imshow(B_depth[i], cmap=plt.get_cmap('RdYlBu'), vmin=-1, vmax = 1)
            axes[3].imshow(Adv_depth[i], cmap=plt.get_cmap('RdYlBu'), vmin=-1, vmax = 1)
    
    def plot_batch(self, batch):
        A_imgs = util.tensor2im(batch['A_img'], self.opt, isDepth=False)
        A_depth = util.tensor2im(batch['A_depth'], self.opt, isDepth=True)
        B_depth = util.tensor2im(batch['B_depth'], self.opt, isDepth=True)
        B_imgs = util.tensor2im(batch['B_img'], self.opt, isDepth=False)
        if self.opt.use_semantic  and self.opt.isTrain:
            A_semantic = batch['A_semantic'].numpy()
            n_col = 3
        else:
            n_col = 2
        n_row = 2
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(50, 50))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        for ax in axes.flatten():
            ax.axis('off')
        for i in range(1):
            
            axes[2*i,0].imshow(A_imgs[i])
            axes[2*i,1].imshow(A_depth[i], cmap=plt.get_cmap('RdYlBu'), vmin=-1, vmax = 1)
            if self.opt.use_semantic  and self.opt.isTrain:
                axes[2*i,2].imshow(A_semantic[i], cmap='jet', vmin=0, vmax=40)
            
            axes[2*i+1,0].imshow(B_imgs[i])
            axes[2*i+1,1].imshow(B_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=-1, vmax = 1)
            
    def save_img_metric(self, img_dict, path, model_name, phase):
        util.mkdirs(os.path.join(path, model_name, phase, 'A2B', 'depth'))
        util.mkdirs(os.path.join(path, model_name, phase, 'A2B', 'normal'))
        util.mkdirs(os.path.join(path, model_name, phase, 'B2A', 'depth'))
        util.mkdirs(os.path.join(path, model_name, phase, 'B2A', 'normal'))
        util.mkdirs(os.path.join(path, model_name, phase, 'A2B2A', 'depth'))
        util.mkdirs(os.path.join(path, model_name, phase, 'A2B2A', 'normal'))
#         util.mkdirs(os.path.join(path, model_name, phase, 'B2A2B', 'depth'))
#         util.mkdirs(os.path.join(path, model_name, phase, 'B2A2B', 'normal'))
        B_depth_fake = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)*1000
        A_rec = util.tensor2im(img_dict['rec_depth_A'], self.opt, isDepth=True)*1000
        A_name = img_dict['name_A']
        
        A_depth_fake = util.tensor2im(img_dict['fake_depth_A'], self.opt, isDepth=True)*1000
        B_name = img_dict['name_B']
        for i in range(B_depth_fake.shape[0]):                       
            imageio.imwrite(os.path.join(path, model_name, phase, 'A2B', 'depth', A_name[i]+'.png'), B_depth_fake[i].astype(np.uint16))
            imageio.imwrite(os.path.join(path, model_name, phase, 'A2B2A', 'depth', A_name[i]+'.png'), A_rec[i].astype(np.uint16))
#             np.save(os.path.join(path, model_name, phase, 'A2B', 'normal', A_name[i]+'.npy'), util.get_normal_metric(B_depth_fake[i]))
            imageio.imwrite(os.path.join(path, model_name, phase, 'B2A', B_name[i]+'.png'), A_depth_fake[i].astype(np.uint16))
#             np.save(os.path.join(path, model_name, phase, 'B2A', 'normal', B_name[i]+'.npy'), util.get_normal_metric(A_depth_fake[i]))
            
    def save_img(self, img_dict, path, model_name, phase):
        util.mkdirs(os.path.join(path, model_name, phase, 'A'))
        util.mkdirs(os.path.join(path, model_name, phase, 'B'))
        A_imgs = util.tensor2im(img_dict['real_img_A'], self.opt, isDepth=False)
        A_depth = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)*1000
        B_depth_fake = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)*1000
        A_name = img_dict['name_A']
        
        B_imgs = util.tensor2im(img_dict['real_img_B'], self.opt, isDepth=False)
        B_depth = util.tensor2im(img_dict['real_depth_B'], self.opt, isDepth=True)*1000
        A_depth_fake = util.tensor2im(img_dict['fake_depth_A'], self.opt, isDepth=True)*1000
        B_name = img_dict['name_B']
        for i in range(A_imgs.shape[0]):
            imageio.imwrite(os.path.join(path, model_name, phase, 'A', A_name[i]+'_img.png'), A_imgs[i])    
            imageio.imwrite(os.path.join(path, model_name, phase, 'A', A_name[i]+'_depth.png'), A_depth[i].astype(np.uint16))                        
            imageio.imwrite(os.path.join(path, model_name, phase, 'A', A_name[i]+'_depth_fake.png'), B_depth_fake[i].astype(np.uint16))
            imageio.imwrite(os.path.join(path, model_name, phase, 'B', B_name[i]+'_img.png'), B_imgs[i])
            imageio.imwrite(os.path.join(path, model_name, phase, 'B', B_name[i]+'_depth.png'), B_depth[i].astype(np.uint16))
            imageio.imwrite(os.path.join(path, model_name, phase, 'B', B_name[i]+'_depth_fake.png'), A_depth_fake[i].astype(np.uint16))
    
    def plot_pretrain_A(self, img_dict):
        
        real_img_A = util.tensor2im(img_dict['real_img_A'], self.opt, isDepth=False)
        real_depth_A = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)
        hole_mask = util.torch2np(img_dict['hole_mask_A'])
        fake_depth_A = util.tensor2im(img_dict['fake_depth_A'], self.opt, isDepth=True)
        img2depth_A = util.tensor2im(img_dict['img_depth_A'], self.opt, isDepth=True)
        
        max_dist = self.opt.max_distance/1000
        batch_size = real_img_A.shape[0]
        n_pic = min(batch_size, self.opt.n_pic)
        n_row =  n_pic
        n_col = 5
        fig_size = (30,40)
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
        fig.subplots_adjust(hspace=0.0, wspace=0.1)
        old_level = self.logger.level
        self.logger.setLevel(100)
        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
        for i in range(n_pic):
            axes[i,0].set_title('A_rgb')
            axes[i,1].set_title('A_depth')
            axes[i,2].set_title('Holes')
            axes[i,3].set_title('fakeA_depth')
            axes[i,4].set_title('img2depth')
            
            axes[i,0].imshow(real_img_A[i])
            axes[i,1].imshow(real_depth_A[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[i,2].imshow(hole_mask[i])
            axes[i,3].imshow(fake_depth_A[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[i,4].imshow(img2depth_A[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            
        self.logger.setLevel(old_level)
        return fig
    def plot_pretrain(self, img_dict):
        
        real_img_A = util.tensor2im(img_dict['real_img_A'], self.opt, isDepth=False)
        real_depth_A = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)
        img2depth_A = util.tensor2im(img_dict['img_depth_A'], self.opt, isDepth=True)
        real_norm_A = util.get_normals(real_depth_A * 1000)
        fake_depth_A = util.tensor2im(img_dict['fake_depth_A'], self.opt, isDepth=True)
        fake_norm_A = util.get_normals(fake_depth_A * 1000)
        hole_mask = util.torch2np(img_dict['hole_mask_A'])
        
        real_img_B = util.tensor2im(img_dict['real_img_B'], self.opt, isDepth=False)
        real_depth_B = util.tensor2im(img_dict['real_depth_B'], self.opt, isDepth=True)
        real_norm_B = util.get_normals(real_depth_B * 1000)
        fake_depth_B = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)
        fake_norm_B = util.get_normals(fake_depth_B * 1000)
        max_dist = self.opt.max_distance/1000
        batch_size = real_img_A.shape[0]
        n_pic = min(batch_size, self.opt.n_pic)
        n_row = 2 * n_pic
        n_col = 6
        fig_size = (30,40)
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
        fig.subplots_adjust(hspace=0.0, wspace=0.1)
        old_level = self.logger.level
        self.logger.setLevel(100)
        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
        for i in range(n_pic):
            axes[2*i,0].set_title('A_rgb')
            axes[2*i,1].set_title('A_depth')
            axes[2*i,2].set_title('A_norm')
            axes[2*i,3].set_title('fakeA_depth')
            axes[2*i,4].set_title('img2depth')
            axes[2*i,5].set_title('fakeA_norm')

            axes[2*i+1,0].set_title('B_rgb')
            axes[2*i+1,1].set_title('B_depth')
            axes[2*i+1,2].set_title('B_norm')
            axes[2*i+1,3].set_title('fakeB_depth')
            axes[2*i+1,4].set_title('hole mask')
            axes[2*i+1,5].set_title('fakeB_norm')
            
            axes[2*i,0].imshow(real_img_A[i])
            axes[2*i,1].imshow(real_depth_A[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,2].imshow(real_norm_A[i])
            axes[2*i,3].imshow(fake_depth_A[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,4].imshow(img2depth_A[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,5].imshow(fake_norm_A[i])
            
            axes[2*i+1,0].imshow(real_img_B[i])
            axes[2*i+1,1].imshow(real_depth_B[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,2].imshow(real_norm_B[i])
            axes[2*i+1,3].imshow(fake_depth_B[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,4].imshow(hole_mask[i])
            axes[2*i+1,5].imshow(fake_norm_B[i])
        self.logger.setLevel(old_level)
        return fig
    
    def plot_holes(self, img_dict):
        real_img_A = util.tensor2im(img_dict['real_img_A'], self.opt, isDepth=False)
        real_depth_A = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)
        real_norm_A = util.get_normals(real_depth_A * 1000)
        fake_depth_A = util.tensor2im(img_dict['fake_depth_A'], self.opt, isDepth=True)
        fake_norm_A = util.get_normals(fake_depth_A * 1000)
        hole_mask = util.torch2np(img_dict['hole_mask'])
        batch_size = Img.shape[0]
        n_pic = min(batch_size, self.opt.n_pic)
        n_row = n_pic
        n_col = 6
        fig_size = (30,30)
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
        fig.subplots_adjust(hspace=0.0, wspace=0.1)
        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
        for i in range(n_pic):
            axes[i,0].set_title('Img')
            axes[i,1].set_title('Tru Hole')
            axes[i,2].set_title('Pred Hole')
            
            axes[i,0].imshow(Img[i])
            axes[i,1].imshow(True_Hole[i],cmap=plt.get_cmap('RdYlBu'))
            axes[i,2].imshow(Pred_Hole[i],cmap=plt.get_cmap('RdYlBu'))
        return fig
    
    def plot_img2depth(self, img_dict):
        real_img_A = util.tensor2im(img_dict['real_img_A'], self.opt, isDepth=False)
        real_depth_A = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)
        real_norm_A = util.get_normals(real_depth_A * 1000)
        fake_depth_A = util.tensor2im(img_dict['fake_depth_A'], self.opt, isDepth=True)
        fake_norm_A = util.get_normals(fake_depth_A * 1000)
        
        real_img_B = util.tensor2im(img_dict['real_img_B'], self.opt, isDepth=False)
        real_depth_B = util.tensor2im(img_dict['real_depth_B'], self.opt, isDepth=True)
        real_norm_B = util.get_normals(real_depth_B * 1000)
        fake_depth_B = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)
        fake_norm_B = util.get_normals(fake_depth_B * 1000)
#         hole_mask = util.torch2np(img_dict['hole_mask'])
        batch_size = real_img_A.shape[0]
        max_dist = self.opt.max_distance/1000
        n_pic = min(batch_size, self.opt.n_pic)
        n_row = 2 * n_pic
        n_col = 5
        fig_size = (30,30)
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
        fig.subplots_adjust(hspace=0.0, wspace=0.1)
        old_level = self.logger.level
        self.logger.setLevel(100)
        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
        for i in range(n_pic):
            axes[2 * i, 0].set_title('A_rgb')
            axes[2 * i, 1].set_title('A_depth')
            axes[2 * i, 2].set_title('A_norm')
            axes[2 * i, 3].set_title('fakeA_depth')
            axes[2 * i, 4].set_title('fakeA_norm')
            axes[2 * i + 1, 0].set_title('B_rgb')
            axes[2 * i + 1, 1].set_title('B_depth')
            axes[2 * i + 1, 2].set_title('B_norm')
            axes[2 * i + 1, 3].set_title('fakeB_depth')
            axes[2 * i + 1, 4].set_title('fakeB_norm')

            axes[2 * i, 0].imshow(real_img_A[i])
            axes[2 * i, 1].imshow(real_depth_A[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2 * i, 2].imshow(real_norm_A[i])
            axes[2 * i, 3].imshow(fake_depth_A[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2 * i, 4].imshow(fake_norm_A[i])
            
            axes[2 * i + 1, 0].imshow(real_img_B[i])
            axes[2 * i + 1, 1].imshow(real_depth_B[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2 * i + 1, 2].imshow(real_norm_B[i])
            axes[2 * i + 1, 3].imshow(fake_depth_B[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2 * i + 1, 4].imshow(fake_norm_B[i])
        self.logger.setLevel(old_level)
        return fig
    
    def plot_imgScannet(self, img_dict):
        A_imgs = util.tensor2im(img_dict['real_img_A'], self.opt, isDepth=False)
        A_depth = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)
        A_norm = util.get_normals(A_depth * 1000)
        B_depth_fake = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)
        B_norm_fake = util.get_normals(B_depth_fake * 1000)
        
        if self.opt.use_cycle_A:
            A_depth_rec = util.tensor2im(img_dict['rec_depth_A'], self.opt, isDepth=True)
            A_norm_rec = util.get_normals(A_depth_rec * 1000)
        else:
            A_depth_rec = np.zeros_like(B_depth_fake)
            A_norm_rec = np.zeros_like(B_norm_fake)
            
#         A_idt = util.tensor2im(img_dict['idt_A'], self.opt, isDepth=True)
        
        B_imgs = util.tensor2im(img_dict['real_img_B'], self.opt, isDepth=False)
        B_depth = util.tensor2im(img_dict['real_depth_B'], self.opt, isDepth=True)
        B_norm = util.get_normals(B_depth * 1000)
        A_depth_fake = util.tensor2im(img_dict['fake_depth_A'], self.opt, isDepth=True)
        A_norm_fake = util.get_normals(A_depth_fake * 1000)
        B_depth_rec = util.tensor2im(img_dict['rec_depth_B'], self.opt, isDepth=True)
        B_norm_rec = util.get_normals(B_depth_rec * 1000)
#         B_idt = util.tensor2im(img_dict['idt_B'], self.opt, isDepth=True)
        
        max_dist = self.opt.max_distance/1000
        batch_size = A_imgs.shape[0]
        n_pic = min(batch_size, self.opt.n_pic)
        n_col = 8
        fig_size = (40,30)
        n_row = 2 * n_pic
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
        fig.subplots_adjust(hspace=0.0, wspace=0.1)
        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
            if (i+1) % 8 == 0:
                ax.axis('on')
        self.logger.setLevel(100)
        old_level = self.logger.level
        for i in range(n_pic):
            axes[2*i,0].set_title('Real RGB')
            axes[2*i,1].set_title('Real Depth')
            axes[2*i,2].set_title('R-S Depth')
            axes[2*i,3].set_title('Cycle Depth A')
            axes[2*i,4].set_title('Real Norm')
            axes[2*i,5].set_title('R-S Norm')
            axes[2*i,6].set_title('Cycle Norm A')
            axes[2*i,7].set_title('Graph')
            
            axes[2*i+1,0].set_title('Syn RGB')
            axes[2*i+1,1].set_title('Syn Depth')
            axes[2*i+1,2].set_title('S-R Depth')
            axes[2*i+1,3].set_title('Cycle Depth B')
            axes[2*i+1,4].set_title('Syn Norm')
            axes[2*i+1,5].set_title('S-R Norm')
            axes[2*i+1,6].set_title('Cycle Norm B')
            axes[2*i+1,7].set_title('Graph')

            axes[2*i,0].imshow(A_imgs[i])
            axes[2*i,1].imshow(A_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,2].imshow(B_depth_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,3].imshow(A_depth_rec[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,4].imshow(A_norm[i])
            axes[2*i,5].imshow(B_norm_fake[i])
            axes[2*i,6].imshow(A_norm_rec[i])
            axes[2*i,7].plot(A_depth[i][100], label = 'Real Depth')
            axes[2*i,7].plot(B_depth_fake[i][100], label = 'R-S Depth')
            axes[2*i,7].legend()
            
            axes[2*i+1,0].imshow(B_imgs[i])
            axes[2*i+1,1].imshow(B_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,2].imshow(A_depth_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,3].imshow(B_depth_rec[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,4].imshow(B_norm[i])
            axes[2*i+1,5].imshow(A_norm_fake[i])
            axes[2*i+1,6].imshow(B_norm_rec[i])
            axes[2*i+1,7].plot(B_depth[i][100], label = 'Syn Depth')
            axes[2*i+1,7].plot(A_depth_fake[i][100], label = 'S-R Depth')
            axes[2*i+1,7].legend()
        self.logger.setLevel(old_level)                        
#             fig.colorbar(r_d, ax=axes[2*i,1],fraction=0.046, pad=0.03)
        return fig
            
    def plot_img(self, img_dict):
        A_imgs = util.tensor2im(img_dict['real_img_A'], self.opt, isDepth=False)
        A_depth = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)
        A_img_depht = util.tensor2im(img_dict['img_depth_A'], self.opt, isDepth=True)
        A_norm = util.get_normals(A_depth * 1000)
        B_depth_fake = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)
        B_norm_fake = util.get_normals(B_depth_fake * 1000)
#         A_depth_rec = util.tensor2im(img_dict['rec_depth_A'], self.opt, isDepth=True)
#         A_norm_rec = util.get_normals(A_depth_rec * 1000)
#         A_idt = util.tensor2im(img_dict['idt_A'], self.opt, isDepth=True)
        
        B_imgs = util.tensor2im(img_dict['real_img_B'], self.opt, isDepth=False)
        B_depth = util.tensor2im(img_dict['real_depth_B'], self.opt, isDepth=True)
        B_norm = util.get_normals(B_depth * 1000)
        A_depth_fake = util.tensor2im(img_dict['fake_depth_A'], self.opt, isDepth=True)
        A_norm_fake = util.get_normals(A_depth_fake * 1000)
        B_depth_rec = util.tensor2im(img_dict['rec_depth_B'], self.opt, isDepth=True)
        B_norm_rec = util.get_normals(B_depth_rec * 1000)
#         B_idt = util.tensor2im(img_dict['idt_B'], self.opt, isDepth=True)
#         if self.opt.use_semantic:
#                 A_semantic = img_dict['real_semantic_A'].data.cpu().numpy()
#                 A_semantic_pred = util.logits_to_label(img_dict['rec_semantic_A'])
#         if self.opt.use_mean_matching:
#             B_shift_fake = util.tensor2im(img_dict['fake_shift_B'], self.opt, isDepth=True)
#             B_shift_real = util.tensor2im(img_dict['real_shift_B'], self.opt, isDepth=True)
#             A_shift_fake = util.tensor2im(img_dict['fake_shift_A'], self.opt, isDepth=True)
#             A_shift_real = util.tensor2im(img_dict['real_shift_A'], self.opt, isDepth=True)
        
        max_dist = self.opt.max_distance/1000
        batch_size = A_imgs.shape[0]
        n_pic = min(batch_size, self.opt.n_pic)
        n_col = 9
        fig_size = (40,30)
        n_row = 2 * n_pic
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
        fig.subplots_adjust(hspace=0.0, wspace=0.1)
        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
            if (i+1) % 9 == 0:
                ax.axis('on')
        self.logger.setLevel(100)
        old_level = self.logger.level
        for i in range(n_pic):
            axes[2*i,0].set_title('Real RGB')
            axes[2*i,1].set_title('Real Depth')
            axes[2*i,2].set_title('R-S Depth')
            axes[2*i,3].set_title('Img2Depth')
            axes[2*i,4].set_title('Cycle Depth A')
            axes[2*i,5].set_title('Real Norm')
            axes[2*i,6].set_title('R-S Norm')
            axes[2*i,7].set_title('Cycle Norm A')
            axes[2*i,8].set_title('Graph')
            
            axes[2*i+1,0].set_title('Syn RGB')
            axes[2*i+1,1].set_title('Syn Depth')
            axes[2*i+1,2].set_title('S-R Depth')
            axes[2*i+1,3].set_title('Cycle Depth B')
            axes[2*i+1,4].set_title('Cycle Depth B')
            axes[2*i+1,5].set_title('Syn Norm')
            axes[2*i+1,6].set_title('S-R Norm')
            axes[2*i+1,7].set_title('Cycle Norm B')
            axes[2*i+1,8].set_title('Graph')

            axes[2*i,0].imshow(A_imgs[i])
            axes[2*i,1].imshow(A_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,2].imshow(B_depth_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,3].imshow(A_img_depht[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,4].imshow(A_img_depht[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,5].imshow(A_norm[i])
            axes[2*i,6].imshow(B_norm_fake[i])
            axes[2*i,7].imshow(B_norm_fake[i])
            axes[2*i,8].plot(A_depth[i][100], label = 'Real Depth')
            axes[2*i,8].plot(B_depth_fake[i][100], label = 'R-S Depth')
            axes[2*i,8].legend()
            
            axes[2*i+1,0].imshow(B_imgs[i])
            axes[2*i+1,1].imshow(B_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,2].imshow(A_depth_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,3].imshow(B_depth_rec[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,4].imshow(B_depth_rec[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,5].imshow(B_norm[i])
            axes[2*i+1,6].imshow(A_norm_fake[i])
            axes[2*i+1,7].imshow(B_norm_rec[i])
            axes[2*i+1,8].plot(B_depth[i][100], label = 'Syn Depth')
            axes[2*i+1,8].plot(A_depth_fake[i][100], label = 'S-R Depth')
            axes[2*i+1,8].legend()
        self.logger.setLevel(old_level)                        
#             fig.colorbar(r_d, ax=axes[2*i,1],fraction=0.046, pad=0.03)
        return fig






#         A_edge = img_dict['real_edge_A'].cpu()
        
#         A_logits_edges = img_dict['rec_edge_A']
#         prob = torch.sigmoid(A_logits_edges)
#         A_fake_edges = prob[:,0,:,:] > 0.5
#         A_fake_edges = A_fake_edges.cpu()*1
        































# from torch.utils.tensorboard import SummaryWriter
# import os
# from datetime import datetime
# import torchvision.transforms.functional as TF
# import matplotlib.pyplot as plt
# class Visualizer():
#     def __init__(self, opt):
#         self.opt = opt
#         if self.opt.isTrain:
#             self.phase = 'Train'
#         else:
#             self.phase = 'Test'
        
#         experiment_name = '{}@{}'.format(opt.name, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
#         self.writer = SummaryWriter(log_dir=os.path.join(opt.logdir, experiment_name))
    
#     def plot_loss(self, losses, global_step, is_epoch=False):
#         if is_epoch:
#             self.writer.add_scalars(self.phase+'epoch', losses, global_step)
#         self.writer.add_scalars(self.phase, losses, global_step)
        
#     def plot_img(self, img_dict, global_step, is_epoch=False):
#         A_imgs = img_dict['real_img_A'].cpu()
#         A_depth = img_dict['real_depth_A'].cpu()
#         B_depth_fake = img_dict['fake_depth_B'].cpu()
#         A_depth_rec = img_dict['rec_depth_A'].cpu()
#         B_idt = img_dict['idt_B'].cpu()
        
#         B_imgs = img_dict['real_img_B'].cpu()
#         B_depth = img_dict['real_depth_B'].cpu()
#         A_depth_fake = img_dict['fake_depth_A'].cpu()
#         A_idt = img_dict['idt_A'].cpu()
#         n_col = 5
#         n_row = 2*A_imgs.shape[0]
#         fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(1.5 * n_col, 1.7 * n_row))
#         for ax in axes.flatten():
#             ax.axis('off')
#         for i in range(A_imgs.shape[0]):
#             axes[2*i,0].set_title('Real RGB')
#             axes[2*i,1].set_title('Real Depth')
#             axes[2*i,2].set_title('R-S Depth')
#             axes[2*i,3].set_title('Cycle Depth')
#             axes[2*i,4].set_title('G_s-r(Real Depth)')
            
#             axes[2*i+1,0].set_title('Syn RGB')
#             axes[2*i+1,1].set_title('Syn Depth')
#             axes[2*i+1,2].set_title('S-R Depth')
#             axes[2*i+1,3].set_title('None')
#             axes[2*i+1,4].set_title('G_r-s(Syn Depth)')
            
#             axes[2*i,0].imshow(TF.to_pil_image(A_imgs[i]))
#             axes[2*i,1].imshow(TF.to_pil_image(A_depth[i]))
#             axes[2*i,2].imshow(TF.to_pil_image(B_depth_fake[i]))
#             axes[2*i,3].imshow(TF.to_pil_image(A_depth_rec[i]))
#             axes[2*i,4].imshow(TF.to_pil_image(B_idt[i]))
            
#             axes[2*i+1,0].imshow(TF.to_pil_image(B_imgs[i]))
#             axes[2*i+1,1].imshow(TF.to_pil_image(B_depth[i]))
#             axes[2*i+1,2].imshow(TF.to_pil_image(A_depth_fake[i]))
#             axes[2*i+1,3].imshow(TF.to_pil_image(A_idt[i]))
#             axes[2*i+1,4].imshow(TF.to_pil_image(A_idt[i]))
#         if is_epoch:
#             self.writer.add_figure(self.phase+'epoch', fig, global_step)
#         self.writer.add_figure(self.phase, fig, global_step)
#         plt.close(fig)

