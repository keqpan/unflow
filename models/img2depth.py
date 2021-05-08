import itertools
from .base_model import BaseModel
from . import network
import torch
import torch.nn as nn
from utils import util
import os

class Img2DepthModel(BaseModel, nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            pass
        return parser
    
    def __init__(self, opt):
        super(Img2DepthModel, self).__init__(opt)
        if self.isTrain:
            self.loss_names = ['depth_dif_A',
                               'depth_dif_B', 'norm_dif_B',
                              ]
        self.loss_names_test = ['depth_dif_A', 'depth_dif_B'] 
                
        self.visuals_names = ['real_img_A', 'real_depth_A',
                              'fake_depth_A',
                              'real_img_B', 'real_depth_B', 
                              'fake_depth_B', 
                             ]
        
        self.model_names = ['netG_F', 'netG_F_S', 'netG_D']
        
        self.netG_F = network.define_Gen(opt, input_type='img', out_type = 'feature')
        self.netG_F_S = network.define_Gen(opt, input_type='img', out_type = 'feature')
        self.netG_D = network.define_Gen(opt, input_type='feature', out_type = 'depth')
        
        self.criterionMaskedL1 = network.MaskedL1Loss()
        self.criterionL1 = nn.L1Loss()
        self.criterionCosSim = network.CosSimLoss()
        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_F.parameters(), self.netG_F_S.parameters(), self.netG_D.parameters()), lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.opt_names = ['optimizer_G']
        self.surf_normals = network.SurfaceNormals()
        self.hole_border = -0.95 ### Holes
    
    def set_input(self, input):
        self.name_A = input['A_name']
        self.real_img_A = input['A_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_A = input['A_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        
        self.name_B = input['B_name']
        self.real_img_B = input['B_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_B = input['B_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        
        
    def forward(self):  
        ###Masks
        self.hole_mask_A = self.get_mask(self.real_depth_A)
        
        self.fake_depth_A = self.netG_D(self.netG_F(self.real_img_A))
        self.fake_depth_B = self.netG_D(self.netG_F_S(self.real_img_B))
        
        self.real_norm_B = self.surf_normals(self.real_depth_B)
        self.fake_norm_B = self.surf_normals(self.fake_depth_B)
        
    def backward_G(self):
        self.loss_depth_dif_A = self.criterionMaskedL1(self.real_depth_A, self.fake_depth_A, ~self.hole_mask_A)
        
        self.loss_depth_dif_B = self.criterionL1(self.real_depth_B, self.fake_depth_B)
        self.loss_norm_dif_B =  self.criterionCosSim(self.real_norm_B, self.fake_norm_B) * 50
        self.loss_G = self.loss_depth_dif_A + self.loss_depth_dif_B + self.loss_norm_dif_B
        self.loss_G.backward()

    def optimize_param(self):
        self.forward()
        self.zero_grad([self.netG_F, self.netG_F_S, self.netG_D])
        self.backward_G()
        self.optimizer_G.step()
    def update_loss_weight(self, global_iter):
        pass
    def calc_test_loss(self):
        self.test_depth_dif_A = self.criterionMaskedL1(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.fake_depth_A, self.opt), ~self.hole_mask_A)
        self.test_depth_dif_B = self.criterionL1(util.data_to_meters(self.real_depth_B, self.opt), util.data_to_meters(self.fake_depth_B, self.opt))
                






# from util.image_pool import ImagePool
# import numpy as np
# import itertools
# from .base_model import BaseModel
# from . import network
# import torch
# import torch.nn as nn
# class Img2DepthModel(BaseModel, nn.Module):
#     @staticmethod
#     def modify_commandline_options(parser, is_train=True):
#         if is_train:
#             pass
#         return parser

#     def __init__(self, opt):
#         super(Img2DepthModel, self).__init__(opt)
#         if self.isTrain:
#             self.loss_names = ['depth_dif_A','depth_dif_B']
                
#         self.visuals_names = ['real_img_A', 'real_depth_A', 'real_norm_A', 'fake_depth_A', 'fake_norm_A',
#                               'real_img_B', 'real_depth_B', 'real_norm_B', 'fake_depth_B', 'fake_norm_B']

#         self.model_names = ['netImage_f', 'netImage_f_syn', 'netTask']
        
#         self.netImage_f = network.define_G(3, 16, 64, 'resnet_6blocks', 'instance',
#                                         False, 'normal', 0.02, opt.gpu_ids, False, n_down = 2)
        
#         self.netImage_f_syn = network.define_G(3, 16, 64, 'resnet_6blocks', 'instance',
#                                          False, 'normal', 0.02, opt.gpu_ids, False, n_down = 2)
#         task_input_features = 16
#         self.netTask = network.define_G(task_input_features, 1, 64, 'resnet_9blocks', 'instance',
#                                          False, 'normal', 0.02, opt.gpu_ids, False, n_down = 2)
        
#         self.depth_dif_A = 0
#         self.depth_dif_B = 0
#         if self.isTrain:
#             self.criterion_task = torch.nn.L1Loss()
#             self.optimizer_G = torch.optim.Adam(itertools.chain(self.netImage_f_syn.parameters(), self.netImage_f.parameters(), self.netTask.parameters()), lr=opt.lr_G)
#             self.optimizers.append(self.optimizer_G)
#             self.opt_names = ['optimizer_G']
#         self.surf_normals = network.SurfaceNormals()
#         self.border = -0.95 ### Holes


#     def set_input(self, input):
#         self.name_A = input['A_name']
#         self.real_img_A = input['A_img'].to(self.device, non_blocking=torch.cuda.is_available())
#         self.real_depth_A = input['A_depth'].to(self.device, non_blocking=torch.cuda.is_available())
#         self.real_norm_A = self.surf_normals(self.real_depth_A)
        
#         self.name_B = input['B_name']
#         self.real_img_B = input['B_img'].to(self.device, non_blocking=torch.cuda.is_available())
#         self.real_depth_B = input['B_depth'].to(self.device, non_blocking=torch.cuda.is_available())
#         self.real_norm_B = self.surf_normals(self.real_depth_A)

#     def forward(self):
#         """Run forward pass; called by both functions <optimize_parameters> and <test>."""
#         self.fake_depth_B = self.netTask(self.netImage_f_syn(self.real_img_B))
#         self.fake_depth_A = self.netTask(self.netImage_f(self.real_img_A))
#         self.fake_norm_A = self.surf_normals(self.fake_depth_A)
#         self.fake_norm_B = self.surf_normals(self.fake_depth_B)
        
#     def backward_G(self, back=True):
#         """Calculate the loss for generators G_A and G_B"""

        
# #         if self.opt.norm_loss:
# #             calc_norm = SurfaceNormals()
# #             self.norm_syn = calc_norm(self.syn_depth)
# #             self.norm_syn_pred = calc_norm(self.pred_syn_depth)
# #             self.norm_real = calc_norm(self.real_depth)
# #             self.norm_real_pred = calc_norm(self.pred_real_depth)
# #             self.loss_syn_norms = self.criterion_task(self.norm_syn, self.norm_syn_pred) 
        
#         self.loss_depth_dif_B = self.criterion_task(self.real_depth_B, self.fake_depth_B) 

#         mask_real = torch.where(self.real_depth_A<-0.97, torch.tensor(0).float().to(self.real_depth_A.device), torch.tensor(1).float().to(self.real_depth_A.device))
#         self.loss_depth_dif_A = self.criterion_task(self.real_depth_A * mask_real, self.fake_depth_A * mask_real) 
        
#         # combined loss and calculate gradients
#         self.loss_G = self.loss_depth_dif_B*0.2  + self.loss_depth_dif_A
#         self.loss_G.backward()

#     def optimize_param(self):
#         """Calculate losses, gradients, and update network weights; called in every training iteration"""
#         # forward
#         self.forward()      # compute fake images and reconstruction images.
#         self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
#         self.backward_G()             # calculate gradients for G_A and G_B
#         self.optimizer_G.step()       # update G_A and G_B's weights

#     def update_loss_weight(self, global_iter):
#         pass