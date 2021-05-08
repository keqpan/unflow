import itertools
from .base_model import BaseModel
from . import network
import torch
import torch.nn as nn
from utils import util
import os

class PreTrainAHoleModel(BaseModel, nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            pass
        return parser
    
    def __init__(self, opt):
        super(PreTrainAHoleModel, self).__init__(opt)
        if self.isTrain:
            self.loss_names = ['depth_dif_A', 'hole_dif_A']
        self.loss_names_test = ['depth_dif_A', 'hole_dif_A'] 
                
        self.visuals_names = ['real_img_A', 'real_depth_A',
                              'fake_depth_A', 'img_depth_A', 
                              'hole_mask_A']
            
        self.model_names = ['netG_A']
        
        self.netG_A = network.define_Gen(opt, input_type='img_feature_depth')
        
        ### Image2Depth 
        self.netG_F = network.define_Gen(opt, input_type='img', out_type = 'feature')
        self.netG_D = network.define_Gen(opt, input_type='feature', out_type = 'depth')

        self.criterionMaskedL1 = network.MaskedL1Loss()
        self.criterionL1 = nn.L1Loss()
        if self.isTrain:
            self.optimizer_G_A = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizers.extend([self.optimizer_G_A])
            self.opt_names = ['optimizer_G_A']
#         self.surf_normals = network.SurfaceNormals()
        self.hole_border = -0.95 ### Holes
    
    def set_input(self, input):
        self.name_A = input['A_name']
        self.real_img_A = input['A_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_A = input['A_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        
    def forward(self):
        ###Img2Depth
        self.img_feature_A = self.netG_F(self.real_img_A)
        self.img_depth_A = self.netG_D(self.img_feature_A)
        
        ###Masks
        self.hole_mask_A = self.get_mask(self.real_depth_A)
        
        self.fake_depth_A = self.netG_A(self.real_depth_A, self.img_feature_A)

    
    def backward_G(self):
        self.loss_G = 0.0
        self.loss_depth_dif_A = self.criterionMaskedL1(self.real_depth_A, self.fake_depth_A, ~self.hole_mask_A)
        self.loss_hole_dif_A = self.criterionMaskedL1(self.img_depth_A, self.fake_depth_A, self.hole_mask_A)
        self.loss_G = self.loss_G + self.loss_hole_dif_A + self.loss_depth_dif_A

        self.loss_G.backward()

    def optimize_param(self):
        self.forward()
        self.zero_grad([self.netG_A])
        self.backward_G()
        self.optimizer_G_A.step()
        
    def calc_test_loss(self):
        self.test_depth_dif_A = self.criterionMaskedL1(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.fake_depth_A, self.opt), ~self.hole_mask_A)
        self.test_hole_dif_A = self.criterionMaskedL1(util.data_to_meters(self.img_depth_A, self.opt), util.data_to_meters(self.fake_depth_A, self.opt), self.hole_mask_A)
       
    def update_loss_weight(self, global_iter):
        pass
