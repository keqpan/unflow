import itertools
from .base_model import BaseModel
from . import network
import torch
import torch.nn as nn
from utils import util
import os

class PreTrainModel(BaseModel, nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            pass
        return parser
    
    def __init__(self, opt):
        super(PreTrainModel, self).__init__(opt)
        if self.isTrain:
            self.loss_names = ['depth_dif_A','norm_dif_A', 'depth_dif_B', 'norm_dif_B', 'hole_dif_A']
        self.loss_names_test = ['depth_dif_A', 'depth_dif_B'] 
                
        self.visuals_names = ['real_img_A', 'real_depth_A', 'real_norm_A',
                              'real_img_B', 'real_depth_B', 'real_norm_B',
                              'fake_depth_A', 'fake_norm_A', 'img_depth_A', 
                              'fake_depth_B', 'fake_norm_B', 'hole_mask_A']
            
        self.model_names = ['netG_A', 'netG_B']
        if opt.use_pretrain_img2depth:
            self.netG_A = network.define_Gen(opt, input_type='img_feature_depth')
        else:
            self.netG_A = network.define_Gen(opt, input_type='img_depth')
        self.netG_B = network.define_Gen(opt, input_type=opt.inp_B)
        
        ### Image2Depth 
        self.extra_model = ['netG_F', 'netG_D']
        self.netG_F = network.define_Gen(opt, input_type='img', out_type = 'feature')
        self.netG_D = network.define_Gen(opt, input_type='feature', out_type = 'depth')

        self.criterionMaskedL1 = network.MaskedL1Loss()
        self.criterionL1 = nn.L1Loss()
        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr_G, betas=(opt.beta1, 0.999), weight_decay=opt.w_decay_G)
            self.optimizers.extend([self.optimizer_G])
            self.criterionCosSim = network.CosSimLoss()
            self.criterionMaskedCosSim = network.MaskedCosSimLoss()
            self.opt_names = ['optimizer_G']
            self.surf_normals = network.SurfaceNormals()
        self.hole_border = opt.hole_border 
    
    def set_input(self, input):
        self.name_A = input['A_name']
        self.name_B = input['B_name']
        self.real_img_A = input['A_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_A = input['A_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        
        self.real_img_B = input['B_img']
        self.real_depth_B = input['B_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        
    def forward(self):
        ###Img2Depth
        with torch.no_grad():
            self.img_feature_A = self.netG_F(self.real_img_A)
            self.img_depth_A = self.netG_D(self.img_feature_A)
        
        ###Masks
        self.hole_mask_A = self.get_mask(self.real_depth_A)
        
        #Depths
        self.fake_depth_A = self.netG_A(self.real_depth_A, self.img_feature_A)
        self.fake_depth_B = self.netG_B(self.real_depth_B)
        
        ###Norm
        if self.isTrain:
            self.real_norm_A = self.surf_normals(self.real_depth_A)
            self.real_norm_B = self.surf_normals(self.real_depth_B)
            self.fake_norm_A = self.surf_normals(self.fake_depth_A)
            self.fake_norm_B = self.surf_normals(self.fake_depth_B)
    
    def backward_G(self):
        loss_A = 0.0
        loss_B = 0.0
        self.loss_depth_dif_A = self.criterionMaskedL1(self.real_depth_A, self.fake_depth_A, ~self.hole_mask_A) 
        self.loss_hole_dif_A = self.criterionMaskedL1(self.img_depth_A, self.fake_depth_A, self.hole_mask_A) * self.opt.l_hole_A
        self.loss_norm_dif_A = self.criterionMaskedCosSim(self.real_norm_A, self.fake_norm_A, ~self.hole_mask_A.repeat(1,3,1,1)) * self.opt.l_normal
        loss_A = loss_A + self.loss_depth_dif_A + self.loss_norm_dif_A + self.loss_hole_dif_A
        
        self.loss_depth_dif_B = self.criterionL1(self.real_depth_B, self.fake_depth_B)
        self.loss_norm_dif_B = self.criterionL1(self.real_norm_B, self.fake_norm_B) * self.opt.l_normal
        loss_B = loss_B + self.loss_depth_dif_B + self.loss_norm_dif_B
        self.loss_G = loss_A + loss_B
        self.loss_G.backward()

    def optimize_param(self):
        self.forward()
        self.zero_grad([self.netG_A, self.netG_B])
        self.backward_G()
        self.optimizer_G.step()
        
    def calc_test_loss(self):
        with torch.no_grad():
            self.test_depth_dif_A = self.criterionMaskedL1(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.fake_depth_A, self.opt), ~self.hole_mask_A)
            self.test_depth_dif_B = self.criterionL1(util.data_to_meters(self.real_depth_B, self.opt), util.data_to_meters(self.fake_depth_B, self.opt))
                
    def update_loss_weight(self, global_iter):
        pass
