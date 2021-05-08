import itertools
# from util.image_pool import ImagePool
from .base_model import BaseModel
from . import network
from util.util import GaussianSmoothing
from util import util
import torch
import torch.nn as nn

class A2BModel(BaseModel, nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--l_depth_A_begin', type=float, default=1.0, help='start of depth range loss')
            parser.add_argument('--l_depth_A_end', type=float, default=1.0, help='finish of depth range loss')
            parser.add_argument('--l_gan_loss', type=float, default=1.0, help='finish of depth range loss')
            parser.add_argument('--l_depth_max_iter', type=int, default=20000, help='max iter with big depth rec. loss')
            parser.add_argument('--num_iter_gen', type=int, default=1, help='iteration of gen per 1 iter of dis')
            parser.add_argument('--num_iter_dis', type=int, default=1, help='iteration of dis per 1 iter of gen')
            
#         parser.add_argument('--ngf_unet', type=int, default=64, help='# of gen filters in the first conv layer for image')
#         parser.add_argument('--norm_unet', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
#         parser.add_argument('--dropout_unet', type=bool, default=False, help='dropout for the generator')
        return parser
    
    
    def __init__(self, opt):
        super(A2BModel, self).__init__(opt)
        if self.isTrain:
            self.loss_names = ['D_A_depth', 'G_A', 'depth_range_A']
        self.visuals_names = ['real_img_A', 'real_depth_A', 'real_depth_B','fake_depth_B']
#         if self.isTrain:
#             self.visuals_names.append['mask']
        if self.isTrain:
            self.model_names = ['netG_A', 'netD_A']
        else: 
            self.model_names = ['netG_A']
        self.netG_A = network.define_Gen(opt, direction='A2B')

        if self.isTrain:
            self.netD_A = network.define_D(opt, input_type = 'depth')
            self.criterionGAN = network.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionDepthRange = network.MaskedL1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.l_depth_A = self.opt.l_depth_A_begin
#             self.holes_pred = network.define_Unet(opt)
#             self.load_holes_pred()
#             self.set_requires_grad([self.holes_pred], False)
    
    def set_input(self, input):
        self.real_img_A = input['A_img'].to(self.device)
        self.real_depth_A = input['A_depth'].to(self.device)
        self.real_depth_B = input['B_depth'].to(self.device)
#         if self.isTrain:
#             mask = self.holes_pred(self.real_img_A)
#             mask = torch.sigmoid(mask)
#             self.mask = mask > 0.5
#             real_depth_A = input['Bad_depth'].to(self.device)
#             self.real_depth_A = torch.where(self.mask, torch.ones_like(real_depth_A)*(-1.0), real_depth_A)
#         else: 
#             self.real_depth_A = input['Bad_depth'].to(self.device)

    
    def forward(self):
        self.fake_depth_B = self.netG_A(self.real_depth_A, self.real_img_A)
        
    def backward_D_base(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        loss_D = 0.5 * (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False))
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
        self.loss_D_A_depth = self.backward_D_base(self.netD_A, self.real_depth_B, self.fake_depth_B)
    
    def backward_G(self):
        loss_A = 0.0
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_depth_B), True) * self.opt.l_gan_loss
        loss_A = loss_A + self.loss_G_A
        
        if self.l_depth_A > 0 :
            self.loss_depth_range_A = self.criterionDepthRange(self.real_depth_A, self.fake_depth_B, self.real_depth_A > -1)* self.l_depth_A 
            loss_A = loss_A + self.loss_depth_range_A

        self.loss_G = loss_A 
        self.loss_G.backward()
        
    def optimize_param(self):
        self.set_requires_grad([self.netD_A], False)
        for _ in range(self.opt.num_iter_gen):
            self.forward()
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
        self.set_requires_grad([self.netD_A], True)
        
        self.set_requires_grad([self.netG_A], False)
        for j in range(self.opt.num_iter_dis):
            if j > 0:
                self.forward()
            self.optimizer_D.zero_grad()
            self.backward_D_A()
            self.optimizer_D.step()
        self.set_requires_grad([self.netG_A], True)
        
    def update_loss_weight(self, global_iter):
        pass
