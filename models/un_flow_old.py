import itertools
# from util.image_pool import ImagePool
from .base_model import BaseModel
from . import network
from util.util import GaussianSmoothing
from util import util
import torch
import torch.nn as nn
import random
'''
Pool for previous img
--------------------
Weight decay =0.0001
spectral norm disc
ngf = 64
n_bloc=4
n_dis=6
Cycle loss on new holes?
normalize norm?
num iter gen on new pictures?
identical and cycle on same pic?
how add normal loss
'''
class SemiCycleGANModel(BaseModel, nn.Module):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--l_cycle_A_begin', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--l_cycle_A_end', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--l_cycle_B_begin', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--l_cycle_B_end', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--l_identity', type=float, default=0.0, help='identical loss')
            parser.add_argument('--l_reconstruction_semantic', type=float, default=0.0, help='weight for reconstruction loss')
            parser.add_argument('--l_depth_A_begin', type=float, default=5.0, help='start of depth range loss')
            parser.add_argument('--l_depth_A_end', type=float, default=0.0, help='finish of depth range loss')
            parser.add_argument('--l_depth_B_begin', type=float, default=5.0, help='start of depth range loss')
            parser.add_argument('--l_depth_B_end', type=float, default=0.0, help='finish of depth range loss')
            parser.add_argument('--l_depth_max_iter', type=int, default=5000, help='max iter with big depth rec. loss')
            parser.add_argument('--use_blur', action='store_true', help='use bluring for l1 loss')
            parser.add_argument('--num_iter_gen', type=int, default=1, help='iteration of gen per 1 iter of dis')
            parser.add_argument('--num_iter_dis', type=int, default=1, help='iteration of dis per 1 iter of gen')
#             parser.add_argument('--mean_A', type=float, default=1680.1208394737955, help='mean of mean depth in A')
#             parser.add_argument('--std_A', type=float, default=487.0543836544621, help='std of mean depth in A')
#             parser.add_argument('--mean_B', type=float, default=2781.0011373752295, help='mean of mean depth in B')
#             parser.add_argument('--std_B', type=float, default=780.4723869231325, help='std of mean depth in B')     
        return parser
    
    def __init__(self, opt):
        super(SemiCycleGANModel, self).__init__(opt)
        if self.isTrain:
            self.loss_names = ['D_A_depth', 'D_B_depth', 'G_A', 'G_B', 'cycle_A']
            if self.opt.use_second_cycle:
                self.loss_names.append('cycle_B')
            if self.opt.use_semantic:
                self.loss_names.append('rec_semantic_A')
            if self.opt.disc_for_normals:
                self.loss_names.extend(['D_A_normal', 'D_B_normal',])
            if self.opt.l_identity > 0 :
                self.loss_names.extend(['idt_A', 'idt_B',])
            if self.opt.l_depth_A_begin > 0:
                self.loss_names.append('depth_range_A')
            if self.opt.l_depth_B_begin > 0:
                self.loss_names.append('depth_range_B')
                
        self.visuals_names = ['real_img_A', 'real_depth_A',
                              'real_img_B', 'real_depth_B',
                              'fake_depth_B', 'fake_depth_A', 'rec_depth_A', 'name_A', 'name_B']
        if self.opt.use_second_cycle:
            self.visuals_names.append('rec_depth_B')
        if self.opt.use_mean_matching:
            self.visuals_names.extend(['real_noise_A', 'fake_noise_A', 'real_noise_B', 'fake_noise_B'])
        if self.isTrain:
            if self.opt.use_semantic:
                self.visuals_names.extend(['real_semantic_A', 'rec_semantic_A']) 
            if self.opt.l_identity > 0:
                self.visuals_names.extend(['idt_A', 'idt_B'])
        
        if self.isTrain:
            self.model_names = ['netG_A', 'netG_B', 'netD_A_depth', 'netD_B_depth']
            if self.opt.disc_for_normals:
                self.model_names.extend([ 'netD_A_normal', 'netD_B_normal'])
        else: 
            self.model_names = ['netG_A', 'netG_B']
        
#         if self.opt.old_generator:
#             self.netG_A = network.define_G(opt, direction='A2B')
#             self.netG_B = network.define_G(opt, direction='B2A')
#         else:
        self.netG_A = network.define_Gen(opt, direction='A2B')
        self.netG_B = network.define_Gen(opt, direction='B2A')
            
        if self.isTrain:
            self.netD_A_depth = network.define_D(opt, input_type = 'depth')
            self.netD_B_depth = network.define_D(opt, input_type = 'depth')
            self.disc = [self.netD_A_depth, self.netD_B_depth]
            if self.opt.disc_for_normals:
                self.netD_A_normal = network.define_D(opt, input_type = 'normal')
                self.netD_B_normal = network.define_D(opt, input_type = 'normal')
                self.disc.extend([self.netD_A_normal, self.netD_B_normal])
#             self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
#             self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.criterionGAN = network.GANLoss(self.opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle_A = network.MaskedL1Loss()
            self.criterionCycle_B = nn.L1Loss()
            self.criterionIdt = nn.L1Loss()
            self.criterionDepthRange = network.MaskedL1Loss()
            if self.opt.use_semantic:
#                 weight_class = torch.tensor([3.0]).to(self.device)   #HYPERPARAM
                self.criterionSemantic = nn.CrossEntropyLoss()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(*[m.parameters() for m in self.disc]), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.extend([self.optimizer_G, self.optimizer_D])
            
            self.surf_normals = network.SurfaceNormals()
#             self.gaus_blur = GaussianSmoothing(1, 7, 10, self.device)  #HYPERPARAM
            self.l_depth_A = self.opt.l_depth_A_begin
            self.l_depth_B = self.opt.l_depth_B_begin
            self.l_cycle_A = self.opt.l_cycle_A_begin
            self.l_cycle_B = self.opt.l_cycle_B_begin
            self.mean_matching = network.MeanMatching() 
            
#             self.mu_shift = (self.opt.mean_B - self.opt.mean_A) / (self.opt.max_distance / 2)
#             self.std_shift = (self.opt.std_B - self.opt.std_A) / (self.opt.max_distance / 2) 
    
    def set_input(self, input):
        self.name_A = input['A_name']
        self.name_B = input['B_name']
        self.real_img_A = input['A_img'].to(self.device) #, non_blocking=torch.cuda.is_available()
        self.real_depth_A = input['A_depth'].to(self.device)
        if self.opt.use_semantic and self.isTrain:
            self.real_semantic_A = input['A_semantic'].to(self.device)
        self.real_img_B = input['B_img'].to(self.device)
        self.real_depth_B = input['B_depth'].to(self.device)
    
    def forward(self):
        if self.opt.use_semantic and self.isTrain:
            self.fake_depth_B, self.fake_semantic_A = self.netG_A(self.real_depth_A, self.real_img_A, return_logits=True)
        else:
            self.fake_depth_B = self.netG_A(self.real_depth_A, self.real_img_A)
        self.fake_depth_A = self.netG_B(self.real_depth_B)
        
        self.rec_depth_A = self.netG_B(self.fake_depth_B)
        if self.opt.use_second_cycle:
            self.rec_depth_B = self.netG_A(self.fake_depth_A, self.real_img_B)
        
        if self.isTrain:
            if self.opt.l_identity > 0:
                self.idt_A = self.netG_A(self.real_depth_B, self.real_img_B)
                self.idt_B = self.netG_B(self.real_depth_A)
        
    def backward_D_base(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        loss_D = 0.5 * (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False))
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
#         fake_B = self.fake_B_pool
        if self.opt.use_mean_matching:
            self.real_noise_B, self.fake_noise_B = self.mean_matching(self.real_depth_B, self.fake_depth_B)
            self.loss_D_A_depth = self.backward_D_base(self.netD_A_depth, self.real_noise_B, self.fake_noise_B)
        else:
            self.loss_D_A_depth = self.backward_D_base(self.netD_A_depth, self.real_depth_B, self.fake_depth_B)
        if self.opt.disc_for_normals:
            self.loss_D_A_normal = self.backward_D_base(self.netD_A_normal, self.surf_normals(self.real_depth_B), self.surf_normals(self.fake_depth_B))
    
    def backward_D_B(self):
        if self.opt.use_mean_matching:
            self.real_noise_A, self.fake_noise_A = self.mean_matching(self.real_depth_A, self.fake_depth_A)
            self.loss_D_B_depth = self.backward_D_base(self.netD_B_depth, self.real_noise_A, self.fake_noise_A )
        else:
            self.loss_D_B_depth = self.backward_D_base(self.netD_B_depth, self.real_depth_A, self.fake_depth_A)
        if self.opt.disc_for_normals:
            self.loss_D_B_normal = self.backward_D_base(self.netD_B_normal, self.surf_normals(self.real_depth_A), self.surf_normals(self.fake_depth_A))
    
    def backward_G(self):
        loss_A = 0.0
        loss_B = 0.0
        self.loss_G_A = self.criterionGAN(self.netD_A_depth(self.fake_depth_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B_depth(self.fake_depth_A), True)
        if self.opt.disc_for_normals:
            self.loss_G_A = self.loss_G_A + self.criterionGAN(self.netD_A_normal(self.surf_normals(self.fake_depth_B)), True)
            self.loss_G_B = self.loss_G_B + self.criterionGAN(self.netD_B_normal(self.surf_normals(self.fake_depth_A)), True)
        loss_A = loss_A + self.loss_G_A
        loss_B = loss_B + self.loss_G_B
        
        self.loss_cycle_A = self.criterionCycle_A(self.rec_depth_A, self.real_depth_A, (self.real_depth_A > -1.0)) * self.l_cycle_A           ###HYPERPARAM
        loss_A = loss_A + self.loss_cycle_A
        if self.opt.use_second_cycle:
            self.loss_cycle_B = self.criterionCycle_B(self.rec_depth_B, self.real_depth_B) * self.l_cycle_B
            loss_B = loss_B + self.loss_cycle_B
        
        if self.opt.l_identity > 0:
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_depth_B) * self.opt.l_identity
            loss_A = loss_A + self.loss_idt_A
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_depth_A) * self.opt.l_identity
            loss_B = loss_B + self.loss_idt_B    
        
        if self.opt.use_semantic:
            self.loss_rec_semantic_A = self.criterionSemantic(self.fake_semantic_A, self.real_semantic_A) * self.opt.l_reconstruction_semantic
            loss_A = loss_A + self.loss_rec_semantic_A

        if self.l_depth_A > 0 :
            if self.opt.use_blur:
                self.loss_depth_range_A = self.criterionDepthRange(self.gaus_blur(self.fake_depth_B), self.gaus_blur(self.real_depth_A), (self.real_depth_A > -1.0)) * self.l_depth_A
            else:
                self.loss_depth_range_A = self.criterionDepthRange(self.fake_depth_B, self.real_depth_A, (self.real_depth_A > -1.0))* self.l_depth_A 
            loss_A = loss_A + self.loss_depth_range_A

        if self.l_depth_B > 0:
            if self.opt.use_blur:
                self.loss_depth_range_B = self.criterionDepthRange(self.gaus_blur(self.fake_depth_A), self.gaus_blur(self.real_depth_B), (self.fake_depth_A > -1.0)) * self.l_depth_B
            else:
                self.loss_depth_range_B = self.criterionDepthRange(self.fake_depth_A, self.real_depth_B, (self.fake_depth_A > -1.0)) * self.l_depth_B
            loss_B = loss_B + self.loss_depth_range_B
        
        loss_G = loss_A + loss_B
        loss_G.backward()
        
    def optimize_param(self):
        self.set_requires_grad(self.disc, False)
        for _ in range(self.opt.num_iter_gen):
            self.forward()
            self.optimizer_G.zero_grad()
#             self.zero_grad([self.netG_A, self.netG_B])
            self.backward_G()
            self.optimizer_G.step()
        self.set_requires_grad(self.disc, True)
        
        self.set_requires_grad([self.netG_A, self.netG_B], False)
        for j in range(self.opt.num_iter_dis):
            if j > 0:
                self.forward()
            self.optimizer_D.zero_grad()
#             self.zero_grad(self.disc)
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()
        self.set_requires_grad([self.netG_A, self.netG_B], True)
                
    def update_loss_weight(self, global_iter):
        if global_iter > self.opt.l_depth_max_iter:
            self.l_depth_A = self.opt.l_depth_A_end
            self.l_depth_B = self.opt.l_depth_B_end
            self.l_cycle_A = self.opt.l_cycle_A_end
            self.l_cycle_B = self.opt.l_cycle_B_end

    def get_L1_loss(self):
        return network.MaskedL1Loss()(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.fake_depth_B, self.opt) , self.real_depth_A > -1.0).item()
    def get_L1_loss_syn(self):
        return network.MaskedL1Loss()(util.data_to_meters(self.real_depth_B, self.opt), util.data_to_meters(self.fake_depth_A, self.opt) , self.fake_depth_A > -1.0).item()
    
    def get_dif(self):
        return network.MaskedLoss()(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.fake_depth_B, self.opt) , self.real_depth_A > -1.0).item()
    
    def get_dif_syn(self):
        return network.MaskedLoss()(util.data_to_meters(self.real_depth_B, self.opt), util.data_to_meters(self.fake_depth_A, self.opt) , self.fake_depth_A > -1.0).item()

    
    
#             parser.add_argument('--lambda_attention', type=float, default=1.0, help='weight for attention loss')
#         if self.opt.attention:
#             l_attention = self.opt.lambda_attention
#             self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_depth_B)[0], True)
#             self.loss_G_A_logit = self.criterionGAN(self.netD_A(self.fake_depth_B)[1], True)
#             self.loss_attention_A = 0.5 * (self.BCE(self.logit_A2B, torch.ones_like(self.logit_A2B).to(self.device)) + self.BCE(self.logit_B2B, torch.zeros_like(self.logit_B2B).to(self.device))) * l_attention
            
#             self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_depth_A)[0], True)
#             self.loss_G_B_logit = self.criterionGAN(self.netD_B(self.fake_depth_A)[1], True)
#             self.loss_attention_B = 0.5 * (self.BCE(self.logit_B2A, torch.ones_like(self.logit_B2A).to(self.device)) + self.BCE(self.logit_A2A, torch.zeros_like(self.logit_A2A).to(self.device))) * l_attention
#             loss_A = loss_A + self.loss_G_A + self.loss_G_A_logit + self.loss_attention_A
#             loss_B = loss_B + self.loss_G_B + self.loss_G_B_logit + self.loss_attention_B
#         else:


#         if self.opt.attention:
#             self.fake_depth_B, self.logit_A2B = self.netG_A(self.real_depth_A, self.real_img_A, attention_flag=self.opt.attention)
#             self.idt_A, self.logit_B2B = self.netG_A(self.real_depth_B, self.real_img_B, attention_flag=self.opt.attention) # G_A should be identity if real_B is fed: ||G_A(B) - B||
#             self.fake_depth_A, self.logit_B2A = self.netG_B(self.real_depth_B, attention_flag=self.opt.attention)
#             self.idt_B, self.logit_A2A = self.netG_B(self.real_depth_A, attention_flag=self.opt.attention)
#             self.rec_depth_A, _ = self.netG_B(self.fake_depth_B, attention_flag=self.opt.attention) # Cycle 
#         else:

#         if self.opt.attention:
#             pred_real, gap_logit_real = netD(real, self.opt.attention)
#             pred_fake, gap_logit_fake = netD(fake.detach(), self.opt.attention)   #.detach()
#             loss_main = 0.5 * (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False))
#             loss_attention = 0.5 * self.opt.lambda_attention * (self.criterionGAN(gap_logit_real, True) + self.criterionGAN(gap_logit_fake, False))
#             loss_D  =  loss_main + loss_attention
#             loss_D.backward()
#         else: