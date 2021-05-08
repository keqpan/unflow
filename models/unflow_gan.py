import itertools
# from util.image_pool import ImagePool
from .base_model import BaseModel
from . import network
from utils.util import GaussianSmoothing
from utils import util
import torch
import torch.nn as nn
from models.semi_cycle_gan import SemiCycleGANModel
'''
Pool for previous img
--------------------
ngf = 64
n_bloc=4
n_dis=6
Cycle loss on new holes?
how add normal loss
'''
class UnFlowGANModel(SemiCycleGANModel):
    
#     def __init__(self, opt):
#         super().__init__(opt)
#         print("self.loss_names", self.loss_names)

    def backward_D_A(self):
#         fake_B = self.fake_B_pool
        if self.opt.disc_for_depth:
            self.loss_D_A_depth = self.backward_D_base(self.netD_A_depth, self.rec_depth_B, self.fake_depth_B)
        if self.opt.disc_for_normals:
            self.loss_D_A_normal = self.backward_D_base(self.netD_A_normal, self.rec_norm_B, self.fake_norm_B)
