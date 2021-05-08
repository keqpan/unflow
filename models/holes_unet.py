import torch
import torch.nn as nn
from .base_model import BaseModel
from . import network

class HolesUnetModel(BaseModel, nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--weight_hole', type=float, default=1.5, help='weight for hole class')
            parser.add_argument('--ngf_unet', type=int, default=64, help='# of gen filters in the first conv layer for image')
            parser.add_argument('--norm_unet', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
            parser.add_argument('--dropout_unet', type=bool, default=False, help='dropout for the generator')
        return parser
    
    def __init__(self, opt):
        super(HolesUnetModel, self).__init__(opt)
        if self.isTrain:
            self.loss_names = ['Binary_CrossEntropy']
        self.visuals_names = ['Image', 'GT_H', 'Pred_H']
        self.model_names = ['Unet']
        self.Unet = network.define_Unet(opt)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([opt.weight_hole]).to(self.device))
        self.optimizer = torch.optim.Adam(self.Unet.parameters(), lr = opt.lr_G, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer)
    
    def set_input(self, input):
        self.Image = input['Image'].to(self.device)
        self.GT_H = input['GT_H'].to(self.device)
    
    def forward(self):
        self.Pred_H = self.Unet(self.Image)
    
    def backward(self):
        self.loss_Binary_CrossEntropy = self.criterion(self.Pred_H, self.GT_H)
        self.loss_Binary_CrossEntropy.backward()
    
    def optimize_param(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        
    def update_loss_weight(self,global_iter):
        pass
