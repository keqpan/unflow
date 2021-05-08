from options.train_options import TrainOptions 
from dataloader import create_dataset
from models import create_model
from utils.visualizer import Visualizer

import wandb
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import os
import copy
from collections import OrderedDict 

def acc_loss(d_acc, d):
    output = OrderedDict([(key, d_acc[key] + d[key]) for key in d_acc.keys()])
    return output
def div_loss(d_acc, n, epoch):
    output = OrderedDict([(key, d_acc[key] / n) for key in d_acc.keys()])
    output['Epoch'] = epoch
    return output

def ploting_func(model):
    if model == 'semi_cycle_gan' or model == "unflow_gan":
        f = lambda imgs : vis.plot_imgScannet(imgs)
    elif model == 'img2depth':
        f = lambda imgs : vis.plot_img2depth(imgs)
    elif model == 'pretrain':
        f = lambda imgs : vis.plot_pretrain(imgs)
    elif model == 'pretrain_A_hole':
        f = lambda imgs : vis.plot_pretrain_A(imgs)
    else:
        raise NotImplementedError('model [%s] is not found' % model)
    return f
if __name__ == '__main__':
    seed_value = 101
    os.environ['PYTHONHASHSEED']=str(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    opt = TrainOptions().parse()   # get training options
    opt_v = copy.deepcopy(opt)
    opt_v.isTrain = False
    opt_v.phase = 'test'
    torch.cuda.set_device(opt.gpu_ids[0])
    torch.backends.cudnn.deterministic = opt.deterministic
    torch.backends.cudnn.benchmark = not opt.deterministic
#     torch.autograd.set_detect_anomaly(True)
    
    vis = Visualizer(opt)
    plot = ploting_func(opt.model)
    if not opt.debug:
        wandb.init(project="un_depth", name=opt.name)
        wandb.config.update(opt)
    dataset = create_dataset(opt)  
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = {}'.format(dataset_size))
#     dataset_v = create_dataset(opt_v)  
#     dataset_size_v = len(dataset_v)    # get the number of images in the dataset.
#     print('The number of test images = {}'.format(dataset_size_v))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup()
    if not opt.debug:
        wandb.watch(model)
    global_iter = 0
    global_iter_v = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        model.train_mode()
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            global_iter += 1
            model.set_input(data)
            model.optimize_param()
#             torch.cuda.empty_cache()
            model.update_loss_weight(global_iter)
            iter_finish_time = time.time()
            if global_iter % opt.loss_freq == 0:
                if not opt.debug:
                    wandb.log(model.get_current_losses(), step = global_iter)
            if global_iter % opt.img_freq == 0:
                print('{} img procesed out of {}, taken {:04.2f} sec per 1 batch'.format((i+1)*opt.batch_size, dataset_size, iter_finish_time - iter_start_time))
                fig = plot(model.get_current_vis())
                if not opt.debug:
                    wandb.log({"chart": fig}, step=global_iter)
                plt.close(fig)
#         print('Validation')
#         for i, data in enumerate(dataset_v):
#             global_iter_v += 1
#             model.set_input(data)
#             model.test()
#             model.calc_test_loss()
#             if i == 0:
#                 mean_loss = model.get_current_losses_test()
#             else:
#                 mean_loss = acc_loss(mean_loss, model.get_current_losses_test())
#             if global_iter_v % (dataset_size_v // opt.batch_size) == 0:
#                 fig = plot(model.get_current_vis())
#                 if not opt.debug:
#                     wandb.log({"chart_val": fig}, step=global_iter)
#                 plt.close(fig)
#         mean_loss = div_loss(mean_loss, dataset_size_v, epoch)
#         if not opt.debug:
#             wandb.log(mean_loss, step = global_iter)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch {}, iters {}'.format(epoch, global_iter))
            model.save_net(epoch)
        print('End of epoch {} / {} \t Time Taken: {:04.2f} sec'.format(epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    model.save_net('last')
    print('Finish')