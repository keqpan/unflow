from options.test_options import TestOptions
from dataloader import create_dataset
from models import create_model
from utils.visualizer import Visualizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    opt = TestOptions().parse()
    torch.cuda.set_device(opt.gpu_ids[0])
    dataset = create_dataset(opt)
    vis = Visualizer(opt)
    dataset_size = len(dataset)        
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup()
    if opt.phase == 'test':
        L1_loss = []
        L1_loss_syn = []
        L1_cycle = []
        L1_cycle_syn = []
        dif = []
        dif_syn = []
        for data in tqdm(dataset.dataloader):
            model.set_input(data)
            model.test()
            if opt.save_img:
                vis.save_img_metric(model.get_current_vis(), opt.img_dir, opt.name, opt.phase)
            L1_loss.append(model.get_L1_loss())
            L1_loss_syn.append(model.get_L1_loss_syn())
            L1_cycle.append(model.get_L1_loss_cycle())
            L1_cycle_syn.append(model.get_L1_loss_cycle_syn())
            dif.append(model.get_dif())
            dif_syn.append(model.get_dif_syn())
        mean = np.mean(L1_loss)
        std = np.std(L1_loss, ddof=1)
        mean_s = np.mean(L1_loss_syn)
        std_s = np.std(L1_loss_syn, ddof=1)
        
        mean_cycle = np.mean(L1_cycle)
        std_cycle = np.std(L1_cycle, ddof=1)
        mean_cycle_s = np.mean(L1_cycle_syn)
        std_cycle_s = np.std(L1_cycle_syn, ddof=1)
        
        mean_dif = np.mean(dif)
        std_dif = np.std(dif, ddof=1)
        mean_dif_syn = np.mean(dif_syn)
        std_dif_syn = np.std(dif_syn, ddof=1)
        
        print('{:04.3f}\u00B1{:04.3f} Difference between real and r2s'.format(mean, std))
        print('{:04.3f}\u00B1{:04.3f} Difference between syn and s2r'.format(mean_s, std_s))
        print()
        print('{:04.3f}\u00B1{:04.3f} Difference between real and cycle real'.format(mean_cycle, std_cycle))
        print('{:04.3f}\u00B1{:04.3f} Difference between syn and cycle syn'.format(mean_cycle_s, std_cycle_s))
        print()
        print('{:04.3f}\u00B1{:04.3f} r2s - real'.format(mean_dif, std_dif))
        print('{:04.3f}\u00B1{:04.3f} s2r - syn'.format(mean_dif_syn, std_dif_syn))
        
    else:
        for data in tqdm(dataset.dataloader):
            model.set_input(data)
            model.test()
            vis.save_img(model.get_current_vis(), opt.img_dir, opt.name, opt.phase)