import glob
import tqdm
import os
from multiprocessing import Pool

import imageio
import numpy as np
import sys
import numpy
from scipy import signal
from scipy import ndimage
from skimage.transform import resize

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = numpy.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

data_part = "train"
fnames = sorted(os.listdir("/data/unpaired_depth/Scannet_data/depths_{}/".format(data_part)))

os.makedirs("/data/unpaired_depth/Scannet_cleaned/depths_{}_gt".format(data_part), exist_ok=True)
os.makedirs("/data/unpaired_depth/Scannet_cleaned/depths_{}".format(data_part), exist_ok=True)
os.makedirs("/data/unpaired_depth/Scannet_cleaned/images_{}".format(data_part), exist_ok=True)

def calc_ssim(fname):
    fname_base, ext = os.path.splitext(fname)
    img = imageio.imread("/data/unpaired_depth/Scannet_data/images_{}/{}{}".format(data_part, fname_base, ".jpg"))
    img = resize(img, (960, 1280), preserve_range=True).astype(np.uint8)
    depth_lq = imageio.imread("/data/unpaired_depth/Scannet_data/depths_{}/{}".format(data_part, fname))
    depth_gt = imageio.imread("/data/unpaired_depth/Scannet_data/depths_{}_gt/{}".format(data_part, fname))
    depth_gt = depth_gt[1::2,1::2]
    
    ssim_mtrx = ssim(depth_lq, depth_gt)
    ssim_mtrx_strided = np.lib.stride_tricks.as_strided(ssim_mtrx, shape=(5,8,192,192), strides=(64*ssim_mtrx.strides[0], 64*ssim_mtrx.strides[1], ssim_mtrx.strides[0], ssim_mtrx.strides[1]))
    depth_gt_strided = np.lib.stride_tricks.as_strided(depth_gt, shape=(5,8,192,192), strides=(64*depth_gt.strides[0], 64*depth_gt.strides[1], depth_gt.strides[0], depth_gt.strides[1]))
    depth_lq_strided = np.lib.stride_tricks.as_strided(depth_lq, shape=(5,8,192,192), strides=(64*depth_lq.strides[0], 64*depth_lq.strides[1], depth_lq.strides[0], depth_lq.strides[1]))
    img_strided = np.lib.stride_tricks.as_strided(img, shape=(5,8,192*2,192*2,3), strides=(2*64*img.strides[0], 2*64*img.strides[1], img.strides[0], img.strides[1], img.strides[2]))

    ssim_idx = ssim_mtrx_strided.mean(axis=3).mean(axis=2) > 0.8
    n = ssim_idx.sum()

    if n > 0:
        depth_gt_good = depth_gt_strided[ssim_idx]
        depth_lq_good = depth_lq_strided[ssim_idx]
        img_good = img_strided[ssim_idx]
        final_idx = (depth_gt_good > 10).mean(axis=2).mean(axis=1) > 0.9
        n_good = final_idx.sum()
        if n_good == 0:
            return
        else:
            depth_gt_good = depth_gt_good[final_idx]
            depth_lq_good = depth_lq_good[final_idx]
            img_good = img_good[final_idx]

            for i in range(n_good):
                imageio.imsave("/data/unpaired_depth/Scannet_cleaned/images_{}/{}_{}{}".format(data_part, fname_base, i, ".jpg"), img_good[i])
                imageio.imsave("/data/unpaired_depth/Scannet_cleaned/depths_{}/{}_{}{}".format(data_part, fname_base, i, ext), depth_lq_good[i])
                imageio.imsave("/data/unpaired_depth/Scannet_cleaned/depths_{}_gt/{}_{}{}".format(data_part, fname_base, i, ext), depth_gt_good[i])

n_processes = 20
# fnames = fnames[:5000] #14626 5000 for test
fnames = fnames[:15000]
with Pool(n_processes) as p:   
    res = list(tqdm.tqdm(p.imap(func=calc_ssim, iterable=fnames), total=len(fnames)))