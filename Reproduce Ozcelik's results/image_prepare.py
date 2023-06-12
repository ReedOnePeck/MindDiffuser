import numpy as np
val_stim = np.load('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_stim_multi_trial_data.npy')
val_img = val_stim.transpose(0, 2, 3, 1)

trn_stim = np.load('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_stim_data.npy')
trn_img = trn_stim.transpose(0, 2, 3, 1)

out_file = './图像重建/对照实验/ic-gan-recons/stims/images_256.npz'
np.savez(out_file, train_images=trn_img, test_images=val_img)
