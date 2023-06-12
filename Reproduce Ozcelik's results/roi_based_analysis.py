## This code reproduces the Figure 5a of the paper.
## It performs region-of-interest (ROI) based analysis of the regressions weights. 


import os
import bdpy
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')

import scipy
from scipy.stats import sem
import nibabel as nib


regression_weights_dir = '/path/to/regressionweights' #dir where regression weights are stored
subject_data_dir = '/path/to/subjectdata' #dir where Subjectdata (in .h5 format) is stored

all_rois = ['ROI_V1',
            'ROI_V2',
            'ROI_V3',
            'ROI_V4',
            'ROI_LOC',
            'ROI_FFA',
            'ROI_PPA',
            # 'ROI_LVC',
            # 'ROI_HVC',
            # 'ROI_VC'
            ]


## Helpers

percentilefunc = lambda arr: np.array([(len(list(np.where(np.array(arr)<=i)[0]))/len(arr))*100  for i in arr])

def get_regression_weights(sub,dim):

    fname = os.path.join(subject_data_dir,f"Subject{sub}.h5")
    subdata = bdpy.BData(fname)

    a = [f for f in os.listdir(regression_weights_dir) if f"Subject{sub}" in f and f"{dim}dim" in f]
    assert len(a) == 1

    print (f"Getting data from file : {a[0]}")
    with open(os.path.join(regression_weights_dir,a[0]),'rb') as f:
        data = pickle.load(f)
    
    l1weights = np.linalg.norm(data['coef_'],axis=0,ord=1)  #get the weights for each voxels (of shape (num_voxels,))
    return l1weights


def get_region_weights(sub,dim,roi=None,percentile=False,shuffle=False):

    fname = os.path.join(subject_data_dir,f"Subject{sub}.h5")
    subdata = bdpy.BData(fname)

    print (f'Getting data for sub{sub} dim{dim} roi_{roi}')

    l1weights = get_regression_weights(sub, dim)
    
    if shuffle == True:
        np.random.shuffle(l1weights)

    if percentile == True:
        l1weights = percentilefunc(l1weights)
    
    voldata = subdata.get_metadata(roi)
    voldata = voldata[2:-3]

    nonnan_indices = np.where(np.isnan(voldata) == False)[0]
    roi_l1weights = l1weights[nonnan_indices]
    
    voxel_x = subdata.get_metadata('voxel_x',where=roi) 
    assert np.nansum(voldata) == voxel_x.shape
    assert roi_l1weights.shape == voxel_x.shape

    return roi_l1weights


def get_semantic_index(sub,roi,dims=[2048,119],**kwargs):
    '''
    semantic_index = instance - noise / (instance + noise)
    '''

    d1weights = get_region_weights(sub, dims[0],roi,**kwargs)
    d2weights = get_region_weights(sub, dims[1],roi,**kwargs)

    semantic_index = ( d1weights - d2weights ) # / (d1weights + d2weights)

    assert semantic_index.shape == d1weights.shape == d2weights.shape

    return semantic_index



def get_plot1(savename=None):
  
  '''
  Legacy Plot to just plot the roiweights.
  '''

    fp = {'fontsize':24}

    plt.figure(figsize=(7,7))
    plt.title(f'Average percentile value of weights \n in a region w.r.t available brain',**fp)
    plt.ylabel('Percentiles',**fp)
    plt.xlabel('Regions',**fp)

    for dind, _dim in enumerate([2048,119,24576]):
        for xind,_roi in enumerate(all_rois):

            
            val_across_subs = []
            total_voxels = 0.
            
            for _sub in range(1,6):
    
                roiweights = get_region_weights(_sub, _dim , _roi,percentile=True)
                total_voxels += roiweights.shape[0]
                mean,std = roiweights.mean(),roiweights.std()
                val_across_subs.append(mean)

            submean,subsem = np.array(val_across_subs).mean() , sem(np.array(val_across_subs))

            hatchstyle = '/' if dind==2 else None
            fillstyle= True if dind==0 else False
            labelstyle = f"{_roi}" if fillstyle == True else None

            plt.bar(1.5*xind + 0.3*dind, submean, width=0.3 ,yerr=subsem,hatch=hatchstyle,fill=fillstyle,label=labelstyle)

    plt.legend(bbox_to_anchor=(1.35,1))
    if savename:
        plt.savefig(savename,bbox_inches='tight')




def get_plot2(dims_to_consider,savename=None):
    fp = {'fontsize':24}

    plt.figure(figsize=(7,7))
    plt.title(f'Semantic index using ranks(instance) minus ranks(dense) weights',**fp)
    plt.ylabel('Semantic index',**fp)
    plt.xlabel('Regions',**fp)

    
    roi_based_data = {r:None for r in all_rois}
    for xind,_roi in enumerate(all_rois):
    
        val_across_subs = []
        total_voxels = 0.
        for _sub in range(1,6):

            roiweights = get_semantic_index(_sub, _roi,dims=dims_to_consider,percentile=True)
            total_voxels += roiweights.shape[0]
            mean = roiweights.mean()
            val_across_subs.append(mean)

        roi_based_data[_roi] = val_across_subs.copy()

        submean,subsem = np.array(val_across_subs).mean() , sem(np.array(val_across_subs))
        
        fillstyle= True
        labelstyle = f"{_roi}" 

        plt.bar(xind , submean, width=0.5 ,yerr=subsem,hatch=hatchstyle,fill=fillstyle,label=labelstyle,capsize=2)

    plt.legend(bbox_to_anchor=(1.3,1.0))
    if savename:
        plt.savefig(savename,bbox_inches='tight')


def perform_analysis():
    import itertools
    from scipy.stats import ttest_ind

    for r1,r2 in itertools.combinations(roi_based_data.keys(),2):
            a,p = ttest_ind(roi_based_data[r1], roi_based_data[r2],equal_var=False)
            if p < 0.008:                                                               # (0.05/6 ~ 0.008) 
                print (f"{r1}  {r2}  {p}")



#%%

if __name__ == '__main__':
    get_plot2(dims_to_consider=[2048,24576])


