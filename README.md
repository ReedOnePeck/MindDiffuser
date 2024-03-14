# <p align="center">  MindDiffuser  </p> 
This is the official code for the paper "MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion"[**ACMMM2023**] (https://dl.acm.org/doi/10.1145/3581783.3613832) 

## <p align="center">  Schematic diagram of MindDiffuser  </p> 
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/Picture2.png)<br>
- (a) Decoders are trained to fit fMRI with averaged CLIP text embeddings 𝑐, CLIP image feature 𝑍𝑖𝐶𝐿𝐼𝑃, and VQ-VAE latent feature 𝑧.
- (b) The two-stage image reconstruction process. In stage 1, an initial reconstructed image is generated using the decoded CLIP text feature 𝑐 and VQ-VAE latent feature 𝑧. In stage 2, the decoded CLIP image feature is used as a constraint to iteratively adjust 𝑐 and 𝑧 until the final reconstruction result matches the original image in terms of both semantic and structure.

## <p align="center">  Algorithm diagram of MindDiffuser  </p> 
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/A.png)

## <p align="center">  A brief comparison of image reconstruction results </p> 
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/plane_00.png)

## <p align="center"> Reconstruction results of MindDiffuser on multiple subjects </p>
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/four_sub_00.png)

## <p align="center">  Experiments  </p> 
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/1686488621334.png)

## <p align="center"> Interpretability analysis </p>
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/cortex_sub2_00.png)

During the feature decoding process, we use L2-regularized linear regression model to automatically select voxels to fit three types
of feature: semantic feature 𝑐, detail feature 𝑧, and structural feature 𝑍𝐶𝐿𝐼𝑃. We ultilize pycortex to project the weights of each 
voxel in the fitted model onto the corresponding 3D coordinates in the visual cortex.




# <p> Steps to reproduce MindDiffuser </p>

If you are pressed for time or unable to reproduce my work, you can also directly extract the reconstruction results of MindDiffuser on subjects 1, 2, 5, and 7 from Baidu Netdisk for comparison.

[百度网盘](https://pan.baidu.com/s/1TtmXEd-fOidlMOzuOtphxg?pwd=izxl)  提取码：izxl 

![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/Pan.png)



## <p>  Preliminaries  </p> 
This code was developed and tested with:

*  Python version 3.8.5
*  PyTorch version 1.11.0
*  A100 40G
*  The conda environment defined in environment_1.yaml

## <p>  Dataset downloading and preparation </p> 
`NSD dataset` http://naturalscenesdataset.org/  <br>
`Data preparation` https://github.com/styvesg/nsd  <br>
- After preprocessing the NSD data, please organize the image stimuli in the training set into a .npy file with dimensions (8859, 3, 512, 512), and the image stimuli in the test set  into a .npy file with dimensions (982, 3, 512, 512), stored in ：your_folder/data/stimuli_data/. And store the fMRI data in ：your_folder/data/response_data/.
- Download "captions_train2017.json" and "captions_val2017.json" from the official website of the COCO dataset(https://cocodataset.org/#download). Save them in the path "your_folder/data/utils_data/".
- Run the code(https://github.com/styvesg/nsd/blob/master/data_preparation.ipynb) to obtain the textual descriptions of the stimulus images from NSD in the COCO dataset.Rename the corresponding file as "cocoID_correct.npy"  and save it in the path "your_folder/data/utils_data/".

![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/NSD.png)

## <p> Model downloading and preparation  </p>
First, set up the conda enviroment as follows:<br>

    conda env create -f environment_1.yml  # create conda env
    conda activate MindDiffuser          # activate conda env  <br>
- To ensure stable execution of our project, it is recommended to first create the virtual environment of Stable Diffusion v1-4 and then add the required Python packages to it. <br>
- You need to download the checkpoint file :sd-v1-4.ckpt and the config file :v1-inference.yaml for Stable Diffusion v1-4 from Hugging Face. Store them in the folders :/yourfolder/data/pretrained_models/checkpoint/: and :/yourfolder/data/pretrained_models/config/ respectively. <br>
- After downloading the "v1-inference.yaml" file, change the value of "max_length" to 15 in line 72.


## <p> Feature extraction </p>
    cd your_folder
    python Feature extractor/Semantic_feature_extraction.py
    python Feature extractor/detail_extracttion.py
    python Feature extractor/Structural_feature_extraction.py
    python Feature extractor/Structural_feature_selection.py


## <p> Feature decoding </p>
    cd your_folder
    python Feature decoding/Semantic_feature_decoding.py
    python Feature decoding/Structural_feature_decoding.py
    python Feature decoding/detail_decoding.py
    
## <p> Image reconstruction </p>

    cd your_folder
    python Image reconstruction/Reconstruction.py

## <p> Reproduce the results of "High-resolution image reconstruction with latent diffusion models from human brain activity"(CVPR2023)  </p>
After extracting and decoding the features, run the following code：

    cd your_folder
    python Reproduce Takagi's results/image_reconstruction.py
## <p> Reproduce the results of "Reconstruction of Perceived Images from fMRI Patterns and Semantic Brain Exploration using Instance-Conditioned GANs" </p>
After configuring the environment and codes provided by Ozcelik, run the following codes:

    cd your_folder
    python Reproduce Ozcelik's results/extract_features.py
    python Reproduce Ozcelik's results/train_regression.py
    python Reproduce Ozcelik's results/reconstruct_images.py

## <p> Cite </p>
Please cite our paper if you use this code in your own work:<br>

    @inproceedings{10.1145/3581783.3613832,
    author = {Lu, Yizhuo and Du, Changde and Zhou, Qiongyi and Wang, Dianpeng and He, Huiguang},
    title = {MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion},
    year = {2023},
    isbn = {9798400701085},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3581783.3613832},
    doi = {10.1145/3581783.3613832},
    booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
    pages = {5899–5908},
    numpages = {10},
    keywords = {fmri, brain-computer interface (bci), probabilistic diffusion model, controlled image reconstruction},
    location = {Ottawa ON, Canada},
    series = {MM '23}
    }
    





