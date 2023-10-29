# <p align="center">  MindDiffuser  </p> 
This is the official code for the paper "MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion"[**ACMMM2023**] (https://dl.acm.org/doi/10.1145/3581783.3613832) <br>


![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/Picture2.png)<br>
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/A.png)<br>

    Schematic diagram of MindDiffuser. 
    (a) Decoders are trained to fit fMRI with averaged CLIP text embeddings 𝑐, CLIP image feature 𝑍𝑖𝐶𝐿𝐼𝑃, and VQ-VAE latent feature 𝑧.
    (b) The two-stage image reconstruction process. In stage 1, an initial reconstructed image is generated using the decoded CLIP text feature 𝑐 and VQ-VAE latent feature 𝑧. In stage 2, the decoded CLIP image feature is used as a constraint to iteratively adjust 𝑐 and 𝑧 until the final reconstruction result matches the original image in terms of both semantic and structure.
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/plane_00.png)<br>

    A brief comparison of image reconstruction results.
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/four_sub_00.png)<br>

    Reconstruction results of MindDiffuser on multiple subjects
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/cortex_sub2_00.png)<br>

    During the feature decoding process, we use L2-regularized linear regression model to automatically select voxels to fit three types
    of feature: semantic feature 𝑐, detail feature 𝑧, and structural feature 𝑍𝐶𝐿𝐼𝑃. We ultilize pycortex to project the weights of each 
    voxel in the fitted model onto the corresponding 3D coordinates in the visual cortex.

# <p align="center">  Preliminaries  </p> 
This code was developed and tested with:

*  Python version 3.8.5
*  PyTorch version 1.11.0
*  A100 40G
*  The conda environment defined in environment_1.yml

# <p align="center">  Dataset  </p> 
`NSD dataset` http://naturalscenesdataset.org/  <br>
`Data preparation` https://github.com/styvesg/nsd
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/NSD.png)

# <p align="center">  Experiments  </p> 
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/1686488621334.png)

## <p> MindDiffuser </p>
### <p> Model preparation  </p>
First, set up the conda enviroment as follows:<br>

    conda env create -f environment_1.yml  # create conda env
    conda activate MindDiffuser          # activate conda env

### <p> Feature extraction </p>
    cd your_folder
    python Feature extractor/Semantic_feature_extraction.py
    python Feature extractor/detail_extracttion.py
    python Feature extractor/Structural_feature_extraction.py
    python Feature extractor/Structural_feature_selection.py

### <p> Feature decoding </p>
    cd your_folder
    python Feature decoding/Semantic_feature_decoding.py
    python Feature decoding/Structural_feature_decoding.py
    python Feature decoding/detail_decoding.py
    
### <p> Image reconstruction </p>
I will upload files such as `features.npy`,`mask.npy`, `checkpoints of decoders`, etc. to the checkpoints folder.

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
    





