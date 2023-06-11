# <p align="center">  MindDiffuser  </p> 
This is the official code for the paper "MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion"<br>

![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/overview.png)<br>

    Schematic diagram of MindDiffuser. <br>
    (a) Decoders are trained to fit fMRI with averaged CLIP text embeddings 𝑐, CLIP image feature 𝑍𝑖𝐶𝐿𝐼𝑃, and VQ-VAE latent feature 𝑧.<br>
    (b) The two-stage image reconstruction process. In stage 1, an initial reconstructed image is generated using the decoded CLIP text feature 𝑐 and VQ-VAE latent            feature 𝑧.<br>
    In stage 2, the decoded CLIP image feature is used as a constraint to iteratively adjust 𝑐 and 𝑧 until the final reconstruction result matches the original image in terms of both semantic and structure.
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/plane_00.png)<br>
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/four_sub_00.png)<br>
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/cortex_sub2_00.png)<br>

# <p align="center">  Preliminaries  </p> 
This code was developed and tested with:

*  Python version 3.8.5
*  PyTorch version 1.11.0
*  A100 40G
*  The conda environment defined in environment.yml


First, set up the conda enviroment as follows:<br>

    conda env create -f environment.yml  # create conda env
    conda activate MindDiffuser          # activate conda env

# <p align="center">  Data  </p> 
    NSD dataset http://naturalscenesdataset.org/
    Data preparation https://github.com/styvesg/nsd
