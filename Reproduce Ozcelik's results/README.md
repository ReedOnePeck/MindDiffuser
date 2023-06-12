# IC-GAN fMRI Reconstruction

Official repository for the **IJCNN 2022 (Accepted Oral)** paper ["**Reconstruction of Perceived Images from fMRI Patterns and Semantic Brain Exploration using Instance-Conditioned GANs**"](https://arxiv.org/abs/2202.12692) by Furkan Ozcelik, Bhavin Choksi, Milad Mozafari, Leila Reddy, Rufin VanRullen.


## Results
The following are a few reconstructions obtained : 
<p align="center"><img src="./figures/Reconstructions.png" width="600" ></p>


## Requirements
- Create conda environment using environment.yml in ic_gan directory by entering `conda env create -f environment.yml` . You can also create environment by checking requirements yourself. 
- For preparation of Kamitani images you should also include some required libraries `pip install pandas scikit-image imageio `
- Before loading ICGAN model you should download the pretrained model and required library using:
```
cd ic_gan
chmod u+x download-weights.sh
sh ./download-weights
pip install pytorch-pretrained-biggan
```
- For copyright reasons, we cannot share images used in this study. You can request access to Imagenet images used in Generic Object Decoding study by applying this [form](https://forms.gle/ujvA34948Xg49jdn9) as stated in [KamitaniLab/GenericObjectDecoding](https://github.com/KamitaniLab/GenericObjectDecoding) repository. Downloaded "images" directory should be added to KamitaniData dir.
- Inverted noise and dense vectors are provided in this [link](https://drive.google.com/file/d/13H_onuCqnexpINDuusraN2jB0asgDo-n/view?usp=sharing) together with extracted instance features but you can extract the instance features your self by applying first 2 steps stated below. (Noise and Dense vector inversion codes will be provided in future.)

## Usage Instructions
After setting up the environment and downloading the images provided with form;
1.  Preprocess images using 
	```
	cd KamitaniData 
	python kamitani_image_prepare.py
	```
2.  Extract instance features of Kamitani images using 
`python extract_features.py`
3. Train regression models using 
`python train_regression.py -sub 3`
(You can change the subject num between 1-5)
You need extracted features in order to run this code successfully.
4. Reconstruct images from test fMRI data using
`python reconstruct_images.py -sub 3`
5. Explore ROI semantics by ROI maximization using
`python explore_roi_semantics.py -sub 3`



## References
- Codes in KamitaniData directory are derived from [WeizmannVision/ssfmri2im](https://github.com/WeizmannVision/ssfmri2im)
- Codes in ic_gan directory are derived from [facebookresearch/ic_gan](https://github.com/facebookresearch/ic_gan)
- Dataset used in the studies are obtained from [KamitaniLab/GenericObjectDecoding](https://github.com/KamitaniLab/GenericObjectDecoding)
