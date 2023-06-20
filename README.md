# Diffusion Models for Contrast Harmonization

We provide the code of our MIDL 2023 submission ["Diffusion Models for Contrast Harmonization of Magnetic Resonance Images"](https://openreview.net/forum?id=Xs_Hd23_PP).
This repository about diffusion models for contrast harmonization is an adaption of the [implementation](https://github.com/JuliaWolleb/Diffusion-based-Segmentation) of ["Diffusion Models for Implicit Image Segmentation Ensembles"](https://arxiv.org/abs/2112.03145), which is based on ["Improved Denoising Diffusion Probabilistic Models"](https://github.com/openai/improved-diffusion).

## Paper Abstract

Magnetic resonance (MR) images from multiple sources often show differences in image contrast related to acquisition settings or the used scanner type. For long-term studies, longitudinal comparability is essential but can be impaired by these contrast differences, leading to biased results when using automated evaluation tools. This study presents a diffusion model-based approach for contrast harmonization. We use a data set consisting of scans of 18 Multiple Sclerosis patients and 22 healthy controls. Each subject was scanned in two MR scanners of different magnetic field strengths 1.5 T and 3 T, resulting in a paired data set that shows scanner-inherent differences. We map images from the source contrast to the target contrast for both directions, from 3 T to 1.5 T and from 1.5 T to 3 T. As we only want to change the contrast, not the anatomical information, our method uses the original image to guide the image-to-image translation process by adding structural information. The aim is that the mapped scans display increased comparability with scans of the target contrast for downstream tasks. We evaluate this method for the task of segmentation of cerebrospinal fluid, grey matter and white matter. Our method achieves good and consistent results for both directions of the mapping.

## Usage

Create a *data* folder. For the dataloader, which can be found in the file *guided_diffusion/mrloader.py*, the paired 2D slices need to be stored in the structure shown below for training. Here, the abbreviations "BL" and "FU" are used for baseline (1.5 T) and follow-up (3 T), these identifying names can be changed, as long as they are also changed in *guided_diffusion/mrloader.py* (lines 29 and 31). 
The test folder structure differs slightly from the training folder structure, as we directly group the slices belonging to one subject. In this example, FU scans are the ground truth, we transform from BL to FU contrast. 

```
data
└───training
│   └───42_1
│       │   BL_42_slice1_BL.nii.gz
│       │   FU_42_slice1_FU.nii.gz
│   └───42_2
│       │  BL_42_slice2_BL.nii.gz
│       │  FU_42_slice2_FU.nii.gz
│   └───42_3
│       │  ...
└───test
│   └───50_test (will be transformed to target contrast using the trained model)
│       └───50_1
│           │   BL_50_slice1_BL.nii.gz
│       └───50_2
│           │   BL_50_slice2_BL.nii.gz
│       └───50_3
│           │  ...
│   └───50_test_gt (only used for visualization / difference calculation)
│       └───50_1
│           │   FU_50_slice1_FU.nii.gz
│       └───50_2
│           │   FU_50_slice2_FU.nii.gz
│       └───50_3
│           │  ...

```
Change the following files:

```
- logger.py                 - Where to save logs & results (adapt line 442).
- mrloader.py               - Define names of input images and ground truths (lines 29 and 31). Make sure ground truth is defined as the last channel during training!
                              Define part to be cropped (lines 62, 68 and 69) to achieve image and label size [224, 224].
- dist_util.py              - Define GPU number (line 27).
- segmentation_sample.py    - Adapt test ID, output folder and ground truth folder (lines 55, 56, 82).
```

Set the flags as follows (adapt if necessary for your data):

```
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4"

```

To train the model, run

```
python3 scripts/segmentation_train.py --data_dir data/training $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```
The model will be saved in the *results* folder.
For sampling run:

```

python scripts/segmentation_sample.py --data_dir dataset/test/50_test  --model_path results_training/savedmodel250000.pt --num_ensemble=1 $MODEL_FLAGS $DIFFUSION_FLAGS

```
Create a *results* folder. The generated images will be stored in the *results* folder. A visualization of the sampling process can be done using [visdom](https://github.com/fossasia/visdom).

## Evaluation

For the evaluation we compared our method against [DeepHarmony](https://www.sciencedirect.com/science/article/pii/S0730725X18306490?via%3Dihub) and [pGAN](https://github.com/icon-lab/pGAN-cGAN)
regarding mean squared error and histogram differences between the original and the generated images. Additionally, we compared the performance of pGAN and our method regarding [FAST](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FAST) segmentation of cerebrospinal fluid, white matter and grey matter.