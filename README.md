# mri-epilepsy-segmentation


The current project is devoted to the Focal Cortical Displasia detection on MRI T1w images.

![fcd](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0122252.g002&type=large)

credit:  https://doi.org/10.1371/journal.pone.0122252.g002

Focal cortical dysplasia (FCD) is one of the most common epileptogenic lesions associated with malformations of cortical development. The findings on magnetic resonance (MR) images are essential for diagnostics and surgical planning in patients with epilepsy. The accurate detection of the FCD relies on the radiologist professionalism, and in many cases, the lesion could be missed. Automated detection systems for FCD are now extensively developing, yet it requires large datasets with annotated data.  The aim of this study is to enhance the detection of FCD with the means of transfer learning.

## 1. Detection

`baseline` folder contains instructions and tests for medical-detection. 
Code reproduction for the paper: https://doi.org/10.1016/j.compmedimag.2019.101662

## 2. Segmentation

`segmentation` folder contains segmentation model train and transfer.
Code reproduction for the paper: https://arxiv.org/abs/2003.04696

## 3. Classification

`classification` folder contains fader net architecture and CNN interpretation unit as well as baseline model.
