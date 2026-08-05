## Investigating Spurious Cue Sensitivity in Semantic Segmentation
This repository contains the official implementation for our paper, "Investigating Spurious Cue Sensitivity in Semantic Segmentation," which was accepted as an Oral presentation at ANNPR 2026 (Artificial Neural Networks in Pattern Recognition).
The full paper is available [here](https://github.com/muhammadahsan/ShortcutsImageSegmentation/blob/main/ShortcutsImageSegmentation.pdf). We hope this repository facilitates reproducing our experimental results and serves as a useful resource for future research. 
## Repository Structure
The repository is organized into model-specific directories. Each directory (e.g., UNet, Trans-UNet, and Swin-Unet) contains:

- Training scripts  
- Testing & evaluation scripts
- Utility functions used throughout the experiments
## Environment Setup
We recommend creating a Python 3.7 environment before installing the required dependencies.

pip install -r requirements.txt

## Prepare data
- International Skin Imaging Collaboration dataset (ISIC)
is an international effort to improve melanoma diagnosis, sponsored by the International Society for Digital Imaging of the Skin (ISDIS). The ISIC Archive contains the largest publicly available collection of quality controlled dermoscopic images of skin lesions. It is widely used for research on skin cancer detection and contains images labeled as benign or malignant. ISIC contains non-clinical artifacts, such as color patches, rulers, and hairs, that can induce spurious correlations. [ISIC](https://challenge.isic-archive.com)

- Black Cats Brown Dogs Dataset (BCBD)
We construct the Black Cats Brown Dogs (BCBD) dataset from the Oxford-IIIT Pet Dataset, which provides pixel-level segmentation masks for images of various cat and dog breeds.
To obtain color-specific subgroups, we first applied an automated color detection pipeline to the segmented animal regions and subsequently verified and corrected the assignments through manual annotation, ensuring reliable color labels. [Cats & Dogs](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/parkhi12cat.pdf)

## Metadata
The metadata files used in our experiments are included in this repository. These files contain the labels and supplementary information required to reproduce the experimental results.
- ISIC metadata: /UNet/ISIC_Metadata
- Cats & Dogs metadata: /UNet/UNet_Pet_Semantic_Seg/Metadata

## Checkpoints
To download the archive containing all pre-trained model weights from Google Drive

- All pre-trained model weights are available in the [Google Drive folder](https://drive.google.com/file/d/12aHhrLonTKSGbxLCBq_PEwxTrAkXv7-N/view?usp=drive_link).

## Running
Run the train script on each model folders like UNt, Trans-UNet, or Swin-Unet. The batch size we used is 64. If you do not have enough GPU memory, the batch size can be reduced to 16 or 8 to save memory and both can reach similar performance.
- Train
  
Run the train script i.e; Unet_ISIC_Seg_Training.ipynb in case of UNet [here](/UNet/Unet_ISIC_Seg_Training.ipynb)
- Test
  
Run the test script i.e; Unet_ISIC_Seg_Testing.ipynb in case of UNet [here](/UNet/Unet_ISIC_Seg_Testing.ipynb)
