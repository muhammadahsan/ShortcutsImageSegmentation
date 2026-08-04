## Investigating Spurious Cue Sensitivity in Semantic Segmentation
The code for the work "Investigating Spurious Cue Sensitivity in Semantic Segmentation". Our paper has been accepted as an Oral at ANNPR 2026 (Artificial Neural Networks in Pattern Recognition - 2026). Find the full paper [here](https://github.com/muhammadahsan/ShortcutsImageSegmentation/blob/main/ShortcutsImageSegmentation.pdf). I hope this will help you to reproduce the results. 

The repository is organized into model-specific directories.  
Each folder (e.g., `UNet, Trans-UNet, or Swin-Unet`) contains:

- training scripts  
- testing/evaluation scripts
- utility functions used in the experiments

## Environment
Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## Prepare data
- International Skin Imaging Collaboration dataset (ISIC)
ISIC is an international effort to improve melanoma diagnosis, sponsored by the International Society for Digital Imaging of the Skin (ISDIS). The ISIC Archive contains the largest publicly available collection of quality controlled dermoscopic images of skin lesions. It is widely used for research on skin cancer detection and contains images labeled as benign or malignant. ISIC contains non-clinical artifacts, such as color patches, rulers, and hairs, that can induce spurious correlations. [ISIC](https://challenge.isic-archive.com)

- Black Cats Brown Dogs Dataset (BCBD)
We construct the Black Cats Brown Dogs (BCBD) dataset from the Oxford-IIIT Pet Dataset, which provides pixel-level segmentation masks for images of various cat and dog breeds.
To obtain color-specific subgroups, we first applied an automated color detection pipeline to the segmented animal regions and subsequently verified and corrected the assignments through manual annotation, ensuring reliable color labels.

## Train/Test


