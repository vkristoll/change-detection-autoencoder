# Unsupervised change detection in VHR images with convolutional autoencoder

This repository contains code related to the implementation of the 2nd unsupervised change detection method, as analyzed in the paper cited below:

V. Kristollari and V. Karathanassi, "Change Detection in VHR Imagery With Severe Co-Registration Errors Using Deep Learning: A Comparative Study," in IEEE Access, vol. 10, pp. 33723-33741, 2022, doi: 10.1109/ACCESS.2022.3161978.

It can be accessed in: https://ieeexplore.ieee.org/document/9740657

![CD study areas](/images/CD_study_areas.PNG)
![CD GA](/images/CD_GA.PNG)

## Code implementation guidelines
### *Create change maps for custom data*

> 1. Modify the folder paths in lines 46, 50, 55 in "create_change_map.py" and then run the file to create change maps with float values
that indicate the distance of the feature maps of the bi-temporal images.
>
> 2. Modify the folder paths in lines 12, 15 in "otsu_threshold.py" to create the binary change maps

*The weights of the trained autoencoder are given in "weights.h5". It has been trained on pansharpened 8 bit WV-2/3 and GE01 images (0.5 m spatial resolution, 4 bands (VNIR)). The network accepts patches of size 224x224x4.*

### *Train on custom data*
>1. Modify the folder paths in "create_training_data.py" and "training_batch_generator.py".
>
>2. Run "autoencoder_training.py"

## Optionally

> If the x, y coordinates of matching points are known, you can run "coregister.py" to create a co-registered image. The paths in lines
 13, 19 need to be modified first.

*Detailed guidelines are included inside each script.*


If you use this code, please cite the below paper.

```
@ARTICLE{Kristollari2022,
  author={Kristollari, Viktoria and Karathanassi, Vassilia},
  journal={IEEE Access}, 
  title={Change Detection in VHR Imagery With Severe Co-Registration Errors Using Deep Learning: A Comparative Study},
  year={2022},
  volume={10},  
  pages={33723-33741},
  doi={10.1109/ACCESS.2022.3161978}
  }
```
