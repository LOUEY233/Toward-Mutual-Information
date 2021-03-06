# UDA-VAE++ for cardiac segmentation
This repository contains the official Pytorch implementation
of the paper: \
"Unsupervised Domain Adaptation for Cardiac Segmentation: Towards Structure Mutual Information Maximization" (accepted by CVPRW 2022)

## Updates:
- 2022.4.21: Paper is available at [https://arxiv.org/abs/2204.09334](https://arxiv.org/abs/2204.09334)
- 2022.4.21: Upload the code
- 2022.6.19: CVF Open Access [Link](https://openaccess.thecvf.com/content/CVPR2022W/Precognition/html/Lu_Unsupervised_Domain_Adaptation_for_Cardiac_Segmentation_Towards_Structure_Mutual_Information_CVPRW_2022_paper.html)

## Prerequisites
* Windows or Linux
* CUDA 10.2, or CPU
* Python 3.6+
* Pytorch 1.2.0 (high pytorch version could decrease the results)
* torchvision
* SimpleITK
* scikit-image
* numpy

# Preparing Dataset
Please refer to [UDA-VAE](https://github.com/FupingWu90/VarDA/tree/main/Dataset)

# Training
Run the following script in terminal
```
python train.py
```

# Contact
Please reach lucha@kean.edu for further questions.\
You can also open an issue (prefered) or a pull request in this Github repository.\
I am willing to answer your question

# Acknowledgement
Some parts of the repository is borrowed from [UDA-VAE](https://github.com/FupingWu90/VarDA). Thanks for sharing!

# TODO List
- [ ] List (important) hyperparameters
- [x] Upload Arxiv Link
- [ ] Upload BibTeX
- [ ] Add hyperparameter explanation
- [x] Add Dependencies
- [x] Upload Model Architecture Figure
- [x] Upload Visual Comparisons
- [ ] Finalize readme

