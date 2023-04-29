# UDA-VAE++ for cardiac segmentation
This repository contains the official Pytorch implementation
of the paper: \
"Unsupervised Domain Adaptation for Cardiac Segmentation: Towards Structure Mutual Information Maximization" (accepted by CVPRW 2022)

## Updates:
- 2022.4.21: Paper is available at [https://arxiv.org/abs/2204.09334](https://arxiv.org/abs/2204.09334)
- 2022.4.21: Upload the code
- 2022.6.19: CVF Open Access [Link](https://openaccess.thecvf.com/content/CVPR2022W/Precognition/html/Lu_Unsupervised_Domain_Adaptation_for_Cardiac_Segmentation_Towards_Structure_Mutual_Information_CVPRW_2022_paper.html)

## Prerequisites
* Windows or Linux (tested on CentOS)
* CUDA 10.2, or CPU
* Python 3.6+
* Pytorch 1.2.0 (high pytorch version could decrease the performance)
* torchvision
* SimpleITK
* scikit-image
* numpy

# Preparing Dataset
Please refer to [UDA-VAE](https://github.com/FupingWu90/VarDA/tree/main/Dataset)

# Model Arch

## Main workflow
![workflow](/repo/UDA-VAE2.pdf)

## Mutual Information Maximization
![mim](/repo/MINE2.pdf)

# Training
Run the following script in terminal
```
python train.py
```

## Hyperparameters and explanations

```python
parser.add_argument('--batch_size',default=12,type=int,help='batch_size')#
    parser.add_argument('--epoch',default=30,type=int,help='epoch')#
    parser.add_argument('--predLambda',default=1e3,type=float,help='segmentation loss')#
    parser.add_argument('--disLambda',default=1e-3,type=float,help='domain distance loss 256 40 40(img size in multi-scale)')#
    parser.add_argument('--disLambda2',default=1e-3,type=float,help='domain distance loss 128 80 80 (img size in multi-scale)')#
    parser.add_argument('--disLambda3',default=1e-4,type=float,help='domain distance loss 64 160 160 (img size in multi-scale)')#
    parser.add_argument('--kldLambda',default=1.0,type=float,help='VAE loss')#
    parser.add_argument('--data_dir',default='./Dataset/Patch192',help='data_dir+(/source)or(/target)')#
    parser.add_argument('--source_num',default=35,type=int,help='number in source domain')#
    parser.add_argument('--target_num',default=45,type=int,help='number in target domain')#
    parser.add_argument('--lr',default=1e-4,type=float,help='learning rate')#
    parser.add_argument('--decay',default=1e-5,type=float,help='lr decay')#
    parser.add_argument('--save_dir',default='./save/save_train_param_num',type=str,help='save path')#
    parser.add_argument('--save_test_dir',default='./save/save_test_label_num',type=str,help='save test path')#
    parser.add_argument('--test_dir',default=['./Dataset/Patch192/LGE/LGE_Test/','./Dataset/Patch192/LGE/LGE_Vali/'],help='test dir')#
    parser.add_argument('--kernel',default=4,type=int,help='gaussian kernel size')#
    parser.add_argument('--global_weight',default=0.5,type=float,help='global MI')# we set the weight the same as the paper Deepinfo Max
    parser.add_argument('--local_weight',default=1.0,type=float,help='local MI')#
    parser.add_argument('--prior_weight',default=0.1,type=float,help='prior_MI')#
    parser.add_argument('--needMI',default=True,type=bool,help='whether need MI') #
    parser.add_argument('--gpu',default='0',type=str,help='gpu_number') #
    parser.add_argument('--num_worker',default=10,type=int) #
    parser.add_argument('--miLambda',default=100,type=float,help='mutual information loss')
```

# BibTeX

If you find this repo useful, please cite our paper:
```
@inproceedings{lu2022unsupervised,
  title={Unsupervised Domain Adaptation for Cardiac Segmentation: Towards Structure Mutual Information Maximization},
  author={Lu, Changjie and Zheng, Shen and Gupta, Gaurav},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2588--2597},
  year={2022}
}
```

# Contact
Please reach lucha@kean.edu for further questions.\
You can also open an issue (prefered) or a pull request in this Github repository.\
I am willing to answer your question

# Acknowledgement
Some parts of the repository is borrowed from [UDA-VAE](https://github.com/FupingWu90/VarDA). Thanks for sharing!

# TODO List
- [x] List (important) hyperparameters
- [x] Upload Arxiv Link
- [x] Upload BibTeX
- [x] Add hyperparameter explanation
- [x] Add Dependencies
- [x] Upload Model Architecture Figure
- [x] Finalize readme

