from torch.utils.data import Dataset
import glob
import numpy as np
import SimpleITK as sitk
from skimage import transform
import torch

class LGE_TrainSet(Dataset):
    def __init__(self,dir,sample_num):
        self.imgdir=dir+'/LGE/'

        self.imgsname = glob.glob(self.imgdir + '*LGE.nii*')

        imgs = np.zeros((1,192,192))
        self.info = []
        self.times = int((35.0 / sample_num) * 4)
        for img_num in range(sample_num):
            itkimg = sitk.ReadImage(self.imgsname[img_num])
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
            npimg = npimg.astype(np.float32)

            imgs = np.concatenate((imgs,npimg),axis=0)
            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs[1:,:,:]

    def __getitem__(self, item):
        imgindex,crop_indice = divmod(item,self.times)

        npimg = self.imgs[imgindex,:,:]
        randx = np.random.randint(-16,16)
        randy = np.random.randint(-16, 16)
        npimg=npimg[96+randx-80:96+randx+80,96+randy-80:96+randy+80]

        # npimg_o = transform.resize(npimg, (80, 80),
        #                      order=3, mode='edge', preserve_range=True)
        #npimg_resize = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        npimg_down2 = transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(npimg, (40,40 ), order=3,mode='edge', preserve_range=True)

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor),torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)

    def __len__(self):

        return self.imgs.shape[0]*self.times



class C0_TrainSet(Dataset):
    def __init__(self,dir,sample_num):
        self.imgdir = dir+'/C0/'

        self.imgsname = glob.glob(self.imgdir + '*C0.nii*')

        imgs = np.zeros((1,192,192))
        labs = np.zeros((1,192,192))
        self.info = []
        self.times = int((35.0 / sample_num) * 4)
        for img_num in range(sample_num):
            itkimg = sitk.ReadImage(self.imgsname[img_num])
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

            imgs = np.concatenate((imgs,npimg),axis=0)

            labname = self.imgsname[img_num].replace('.nii','_manual.nii')
            itklab = sitk.ReadImage(labname)
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

            labs = np.concatenate((labs, nplab), axis=0)

            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs[1:,:,:]
        self.labs = labs[1:,:,:]
        self.imgs.astype(np.float32)
        self.labs.astype(np.float32)



    def __getitem__(self, item):
        imgindex,crop_indice = divmod(item,self.times)

        npimg = self.imgs[imgindex,:,:]
        nplab = self.labs[imgindex,:,:]

        # npimg = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        # nplab = transform.resize(nplab, (96, 96), order=0,mode='edge', preserve_range=True)
        randx = np.random.randint(-16,16)
        randy = np.random.randint(-16, 16)
        npimg=npimg[96+randx-80:96+randx+80,96+randy-80:96+randy+80]
        nplab=nplab[96+randx-80:96+randx+80,96+randy-80:96+randy+80]

        # npimg_o=transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        # nplab_o=transform.resize(nplab, (80,80 ), order=0,mode='edge', preserve_range=True)

        npimg_down2 = transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(npimg, (40,40 ), order=3,mode='edge', preserve_range=True)

        nplab_down2 = transform.resize(nplab, (80,80 ), order=0,mode='edge', preserve_range=True)
        nplab_down4 = transform.resize(nplab, (40,40), order=0,mode='edge', preserve_range=True)

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(nplab).type(dtype=torch.LongTensor),torch.from_numpy(nplab_down2).type(dtype=torch.LongTensor),torch.from_numpy(nplab_down4).type(dtype=torch.LongTensor),torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)

    def __len__(self):
        return self.imgs.shape[0]*self.times