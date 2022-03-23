import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn


def dice_compute(pred, groundtruth):           #batchsize*channel*W*W
    # for j in range(pred.shape[0]):
    #     for i in range(pred.shape[1]):
    #         if np.sum(pred[j,i,:,:])==0 and np.sum(groundtruth[j,i,:,:])==0:
    #             pred[j, i, :, :]=pred[j, i, :, :]+1
    #             groundtruth[j, i, :, :]=groundtruth[j,i,:,:]+1
    #
    # dice = 2*np.sum(pred*groundtruth,axis=(2,3),dtype=np.float16)/(np.sum(pred,axis=(2,3),dtype=np.float16)+np.sum(groundtruth,axis=(2,3),dtype=np.float16))
    dice=[]
    for i in range(4):
        dice_i = 2*(np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)+0.0001)
        dice=dice+[dice_i]


    return np.array(dice,dtype=np.float32)


def IOU_compute(pred, groundtruth):
    iou=[]
    for i in range(4):
        iou_i = (np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)-np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)
        iou=iou+[iou_i]


    return np.array(iou,dtype=np.float32)


def Hausdorff_compute(pred,groundtruth,spacing):
    pred = np.squeeze(pred)
    groundtruth = np.squeeze(groundtruth)

    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)

    overlap_results = np.zeros((1,4, 5))
    surface_distance_results = np.zeros((1,4, 5))

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for i in range(4):
        pred_i = (pred==i).astype(np.float32)
        if np.sum(pred_i)==0:
            overlap_results[0,i,:]=0
            surface_distance_results[0,i,:]=0
        else:
            # Overlap measures
            overlap_measures_filter.Execute(ITKTrue==i, ITKPred==i)
            overlap_results[0,i, 0] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[0,i, 1] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[0,i, 2] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[0,i, 3] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[0,i, 4] = overlap_measures_filter.GetFalsePositiveError()
            # Hausdorff distance
            hausdorff_distance_filter.Execute(ITKTrue==i, ITKPred==i)

            surface_distance_results[0,i, 0] = hausdorff_distance_filter.GetHausdorffDistance()
            # Symmetric surface distance measures

            reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKTrue == i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(ITKTrue == i)
            statistics_image_filter = sitk.StatisticsImageFilter()
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(statistics_image_filter.GetSum())

            segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKPred==i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(ITKPred==i)
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

            # Multiply the binary surface segmentations with the distance maps. The resulting distance
            # maps contain non-zero values only on the surface (they can also contain zero on the surface)
            seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

            # Get all non-zero distances and then add zero distances if required.
            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
            seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
            ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
            # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
            # segmentations, though in our case it is. More on this below.
            surface_distance_results[0,i, 1] = np.mean(all_surface_distances)
            surface_distance_results[0,i, 2] = np.median(all_surface_distances)
            surface_distance_results[0,i, 3] = np.std(all_surface_distances)
            surface_distance_results[0,i, 4] = np.max(all_surface_distances)


    return overlap_results,surface_distance_results


def multi_dice_iou_compute(pred,label):
    truemax, truearg = torch.max(pred, 1, keepdim=False)
    truearg = truearg.detach().cpu().numpy()
    # nplabs = np.stack((truearg == 0, truearg == 1, truearg == 2, truearg == 3, \
    #                    truearg == 4, truearg == 5, truearg == 6, truearg == 7), 1)
    nplabs = np.stack((truearg == 0, truearg == 1, truearg == 2, truearg == 3, truearg == 4, truearg == 5), 1)
    # truelabel = (truearg == 0) * 550 + (truearg == 1) * 420 + (truearg == 2) * 600 + (truearg == 3) * 500 + \
    #             (truearg == 4) * 250 + (truearg == 5) * 850 + (truearg == 6) * 820 + (truearg == 7) * 0

    dice = dice_compute(nplabs, label.cpu().numpy())
    Iou = IOU_compute(nplabs, label.cpu().numpy())

    return dice,Iou


class BalancedBCELoss(nn.Module):
    def __init__(self,target):
        super(BalancedBCELoss,self).__init__()
        self.eps=1e-6
        weight = torch.tensor([torch.reciprocal(torch.sum(target==0).float()+self.eps),torch.reciprocal(torch.sum(target==1).float()+self.eps),torch.reciprocal(torch.sum(target==2).float()+self.eps),torch.reciprocal(torch.sum(target==3).float()+self.eps)])
        self.criterion = nn.CrossEntropyLoss(weight)

    def forward(self, output,target):
        loss = self.criterion(output,target)

        return loss



class Gaussian_Kernel_Function(nn.Module):
    def __init__(self,std):
        super(Gaussian_Kernel_Function, self).__init__()
        self.sigma=std**2

    def forward(self, fa,fb):
        asize = fa.size()
        bsize = fb.size()

        fa1 = fa.view(-1, 1, asize[1])
        fa2 = fa.view(1, -1, asize[1])

        fb1 = fb.view(-1, 1, bsize[1])
        fb2 = fb.view(1, -1, bsize[1])

        aa = fa1-fa2
        vaa = torch.mean(torch.exp(torch.div(-torch.pow(torch.norm(aa,2,dim=2),2),self.sigma)))

        bb = fb1-fb2
        vbb = torch.mean(torch.exp(torch.div(-torch.pow(torch.norm(bb,2,dim=2),2),self.sigma)))

        ab = fa1-fb2
        vab = torch.mean(torch.exp(torch.div(-torch.pow(torch.norm(ab,2,dim=2),2),self.sigma)))

        loss = vaa+vbb-2.0*vab

        return loss

class Get_distance_loss():
    def __init__(self,kernel=4):
        self.kernel = kernel
        self.gaussian_distance = Gaussian_Distance(self.kernel).cuda()

    def get_cluster(self,mu_a,logvar_a,mu_b,logvar_b):
        source_kl = []
        target_kl = []
        for i in range(len(mu_a)):
            kla = torch.mean(1 / 2 * (-logvar_a[i] + mu_a[i] + torch.exp(logvar_a[i]) - 1)).item()
            source_kl.append(kla)
            klb = torch.mean(1 / 2 * (-logvar_b[i] + mu_b[i] + torch.exp(logvar_b[i]) - 1)).item()
            target_kl.append(klb)
        new_mu_a = torch.zeros_like(mu_a)
        new_logvar_a = torch.zeros_like(logvar_a)
        new_mu_b = torch.zeros_like(mu_b)
        new_logvar_b = torch.zeros_like(logvar_b)

        source_kl = np.array(source_kl)
        target_kl = np.array(target_kl)
        index_mu_a = source_kl.argsort()
        index_mu_b = target_kl.argsort()
        # print(source_kl,target_kl)
        for i in range(len(index_mu_a)):
            new_mu_a[i] = mu_a[index_mu_a[i]]
            new_logvar_a[i] = logvar_a[index_mu_a[i]]
        for i in range(len(index_mu_b)):
            new_mu_b[i] = mu_b[index_mu_b[i]]
            new_logvar_b[i] = logvar_b[index_mu_b[i]]

        return new_mu_a,new_logvar_a,new_mu_b,new_logvar_b

    def get_loss(self,mu_a,logvar_a,mu_b,logvar_b):
        new_mu_a,new_logvar_a,new_mu_b,new_logvar_b = self.get_cluster(mu_a,logvar_a,mu_b,logvar_b)
        new_mu_a1,new_mu_a2,new_mu_a3 = new_mu_a.chunk(3,0)
        new_logvar_a1,new_logvar_a2,new_logvar_a3 = new_logvar_a.chunk(3,0)
        new_mu_b1,new_mu_b2,new_mu_b3 = new_mu_b.chunk(3,0)
        new_logvar_b1,new_logvar_b2,new_logvar_b3 = new_logvar_b.chunk(3,0)
        loss1 = self.gaussian_distance(new_mu_a1,new_logvar_a1,new_mu_b1,new_logvar_b1)
        loss2 = self.gaussian_distance(new_mu_a2,new_logvar_a2,new_mu_b2,new_logvar_b2)
        loss3 = self.gaussian_distance(new_mu_a3,new_logvar_a3,new_mu_b3,new_logvar_b3)

        return loss1+loss2+loss3


class Contrastive_loss():
    def __init__(self, kernel=4):
        self.kernel = kernel
        self.gaussian_distance = Gaussian_Distance(self.kernel).cuda()

    def get_loss(self,mu_a,logvar_a,mu_b,logvar_b):
        mu_rand = torch.rand_like(mu_a)
        logvar_rand = torch.rand_like(logvar_a)

        loss_t_s = self.gaussian_distance(mu_a,logvar_a,mu_b,logvar_b)
        loss_t_n = self.gaussian_distance(mu_b,logvar_b,mu_rand,logvar_rand)

        loss = loss_t_s/(loss_t_n+1e-6)*loss_t_s
        return loss


class Gaussian_Distance(nn.Module):
    def __init__(self,kern=1):
        super(Gaussian_Distance, self).__init__()
        self.kern=kern
        self.avgpool = nn.AvgPool2d(kernel_size=kern, stride=kern)


    def forward(self, mu_a,logvar_a,mu_b,logvar_b):
        mu_a = self.avgpool(mu_a)
        mu_b = self.avgpool(mu_b)

        var_a = self.avgpool(torch.exp(logvar_a))/(self.kern*self.kern)
        var_b = self.avgpool(torch.exp(logvar_b))/(self.kern*self.kern)

        mu_a1 = mu_a.view(mu_a.size(0),1,-1)
        mu_a2 = mu_a.view(1,mu_a.size(0),-1)
        var_a1 = var_a.view(var_a.size(0),1,-1)
        var_a2 = var_a.view(1,var_a.size(0),-1)

        mu_b1 = mu_b.view(mu_b.size(0),1,-1)
        mu_b2 = mu_b.view(1,mu_b.size(0),-1)
        var_b1 = var_b.view(var_b.size(0),1,-1)
        var_b2 = var_b.view(1,var_b.size(0),-1)

        vaa = torch.sum(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_a1-mu_a2,2),var_a1+var_a2),-0.5)),torch.sqrt(var_a1+var_a2)))
        vab = torch.sum(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_a1-mu_b2,2),var_a1+var_b2),-0.5)),torch.sqrt(var_a1+var_b2)))
        vbb = torch.sum(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_b1-mu_b2,2),var_b1+var_b2),-0.5)),torch.sqrt(var_b1+var_b2)))

        loss = vaa+vbb-torch.mul(vab,2.0)

        return loss

if __name__ == '__main__':
    net = Get_distance_loss(4)
    mu_a = torch.randn((12,256,40,40))
    mu_b = torch.randn((12,256,40,40))
    logvar_a = torch.randn((12,256,40,40))
    logvar_b = torch.randn((12,256,40,40))
    loss = net.get_loss(mu_a,logvar_a,mu_b,logvar_b)
    net2 = Gaussian_Distance(4)
    loss2 = net2(mu_a,logvar_a,mu_b,logvar_b)
    net3 = Contrastive_loss(4)
    loss3 = net3.get_loss(mu_a,logvar_a,mu_b,logvar_b) + net3.gaussian_distance(mu_a,logvar_a,mu_b,logvar_b)
    print(loss)
    print(loss2)
    print(loss3)
    # a = torch.randn((3,1,6,6))
    # a1,a2,a3 = torch.chunk(a,3,dim=0)
    # print(a1)