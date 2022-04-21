import numpy as np
import glob
import SimpleITK as sitk
import torch
from loss_function import dice_compute, Hausdorff_compute, IOU_compute
from network_wo_A import VAE


def SegNet_test_mr(test_dir, mrSegNet, gate,epoch,ePOCH, save_DIR,save_IMG_DIR):
    criterion=0
    for dir in test_dir:
        labsname = glob.glob(dir + '*manual.nii*')
        total_dice = np.zeros((4,))
        total_Iou = np.zeros((4,))

        total_overlap =np.zeros((1,4,5))
        total_surface_distance=np.zeros((1,4,5))

        num = 0
        mrSegNet.eval()
        for i in range(len(labsname)):
            itklab = sitk.ReadImage(labsname[i])
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

            imgname = labsname[i].replace('_manual.nii', '.nii')
            itkimg = sitk.ReadImage(imgname)
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
            npimg = npimg.astype(np.float32)

            # data = np.transpose(
            #     transform.resize(np.transpose(npimg, (1, 2, 0)), (96, 96),
            #                      order=3, mode='edge', preserve_range=True), (2, 0, 1))
            data=torch.from_numpy(np.expand_dims(npimg,axis=1)).type(dtype=torch.FloatTensor).cuda()
            label=torch.from_numpy(nplab).cuda()
            truearg = np.zeros((data.size(0),data.size(2),data.size(3)))

            for slice in range(data.size(0)):
                output,_,_, _, _, _ ,_,_,_,_,_,_,_,_,_,_,_,_,_,_= mrSegNet(data[slice:slice+1,:,:,:], gate)

                truemax, truearg0 = torch.max(output, 1, keepdim=False)
                truearg0 = truearg0.detach().cpu().numpy()
                truearg[slice:slice+1,:,:]=truearg0
            #truearg = np.transpose(transform.resize(np

            #
            # truemax, truearg = torch.max(output, 1, keepdim=False)
            # truearg = truearg.detach().cpu().numpy()
            # truearg = np.transpose(transform.resize(np.transpose(truearg, (1, 2, 0)), (192,192), order=0,mode='edge', preserve_range=True), (2, 0, 1)).astype(np.int64)

            dice = dice_compute(truearg,label.cpu().numpy())
            Iou = IOU_compute(truearg,label.cpu().numpy())
            overlap_result, surface_distance_result = Hausdorff_compute(truearg,label.cpu().numpy(),itkimg.GetSpacing())

            total_dice = np.vstack((total_dice,dice))
            total_Iou = np.vstack((total_Iou,Iou))

            total_overlap = np.concatenate((total_overlap,overlap_result),axis=0)
            total_surface_distance = np.concatenate((total_surface_distance,surface_distance_result),axis=0)

            num+=1
        if num==0:
            return
        else:
            meanDice = np.mean(total_dice[1:],axis=0)
            stdDice = np.std(total_dice[1:],axis=0)

            meanIou = np.mean(total_Iou[1:],axis=0)
            stdIou = np.std(total_Iou[1:],axis=0)

            mean_overlap = np.mean(total_overlap[1:], axis=0)
            std_overlap = np.std(total_overlap[1:], axis=0)

            mean_surface_distance = np.mean(total_surface_distance[1:], axis=0)
            std_surface_distance = np.std(total_surface_distance[1:], axis=0)

            if 'Vali' in dir:
                phase='validate'
            else:
                criterion = np.mean(meanDice[1:])
                phase='test'
            with open("%s/lge_testout_index.txt" % (save_DIR), "a") as f:
                f.writelines(["\n\nepoch:", str(epoch), " ",phase," ", "\n","meanDice:",""\
                                 ,str(meanDice.tolist()),"stdDice:","",str(stdDice.tolist()),"","\n","meanIou:","",str(meanIou.tolist()),"stdIou:","",str(stdIou.tolist()), \
                                  "", "\n\n","jaccard, dice, volume_similarity, false_negative, false_positive:", "\n","mean:", str(mean_overlap.tolist()),"\n", "std:", "", str(std_overlap.tolist()), \
                                  "", "\n\n","hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance:", "\n","mean:", str(mean_surface_distance.tolist()), "\n","std:", str(std_surface_distance.tolist())])
    return criterion

if __name__ == '__main__':
    vaeencoder = VAE().cuda()
    dataset_dir = './Dataset/Patch192'
    TestDir = [dataset_dir+'/LGE/LGE_Test/',dataset_dir+'/LGE/LGE_Vali/']
    epoch = 1
    EPOCH = 30
    sample_num = 45
    prefix = './save'
    SAVE_DIR = prefix + '/save_train_param' + '_num' + str(sample_num)
    SAVE_IMG_DIR = prefix+'/save_test_label'+'_num'+str(sample_num)
    criter = SegNet_test_mr(TestDir, vaeencoder, 0, epoch, EPOCH, SAVE_DIR, SAVE_IMG_DIR)
    print(criter)
