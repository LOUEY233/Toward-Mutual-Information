import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from test import SegNet_test_mr
from torch.utils.data import Dataset, DataLoader
from config import *
import os
from network_wo_A import *
from loss_function import *
from dataload import *
from MINE2 import *
import argparse


def ADA_Train(source_MINE,target_MINE,Train_LoaderA,Train_LoaderB,encoder,decoderA,decoderAdown2,decoderAdown4,decoderB,decoderBdown2,decoderBdown4,gate,DistanceNet,lr,kldlamda,predlamda,dislamda,dislamdadown2,dislamdadown4,epoch,optim, optim_loss,savedir):
    lr=lr*(0.9**(epoch))
    for param_group in optim.param_groups:
        param_group['lr'] = lr
    if args.needMI:
        for param_group in optim_loss.param_groups:
            param_group['lr'] = lr


    A_iter = iter(Train_LoaderA)
    B_iter = iter(Train_LoaderB)

    i=0

    while i<len(A_iter) and i<len(B_iter):
        ct,ct_down2,ct_down4,label,label_down2,label_down4 ,info_ct= A_iter.next()
        mr,mr_down2,mr_down4,info_mr= B_iter.next()

        ct= ct.cuda()
        ct_down2= ct_down2.cuda()
        ct_down4= ct_down4.cuda()
        info_ct = info_ct.cuda()

        mr= mr.cuda()
        mr_down4= mr_down4.cuda()
        mr_down2= mr_down2.cuda()
        info_mr = info_mr.cuda()

        label= label.cuda()
        label_onehot =torch.FloatTensor(label.size(0), 4,label.size(1),label.size(2)).cuda()
        label_onehot.zero_()
        label_onehot.scatter_(1, label.unsqueeze(dim=1), 1)

        label_down2= label_down2.cuda()
        label_down2_onehot =torch.FloatTensor(label_down2.size(0), 4,label_down2.size(1),label_down2.size(2)).cuda()
        label_down2_onehot.zero_()
        label_down2_onehot.scatter_(1, label_down2.unsqueeze(dim=1), 1)

        label_down4= label_down4.cuda()
        label_down4_onehot =torch.FloatTensor(label_down4.size(0), 4,label_down4.size(1),label_down4.size(2)).cuda()
        label_down4_onehot.zero_()
        label_down4_onehot.scatter_(1, label_down4.unsqueeze(dim=1), 1)

        fusionseg,pred_ct, out_ct,feat_ct, mu_ct,logvar_ct, pred_down2_ct, outdown2_ct,featdown2_ct, mudown2_ct,logvardown2_ct,pred_down4_ct, outdown4_ct,featdown4_ct, mudown4_ct,logvardown4_ct,info_pred_ct,deout2_ct,deout3_ct,deout4_ct= encoder(ct,gate)
        #info_pred_ct = Infonet(info_pred_ct)

        info_cri = nn.CrossEntropyLoss().cuda()
        #infoloss_ct = info_cri(info_pred_ct,info_ct)

        seg_criterian = BalancedBCELoss(label)   # 4
        seg_criterian = seg_criterian.cuda()
        segloss_output = seg_criterian(out_ct, label)
        fusionsegloss_output = seg_criterian(fusionseg, label)

        segdown2_criterian = BalancedBCELoss(label_down2) # 4
        segdown2_criterian = segdown2_criterian.cuda()
        segdown2loss_output = segdown2_criterian(outdown2_ct, label_down2)

        segdown4_criterian = BalancedBCELoss(label_down4) # 4
        segdown4_criterian = segdown4_criterian.cuda()
        segdown4loss_output = segdown4_criterian(outdown4_ct, label_down4)

        recon_ct=decoderA(feat_ct,label_onehot)       # 128
        BCE_ct = F.binary_cross_entropy(recon_ct, ct)
        KLD_ct = -0.5 * torch.mean(1 + logvar_ct - mu_ct.pow(2) - logvar_ct.exp())

        recondown2_ct=decoderAdown2(featdown2_ct,label_down2_onehot)
        BCE_down2_ct = F.binary_cross_entropy(recondown2_ct, ct_down2)
        KLD_down2_ct = -0.5 * torch.mean(1 + logvardown2_ct - mudown2_ct.pow(2) - logvardown2_ct.exp())

        recondown4_ct=decoderAdown4(featdown4_ct,label_down4_onehot)
        BCE_down4_ct = F.binary_cross_entropy(recondown4_ct, ct_down4)
        KLD_down4_ct = -0.5 * torch.mean(1 + logvardown4_ct - mudown4_ct.pow(2) - logvardown4_ct.exp())

        _,pred_mr,out_mr,feat_mr, mu_mr,logvar_mr, preddown2_mr, seg_outdown4_mr,featdown2_mr, mudown2_mr,logvardown2_mr,preddown4_mr, seg_outdown2_mr,featdown4_mr, mudown4_mr,logvardown4_mr,info_pred_mr,deout2_mr,deout3_mr,deout4_mr= encoder(mr,gate)
        #info_pred_mr = Infonet(info_pred_mr)

        #infoloss_mr = info_cri(info_pred_mr,info_mr)

        recon_mr=decoderB(feat_mr,pred_mr)
        BCE_mr = F.binary_cross_entropy(recon_mr, mr)
        KLD_mr = -0.5 * torch.mean(1 + logvar_mr - mu_mr.pow(2) - logvar_mr.exp())

        recondown2_mr=decoderBdown2(featdown2_mr,preddown2_mr)
        BCE_down2_mr = F.binary_cross_entropy(recondown2_mr, mr_down2)
        KLD_down2_mr = -0.5 * torch.mean(1 + logvardown2_mr - mudown2_mr.pow(2) - logvardown2_mr.exp())

        recondown4_mr=decoderBdown4(featdown4_mr,preddown4_mr)
        BCE_down4_mr = F.binary_cross_entropy(recondown4_mr, mr_down4)
        KLD_down4_mr = -0.5 * torch.mean(1 + logvardown4_mr - mudown4_mr.pow(2) - logvardown4_mr.exp())

        distance_loss = DistanceNet.get_loss(mu_ct,logvar_ct,mu_mr,logvar_mr) + DistanceNet.gaussian_distance(mu_ct,logvar_ct,mu_mr,logvar_mr)
        distance_down2_loss = DistanceNet.get_loss(mudown2_ct,logvardown2_ct,mudown2_mr,logvardown2_mr) + DistanceNet.gaussian_distance(mudown2_ct,logvardown2_ct,mudown2_mr,logvardown2_mr)
        distance_down4_loss = DistanceNet.get_loss(mudown4_ct,logvardown4_ct,mudown4_mr,logvardown4_mr) + DistanceNet.gaussian_distance(mudown4_ct,logvardown4_ct,mudown4_mr,logvardown4_mr)

        # MI_loss
        # print(seg_outdown4_mr.size(), seg_outdown2_mr.size(), out_mr.size())
        # print(recondown4_mr.size(), recondown2_mr.size(), recon_mr.size())
        if args.needMI:
            loss_target = target_MINE(seg_outdown2_mr, seg_outdown4_mr, out_mr, recondown4_mr, recondown2_ct,
                                      recon_mr).cuda()
            loss_target = loss_target.mean()
            loss_source = source_MINE(outdown4_ct, outdown2_ct, out_ct, recondown4_ct, recondown2_ct, recon_ct).cuda()
            loss_source = loss_source.mean()

        balanced_loss = 10.0*BCE_mr+torch.mul(KLD_mr,kldlamda)+10.0*BCE_ct+torch.mul(KLD_ct,kldlamda)+torch.mul(distance_loss,dislamda)+predlamda*(segloss_output+fusionsegloss_output)+ \
                        10.0*BCE_down2_ct + torch.mul(KLD_down2_ct, kldlamda) + 10.0*BCE_down2_mr + torch.mul(KLD_down2_mr, kldlamda) + torch.mul(distance_down2_loss, dislamdadown2) + predlamda * segdown2loss_output+ \
                        10.0*BCE_down4_ct + torch.mul(KLD_down4_ct, kldlamda) + 10.0*BCE_down4_mr + torch.mul(KLD_down4_mr, kldlamda) + torch.mul(distance_down4_loss, dislamdadown4) + predlamda * segdown4loss_output
        if args.needMI:
            balanced_loss += args.miLambda*(loss_target+loss_source)
        # balanced_loss = BCE_mr + torch.mul(KLD_mr, kldlamda) + BCE_ct + torch.mul(KLD_ct,kldlamda) + 0.01*torch.mul(distance_loss, dislamda)\
        #                 + 0.001*predlamda * (segloss_output + fusionsegloss_output) + \
        #                 BCE_down2_ct + torch.mul(KLD_down2_ct, kldlamda) + BCE_down2_mr + torch.mul(
        #     KLD_down2_mr, kldlamda) + 0.01*torch.mul(distance_down2_loss, dislamdadown2) + 0.001*predlamda * segdown2loss_output + \
        #                 BCE_down4_ct + torch.mul(KLD_down4_ct, kldlamda) +  BCE_down4_mr + torch.mul(
        #     KLD_down4_mr, kldlamda) + 0.01*torch.mul(distance_down4_loss, dislamdadown4) + 0.001*predlamda * segdown4loss_output + \
        #                 1.0 * (loss_source+loss_target)
        if args.needMI:
            optim_loss.zero_grad()
        optim.zero_grad()
        balanced_loss.backward()
        # networks = [encoder, decoderA, decoderAdown2, decoderAdown4, decoderB, decoderBdown2, decoderBdown4,
        # source_MINE,target_MINE]
        # for item in networks:
        #     nn.utils.clip_grad_norm_(item.parameters(), max_norm=4, norm_type=2)
        if args.needMI:
            optim_loss.step()
        optim.step()
        if i % 20 == 0:
            print(epoch,i,lr)
            print("total_loss:",balanced_loss)
            print("BCE_mr",BCE_mr)
            print("torch.mul(KLD_mr,kldlamda)",0.01*torch.mul(distance_loss, dislamda))
            print("BCE_ct",BCE_ct)
            print("torch.mul(KLD_ct,kldlamda)",torch.mul(KLD_ct,kldlamda))
            print("torch.mul(distance_loss,dislamda)",0.01*torch.mul(distance_loss, dislamda))
            print("predlamda*(segloss_output+fusionsegloss_output)",0.001*predlamda * (segloss_output + fusionsegloss_output))
            print("10.0*BCE_down2_ct",BCE_down2_ct)
            print("torch.mul(KLD_down2_ct, kldlamda)",torch.mul(KLD_down2_ct, kldlamda))
            print("10.0*BCE_down2_mr",BCE_down2_mr)
            print("torch.mul(KLD_down2_mr, kldlamda)",torch.mul(KLD_down2_mr, kldlamda))
            print("torch.mul(distance_down2_loss, dislamdadown2)",0.01*torch.mul(distance_down2_loss, dislamdadown2))
            print("predlamda * segdown2loss_output",0.001*predlamda * segdown2loss_output)
            print("10.0*BCE_down4_ct",BCE_down4_ct)
            print("torch.mul(KLD_down4_ct, kldlamda)",torch.mul(KLD_down4_ct, kldlamda))
            print("10.0*BCE_down4_mr + torch.mul(KLD_down4_mr, kldlamda)",BCE_down4_mr + torch.mul(KLD_down4_mr, kldlamda))
            print("torch.mul(distance_down4_loss, dislamdadown4)",0.01*torch.mul(distance_down4_loss, dislamdadown4))
            print("predlamda * segdown4loss_output",0.001*predlamda * segdown4loss_output)
            if args.needMI:
                print("1.0*(loss1+loss2+loss3)",100.0*(loss_target+loss_source))
            # print('epoch %d , %d th iter; seglr,ADA_totalloss,segloss,distance_loss1,distance_loss2: %.6f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f'\
            #       % (epoch, i,lr, balanced_loss.item(),BCE_mr.item(),KLD_mr.item(),BCE_ct.item(),KLD_ct.item(),fusionsegloss_output.item(),segloss_output.item(),segdown2loss_output.item(),segdown4loss_output.item(),distance_loss.item(),distance_down2_loss.item(),distance_down4_loss.item(),loss1.item(),loss2.item(),loss3.item()))

        i=i+1

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    cudnn.benchmark = True

    PredLamda = args.predLambda
    DisLamda = args.disLambda
    DisLamdaDown2 = args.disLambda2
    DisLamdaDown4= args.disLambda3
    InfoLamda=0.0

    sample_nums = [args.target_num]

    for sample_num in sample_nums:

        SAVE_DIR = args.save_dir+str(sample_num)
        SAVE_IMG_DIR = args.save_test_dir+str(sample_num)
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        if not os.path.exists(SAVE_IMG_DIR):
            os.mkdir(SAVE_IMG_DIR)

        vaeencoder = VAE()
        vaeencoder = vaeencoder.cuda()

        source_vaedecoder = VAEDecode()
        source_vaedecoder = source_vaedecoder.cuda()

        source_down2_vaedecoder = VAEDecode_down2()
        source_down2_vaedecoder = source_down2_vaedecoder.cuda()

        source_down4_vaedecoder = VAEDecode_down4()
        source_down4_vaedecoder = source_down4_vaedecoder.cuda()

        target_vaedecoder = VAEDecode()
        target_vaedecoder = target_vaedecoder.cuda()

        target_down2_vaedecoder = VAEDecode_down2()
        target_down2_vaedecoder = target_down2_vaedecoder.cuda()

        target_down4_vaedecoder = VAEDecode_down4()
        target_down4_vaedecoder = target_down4_vaedecoder.cuda()

        #Infonet = InfoNet().cuda()

        if args.needMI:
        # MI_estimation
            source_MINE = DeepInfoMax(args.global_weight,args.local_weight,args.prior_weight)
            target_MINE = DeepInfoMax(args.global_weight,args.local_weight,args.prior_weight)
            source_MINE = source_MINE.cuda()
            target_MINE = target_MINE.cuda()
        else:
            source_MINE = 0
            target_MINE = 0

        DistanceNet = Contrastive_loss(args.kernel)  # 64,Num_Feature2,(12,12)
        # DistanceNet2 = nn.DataParallel(DistanceNet2, device_ids=[0,1])
        # torch.nn.utils.clip_grad_norm(loss_mi1.parameters(),max_norm=5, norm_type=2)
        # torch.nn.utils.clip_grad_norm(loss_mi2.parameters(), max_norm=5, norm_type=2)
        # torch.nn.utils.clip_grad_norm(loss_mi3.parameters(), max_norm=5, norm_type=2)


        DA_optim = torch.optim.Adam([{'params': vaeencoder.parameters()},
                                     {'params': source_vaedecoder.parameters()},
                                     {'params': source_down2_vaedecoder.parameters()},
                                     {'params': source_down4_vaedecoder.parameters()},
                                     {'params': target_vaedecoder.parameters()},
                                     {'params': target_down2_vaedecoder.parameters()},
                                     {'params': target_down4_vaedecoder.parameters()},
                                     ], lr=args.lr,
                                    weight_decay=args.decay)
        if args.needMI:
            Loss_optim = torch.optim.Adam([{'params': source_MINE.parameters()},
                                       {'params': target_MINE.parameters()},
                                       ],lr=args.lr,weight_decay=args.decay)
        else:
            Loss_optim = 0

        # scheduler1 = MultiStepLR(DA_optim, milestones=args.milestone, gamma=0.20)
        # scheduler2 = MultiStepLR(Loss_optim,milestones=args.milestone,gamma=0.20)


        SourceData = C0_TrainSet(args.data_dir,args.target_num)
        SourceData_loader = DataLoader(SourceData, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker,
                                       pin_memory=True,drop_last = True)

        TargetData = LGE_TrainSet(args.data_dir,sample_num)
        TargetData_loader = DataLoader(TargetData, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker,
                                       pin_memory=True,drop_last = True)

        vaeencoder.apply(init_weights)
        source_vaedecoder.apply(init_weights)
        source_down2_vaedecoder.apply(init_weights)
        source_down4_vaedecoder.apply(init_weights)
        target_vaedecoder.apply(init_weights)
        target_down2_vaedecoder.apply(init_weights)
        target_down4_vaedecoder.apply(init_weights)
        if args.needMI:
            source_MINE.apply(init_weights)
            target_MINE.apply(init_weights)

        criterion=0
        best_epoch=0
        for epoch in range(args.epoch):
            vaeencoder.train()
            source_vaedecoder.train()
            source_down2_vaedecoder.train()
            source_down4_vaedecoder.train()
            target_vaedecoder.train()
            target_down2_vaedecoder.train()
            target_down4_vaedecoder.train()
            if args.needMI:
                source_MINE.train()
                target_MINE.train()
            ADA_Train(source_MINE,target_MINE,SourceData_loader,TargetData_loader,vaeencoder,source_vaedecoder,source_down2_vaedecoder,source_down4_vaedecoder,target_vaedecoder,target_down2_vaedecoder,target_down4_vaedecoder,1.0,DistanceNet,args.lr,args.kldLambda,PredLamda,DisLamda,DisLamdaDown2,DisLamdaDown4,epoch,DA_optim,Loss_optim, SAVE_DIR)
            # scheduler1.step(epoch)
            # scheduler2.step(epoch)
            vaeencoder.eval()
            criter =SegNet_test_mr(args.test_dir, vaeencoder,0, epoch,args.epoch, SAVE_DIR,SAVE_IMG_DIR)
            if criter > criterion:
                best_epoch = epoch
                criterion=criter
                torch.save(vaeencoder.state_dict(), os.path.join(SAVE_DIR, 'encoder_param.pkl'))
                torch.save(source_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderA_param.pkl'))
                torch.save(source_down2_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderAdown2_param.pkl'))
                torch.save(source_down4_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderAdown4_param.pkl'))
                torch.save(target_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderB_param.pkl'))
                torch.save(target_down2_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderBdown2_param.pkl'))
                torch.save(target_down4_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderBdown4_param.pkl'))
                if args.needMI:
                    torch.save(source_MINE.state_dict(), os.path.join(SAVE_DIR, 'source_MI.pkl'))
                    torch.save(target_MINE.state_dict(), os.path.join(SAVE_DIR, 'target_MI.pkl'))

        print ('\n')
        print ('\n')
        print ('best epoch:%d' % (best_epoch))
        with open("%s/lge_testout_index.txt" % (SAVE_DIR), "a") as f:
            f.writelines(["\n\nbest epoch:%d" % (best_epoch)])


        del vaeencoder, source_vaedecoder, source_down2_vaedecoder, source_down4_vaedecoder, target_vaedecoder, target_down2_vaedecoder, target_down4_vaedecoder,source_MINE,target_MINE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VARDA')
    parser.add_argument('--batch_size',default=12,type=int,help='batch_size')#
    parser.add_argument('--epoch',default=30,type=int,help='epoch')#
    parser.add_argument('--predLambda',default=1e3,type=float,help='segmentation loss')#
    parser.add_argument('--disLambda',default=1e-3,type=float,help='domain distance loss 256 40 40')#
    parser.add_argument('--disLambda2',default=1e-3,type=float,help='domain distance loss 128 80 80')#
    parser.add_argument('--disLambda3',default=1e-4,type=float,help='domain distance loss 64 160 160')#
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
    parser.add_argument('--global_weight',default=0.5,type=float,help='global MI')#
    parser.add_argument('--local_weight',default=1.0,type=float,help='local MI')#
    parser.add_argument('--prior_weight',default=0.1,type=float,help='prior_MI')#
    parser.add_argument('--needMI',default=True,type=bool,help='whether need MI') #
    parser.add_argument('--gpu',default='0',type=str,help='gpu_number') #
    parser.add_argument('--num_worker',default=10,type=int) #
    parser.add_argument('--miLambda',default=100,type=float,help='mutual information loss')
    parser.add_argument("--milestone", type=int, default=[5, 10, 15], help="When to decay learning rate")

    args = parser.parse_args()
    main(args)



