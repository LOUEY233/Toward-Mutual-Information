import torch.nn.init as init
import torch.nn as nn
import torch

def init_conv(conv):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class Spatial_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Spatial_Attention, self).__init__()
        self.activate = nn.Sequential(nn.Conv2d(in_channel, 1,kernel_size = 1),
                                      )

    def forward(self, x):
        actition = self.activate(x)
        out = torch.mul(x, actition)

        return out

class Self_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Self_Attention, self).__init__()
        self.chanel_in = in_channel

        self.f = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_channel, out_channels=in_channel//8, kernel_size=1)
        self.v = nn.Conv2d(in_channels=in_channel//8, out_channels=in_channel, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)  # B * (C//8) * (W * H)
        self_attetion = self_attetion.view(m_batchsize, -1, width, height)  # B * (C//8) * W * H

        self_attetion = self.v(self_attetion)   # B * C * W * H

        out = self.gamma * self_attetion   #############  +x

        return out


class VAE(nn.Module):
    def __init__(self, KERNEL=3,PADDING=1):
        super(VAE, self).__init__()
        self.feat = 16
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convt1=nn.ConvTranspose2d(self.feat*64,self.feat*32,kernel_size=2,stride=2)
        self.convt2=nn.ConvTranspose2d(self.feat*32,self.feat*16,kernel_size=2,stride=2)
        self.convt3=nn.ConvTranspose2d(self.feat*16,self.feat*8,kernel_size=2,stride=2)
        self.convt4=nn.ConvTranspose2d(self.feat*8,self.feat*4,kernel_size=2,stride=2)

        self.conv_seq1 = nn.Sequential( nn.Conv2d(1,self.feat*4,kernel_size=KERNEL,padding=PADDING),
                                        nn.InstanceNorm2d(self.feat*4),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(self.feat*4,self.feat*4,kernel_size=KERNEL,padding=PADDING),
                                        nn.InstanceNorm2d(self.feat*4),
                                        nn.ReLU(inplace=True))
        self.conv_seq2 = nn.Sequential(nn.Conv2d(self.feat*4, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*8),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*8),
                                       nn.ReLU(inplace=True))
        self.conv_seq3 = nn.Sequential(nn.Conv2d(self.feat*8, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*16),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.feat*16, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*16),
                                       nn.ReLU(inplace=True))
        self.conv_seq4 = nn.Sequential(nn.Conv2d(self.feat*16, self.feat*32, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.feat*32, self.feat*32, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*32),
                                       nn.ReLU(inplace=True))
        self.conv_seq5 = nn.Sequential(nn.Conv2d(self.feat*32, self.feat*64, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.feat*64, self.feat*64, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*64),
                                       nn.ReLU(inplace=True))


        self.deconv_seq1 = nn.Sequential(nn.Conv2d(self.feat*64, self.feat*32, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*32),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),
                                       nn.Conv2d(self.feat*32, self.feat*32, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*32),
                                       nn.ReLU(inplace=True))
        self.deconv_seq2 = nn.Sequential(nn.Conv2d(self.feat*32, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*16),
                                       nn.ReLU(inplace=True),
                                         nn.Conv2d(self.feat*16, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*16),
                                         nn.ReLU(inplace=True),
                                        )

        self.down4fc1 = nn.Sequential(Spatial_Attention(self.feat*16),
                                      nn.InstanceNorm2d(self.feat*16),
                                      nn.Tanh())
        self.down4fc2 = nn.Sequential(nn.Conv2d(self.feat*16, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                      nn.InstanceNorm2d(self.feat*16),
                                      nn.Tanh())
        self.segdown4_seq = nn.Sequential(nn.Conv2d(self.feat*16, 4, kernel_size=KERNEL, padding=PADDING),)

        self.deconv_seq3 = nn.Sequential(nn.Conv2d(self.feat*16, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*8),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),
                                       nn.Conv2d(self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*8),
                                       nn.ReLU(inplace=True))

        self.down2fc1 = nn.Sequential(Spatial_Attention(self.feat*8),
                                      nn.InstanceNorm2d(self.feat*8),
                                      nn.Tanh())
        self.down2fc2 = nn.Sequential(nn.Conv2d(self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                      nn.InstanceNorm2d(self.feat*8),
                                      nn.Tanh())
        self.segdown2_seq = nn.Sequential(nn.Conv2d(self.feat*8, 4, kernel_size=KERNEL, padding=PADDING),)

        self.deconv_seq4 = nn.Sequential(nn.Conv2d(self.feat*8, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*4),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),
                                       nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*4),
                                       nn.ReLU(inplace=True),)

        self.fc1 = nn.Sequential( Spatial_Attention(self.feat*4),
                                 nn.InstanceNorm2d(self.feat*4),
                                 nn.Tanh())
        self.fc2 = nn.Sequential(nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                 nn.InstanceNorm2d(self.feat*4),
                                 nn.Tanh())

        self.deconv_seq5 = nn.Sequential(nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*4),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.feat*4, 4, kernel_size=KERNEL, padding=PADDING))
        self.soft = nn.Softmax2d()

        self.upsample2 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.segfusion = nn.Sequential(nn.Conv2d(4*3, 12, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(12),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(4 * 3, 4, kernel_size=KERNEL, padding=PADDING),)


    def reparameterize(self, mu, logvar,gate):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp*gate
        return z

    def bottleneck(self, h,gate):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar,gate)
        return z, mu, logvar

    def bottleneckdown2(self, h,gate):
        mu, logvar = self.down2fc1(h), self.down2fc2(h)
        z = self.reparameterize(mu, logvar,gate)
        return z, mu, logvar

    def bottleneckdown4(self, h,gate):
        mu, logvar = self.down4fc1(h), self.down4fc2(h)
        z = self.reparameterize(mu, logvar,gate)
        return z, mu, logvar

    def encode(self,x,gate):
        out1 = self.conv_seq1(x)     # 1->64
        out2 = self.conv_seq2(self.maxpool(out1))   # 64->128
        out3 = self.conv_seq3(self.maxpool(out2))   # 128->256
        out4 = self.conv_seq4(self.maxpool(out3))   # 256->512
        out5 = self.conv_seq5(self.maxpool(out4))   # 512->1024

        deout1 = self.deconv_seq1(torch.cat((self.convt1(out5),out4),1))   # 1024->512
        deout2 = self.deconv_seq2(torch.cat((self.convt2(deout1),out3),1)) # 512->256
        feat_down4,down4_mu,down4_logvar = self.bottleneckdown4(deout2,gate) # z mu var (256)
        segout_down4 = self.segdown4_seq(feat_down4)  # z(256)->4
        pred_down4 = self.soft(segout_down4)  # 4 -> 4 (softmax)
        deout3 = self.deconv_seq3(torch.cat((self.convt3(feat_down4),self.convt3(deout2)),1)) # (128+128)->128
        feat_down2,down2_mu,down2_logvar = self.bottleneckdown2(deout3,gate)  #z mu var (128)
        segout_down2 = self.segdown2_seq(feat_down2)  # z(128) -> 4
        pred_down2 = self.soft(segout_down2) # 4 -> 4 (softmax)
        deout4 = self.deconv_seq4(torch.cat((self.convt4(feat_down2),self.convt4(deout3)),1)) # (64+64) -> 64
        z, mu, logvar = self.bottleneck(deout4,gate) #z mu var (64)
        return z, mu, logvar,pred_down2,segout_down2,feat_down2,down2_mu,down2_logvar,pred_down4,segout_down4,feat_down4,down4_mu,down4_logvar,out5


    def forward(self, x,gate):
        z, mu, logvar,pred_down2, segout_down2, feat_down2, down2_mu, down2_logvar,pred_down4, segout_down4, feat_down4, down4_mu, down4_logvar,out5 = self.encode(x,gate)
        out = self.deconv_seq5(z)
        pred = self.soft(out)
        fusion_seg = self.segfusion(torch.cat((pred,self.upsample2(pred_down2),self.upsample4(pred_down4)),dim=1)) # 4 + 4 + 4

        return fusion_seg,pred,out,z, mu, logvar,pred_down2, segout_down2, feat_down2, down2_mu, down2_logvar,pred_down4, segout_down4, feat_down4, down4_mu, down4_logvar,out5

class InfoNet(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(InfoNet, self).__init__()
        self.feat = 16

        self.info_seq=nn.Sequential(nn.Linear(self.feat*64*10*10,self.feat*16),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.feat*16,6))
    def forward(self, z):
        z=self.info_seq(z.view(z.size(0),-1))
        return z

class VAEDecode(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(VAEDecode, self).__init__()
        self.feat=16

        self.decoderB=nn.Sequential(
            nn.Conv2d(self.feat*4+4, self.feat*8, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*8, self.feat*4, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*4, self.feat*2, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*2, self.feat*2, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*2, 1, kernel_size=KERNEL, padding=PADDING),
            nn.Sigmoid(),
        )

    def forward(self, z,y):
        z=self.decoderB(torch.cat((z,y),dim=1))
        return z

class VAEDecode_down2(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(VAEDecode_down2, self).__init__()
        self.feat =16

        self.decoderB=nn.Sequential(
            nn.Conv2d(self.feat*8 + 4, self.feat*8, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*8, self.feat*4, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*4, self.feat*2, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*2, self.feat*2, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*2, 1, kernel_size=KERNEL, padding=PADDING),
            nn.Sigmoid(),
        )

    def forward(self, z,y):
        z=self.decoderB(torch.cat((z,y),dim=1))
        return z

class VAEDecode_down4(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(VAEDecode_down4, self).__init__()
        self.feat =16

        self.decoderB=nn.Sequential(
            nn.Conv2d(self.feat*16 + 4, self.feat*8, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*8, self.feat*4, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*4, self.feat*2, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*2, self.feat*2, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*2, 1, kernel_size=KERNEL, padding=PADDING),
            nn.Sigmoid(),
        )

    def forward(self, z,y):
        z=self.decoderB(torch.cat((z,y),dim=1))
        return z

class Discriminator(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(Discriminator, self).__init__()

        self.decoder=nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=2),  # 190
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, stride=2),  # 190
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, dilation=2),  # 190
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2,dilation=2),  # 190
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.linear_seq=nn.Sequential(nn.Linear(32*5*5,256),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(256, 64),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(64, 1),
                                      )

    def forward(self, y):
        out= self.decoder(y)
        out = self.linear_seq(out.view(out.size(0),-1))
        out = out.mean()
        return out
