import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from args import args
from dataset_prepare import MyDataset,cal_psnr
from torch.utils.data.dataloader import DataLoader
import torchvision
#定义一个基本的卷积函数：（方便后续函数使用）
def conv(in_channels,out_channels,kernel_size):
    #stride默认等于1
    return nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=(kernel_size//2))

#定义EDSR的类
class EDSR(nn.Module):
    #需要继承nn.Module类，并实现forward方法，只要在nn.Module的子类中定义forward方法，backward函数就会被自动实现
    #所有的输入格式为N*C*H*W  (mini-batch)
    def __init__(self,args) -> None:
        #使用super函数调用nn.Module中的__init()__函数,子类函数必须调用父类的构造函数
        super(EDSR,self).__init__()
        #去除args参数中的输入通道，RGB一般是3
        input_channels=args.n_color
        output_channels=input_channels
        #定义残差快的数量，原文应该是等于32：
        n_resblocks=args.n_resblocks
        #接着是卷积核的数量（通道数），EDSR默认是256：
        n_filters=args.n_filters
        kernel_size=args.kernel_size
        #设置scale factor:
        scale=int(args.scale)
        #设置残差系数
        res_scale=args.rescale
        #输入数据(C*H*W)3*48*48
        #先减去图片的均值
        self.sub_mean = MeanShift(255)
        #再加上图片的均值操作
        self.add_mean = MeanShift(255,sign=1)
        head=[conv(input_channels,n_filters,kernel_size)]
        #输出数据为256*48*48
        #定义残差块以及一层卷积层
        body=[Resblocks(n_filters,kernel_size,res_scale=res_scale) for _ in range(n_resblocks)]
        body.append(conv(n_filters,n_filters,kernel_size))
        #接着定义重建模块的上采样层：
        tail=[Upsampler(n_fileters=n_filters,kernel_size=kernel_size,scale=scale)]
        #output size(C*H*W)=256*48(x scale)*48(x scale)
        tail.append(conv(n_filters,output_channels,kernel_size))
        self.head=nn.Sequential(*head)
        self.body=nn.Sequential(*body)
        self.tail=nn.Sequential(*tail)
    def forward(self,x):
        #此时未考虑减去均值。
        x = self.sub_mean(x)
        x=self.head(x)
        res=self.body(x)
        x=res+x
        x=self.tail(x)
        x = self.add_mean(x)
        return x
        
        
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        #.view(-1，6)函数，-1说明动态调整第一个维度。
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        #网络不对该层计算梯度。
        for p in self.parameters():
            p.requires_grad = False        
        
class Resblocks(nn.Module):
    #包汉两个conv，一个relu和一个multi,每一层的通道数是一样的，并且卷积核的大小也是一样的
    def __init__(self,n_fileters,kernel_size,res_scale=1) -> None:
        super().__init__()
        self.body=nn.Sequential(
            conv(n_fileters,n_fileters,kernel_size),
            nn.ReLU(inplace=True),#这里使得inplace等于True会修改输入进来的值，不会另外开一个地址。
            conv(n_fileters,n_fileters,kernel_size),
        )
        self.res_scale=res_scale
    def forward(self, x):
        res=self.body(x).mul(self.res_scale)
        res=res+x
        return res

#再定义一个上采样模块：
class Upsampler(nn.Sequential):
    #传入一个方大参数scale
    def __init__(self,n_fileters,kernel_size,scale):
        
        #s首先判断scale=2 or scale =3
        m=[]
        if (scale==2) | (scale==3):
            m.append(conv(n_fileters,scale**2*n_fileters,kernel_size)),
            m.append(nn.PixelShuffle(scale))
        else:
            for _ in range(math.log(scale,2)):
                m.append(conv(n_fileters,4*n_fileters,kernel_size))
                m.append(nn.PixelShuffle(2))
        super(Upsampler,self).__init__(*m)
# model=EDSR(args=args)
# data_trainer=MyDataset(args,train=True)
# train_dataloader = DataLoader(dataset=data_trainer,
#                                     batch_size=args.batch_size,
#                                     shuffle=True,
#                                     num_workers=args.num_workers)
# loss_function=nn.L1Loss()
# #这里必须要注意的是，每个输入图像的尺寸都是不一样的。
# for data in train_dataloader:
#     input, label=data
#     print(input.size(),label.size())
#     output=model(input)
#     loss=loss_function(output,label)
#     print(loss)
        
                
        
