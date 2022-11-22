from torch.utils.data import Dataset
#方便进行文件操作
import os
#一般有三种打开图片的方式：
#1. img = cv2.imread(path)    2.  img = skimage.io.imread(path). 前两者的返回对象均为np.adarray()类型。
#我们这里先尝试一波第三种，即返回img对象。
import math
import numpy as np
import torch
from imageio import imread
import random
import pickle
import glob
import matplotlib.pyplot as plt

# #输入的low_resolution的图片：(采用bicubic下采样),这里只有文件夹形式，根据参数输入进来的scale来进行后续图片的读入。
# input_dir='./DIV2K/DIV2K_train_LR_bicubic'
# #高清（label）high_resolution的图片位置：
# label_dir='./DIV2K/DIV2K_train_HR'


class MyDataset(Dataset):
    def __init__(self,args,train=True) -> None:
        self.args=args
        self.scale=self.args.scale.split('+')
        self.train=train
        if self.train:
            self.dir_hr='./DIV2K/DIV2K_train_HR'
            self.dir_lr='./DIV2K/DIV2K_train_LR_bicubic'
            self.input_dir=args.input_dir
            self.label_dir=args.label_dir
        else:
            self.dir_hr='./DIV2K/DIV2K_test_HR'
            self.dir_lr='./DIV2K/DIV2K_test_LR'
            self.input_dir=args.input_dir_test
            self.label_dir=args.label_dir_test
            #这里面的input_files的图片名称为0001x2.png   但output_files的图片名称为 0001.png
        list_hr,list_lr=self.scan()
        self.apath='./DIV2K'
        path_bin='./DIV2K/bin'
        if not (os.path.isfile(os.path.join(os.getcwd(),self.label_dir,'0001.pt'))|os.path.isfile(os.path.join(os.getcwd(),self.label_dir,'0801.pt'))):
            
            os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    #这里是在bin里面生成一样的文件夹。
                    exist_ok=True
                )
            for s in self.scale:
                    os.makedirs(
                        os.path.join(
                            self.dir_lr.replace(self.apath, path_bin),
                            'X{}'.format(s)
                        ),
                        exist_ok=True
                    )
            self.images_hr, self.images_lr = [], [[] for _ in self.scale]

            print('making binary:')
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace('.png', '.pt')
                self.images_hr.append(b)
                #这里的b是写进去的文件，而h是图片
                with open(b, 'wb') as _f:
                    pickle.dump(imread(h), _f)
            print('Making lr binary')
            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace('.png', '.pt')
                    self.images_lr[i].append(b)
                    with open(b, 'wb') as _f:
                        pickle.dump(imread(l), _f)
        if self.train:
            self.input_files=os.listdir(os.path.join(self.input_dir,'X{}'.format(self.scale[0])))
            self.label_files=os.listdir(self.label_dir)
        else:
            self.input_files=os.listdir(os.path.join(self.input_dir,'X{}'.format(self.scale[0])))
            self.label_files=os.listdir(self.label_dir)
    def np2Tensor(self,img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            #这里是将numpy数组转为tensor形式。
        tensor = torch.from_numpy(np_transpose).float()
        return tensor
    def get_patch(self,lr,hr):
        if self.train:
            #假设patch_size=96
            patch_size=self.args.patch_size
            #scale=2
            scale=int(self.args.scale)
            #先对lr进行操作，lh为lr图像的高度，lw为lw图像的宽度
            lh,lw=lr.shape[:2]
            #48
            lp=patch_size//scale
            #96
            tp=patch_size
            lx=random.randrange(0,lw-lp)
            ly=random.randrange(0,lh-lp)
            tx=scale*lx
            ty=scale*ly
            lr=self.np2Tensor(lr[ly:ly+lp,lx:lx+lp,:])
            hr=self.np2Tensor(hr[ty:ty+tp,tx:tx+tp,:])
        else:
            lr=self.np2Tensor(lr)
            hr=self.np2Tensor(hr)
        return lr,hr
    #这里的__len__作用是可以使得重复甄选。
    def __len__(self) -> int:
        #repeat=20
        if self.train:
            return len(self.input_files)*20
        else:
            return len(self.input_files)
    def __getitem__(self, index: int):
        index=index % len(self.input_files)
        if not self.train:
            input_path=os.path.join(os.getcwd(),"DIV2K/bin/DIV2K_test_LR/X{}".format(self.scale[0]),self.input_files[index])
            label_path=os.path.join(os.getcwd(),"DIV2K/bin/DIV2K_test_HR",self.label_files[index])
        else:
            input_path=os.path.join(os.getcwd(),"DIV2K/bin/DIV2K_train_LR_bicubic/X{}".format(self.scale[0]),self.input_files[index])
            label_path=os.path.join(os.getcwd(),"DIV2K/bin/DIV2K_train_HR",self.label_files[index])
        #这里的格式均为numpy格式
        
        with open(input_path,'rb') as f:
            input_image=pickle.load(f)
        with open(label_path,'rb') as f:
            label_image=pickle.load(f)
        lr,hr=self.get_patch(input_image,label_image)
        # print(lr.size(),hr.size(),'\n')
        return (lr,hr)
    def scan(self):
        # print('运行嘞srdata中的_scan函数')
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + 'png'))
            #self.dir_hr=../dataset/DIV2K/HR.png
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            #'001'  '.png'
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, '.png'
                    )
                ))

        return names_hr, names_lr
    
def cal_psnr(sr,hr,scale):
    
    diff=(hr-sr)/255
    scale=int(scale)
    gray_coeffs=[65.738, 129.057, 25.064]
    convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
    diff = diff.mul(convert).sum(dim=1)
    valid=diff[...,scale:-scale,scale:-scale]
    mse=valid.pow(2).mean()
    
    # mse=diff.pow(2).mean()
    return -10*math.log10(mse)
    # return 10*math.log10(255**2/mse)
#定义一个有关于numpy数组转为tensor的函数
def plot_psnr(psnr,epoch):
    axis = np.linspace(1, epoch+1, epoch+1)
    label='PSNR on DIV2K of first {} epochs'.format(epoch+1)
    fig=plt.figure()
    plt.plot(axis,psnr,'r')
    plt.title(label)
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig('psnr_{}.pdf'.format(epoch+1))
    plt.close(fig)
    
