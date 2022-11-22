#import os，后续有相关的基于系统的操作
import os
import torch 
import torch.nn as nn
#import optim优化器
import torch.optim as optim
#import tqdm，方便训练时命令行查看进度。
from tqdm import tqdm
import torchvision
from dataset_prepare import MyDataset,cal_psnr,plot_psnr
from torch.utils.data.dataloader import DataLoader
#这里导入一个能够动态调整学习率的库函数 scheduler,
import torch.optim.lr_scheduler as lrs
from args import args
from model import EDSR
import numpy as np
import copy
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using {} device.'.format(device))
  
    data_trainer=MyDataset(args,train=True)
    train_dataloader = DataLoader(dataset=data_trainer,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)
    test_trainer=MyDataset(args,train=False)
    test_dataloader=DataLoader(dataset=test_trainer,
                                    batch_size=1)
    #这里打印一下train与test的各个数据集长度
    # print(len(train_dataloader),len(test_dataloader))
    #定义模型
    model=EDSR(args=args).to(device=device)
    #这里定义损失函数，L1损失函数。
    loss_function=nn.L1Loss()
    #初始学习率为1e-4,其余的参数都是采用默认值。
    optimizer=optim.Adam(params=model.parameters(),lr=1e-4)
    #这里需要一个动态调整学习率的库scheduler:
    scheduler=lrs.StepLR(optimizer=optimizer,step_size=200,gamma=0.5)
    best_epoch=0
    best_psnr=0.0
    psnr=[]
    print('Now enter the epoch to train and test:\n')
    for epoch in range(args.epoch):
        #首先进入train模式
        model.train()
        run_loss=0.0
        #是否要重新更新学习率？
        train_bar=tqdm(train_dataloader,total=len(train_dataloader))
        for data in train_bar:
            inputs, labels=data
            #打印LR与HR之间的PSNR
            #print('\n The PSNR is:',cal_psnr(inputs,labels,scale=args.scale))
            #将两者转到GPU上去
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            #这里计算模型预测值与label之间的损失
            loss=loss_function(outputs,labels)
            #清空上一次梯度值
            optimizer.zero_grad()
            #反向传播损失
            loss.backward()
            #接着参数更新，以及学习率的更新
            optimizer.step()
            run_loss+=loss.item()
            train_bar.desc="Train epoch[{}/{}].The current loss is  {:.3f}.".format(epoch+1,args.epoch,run_loss)
        #接着直接进入验证：
        scheduler.step()
        model.eval()
        psnr_avr=[]
        test_bar=tqdm(test_dataloader,total=len(test_dataloader))
        for data in test_bar:
            inputs, labels=data
            inputs=inputs.to(device)
            labels=labels.to(device)
            with torch.no_grad():
                outputs=model(inputs)
            psnr_cur=cal_psnr(labels,outputs,args.scale)
            psnr_avr.append(psnr_cur)
            test_bar.desc="Test epoch[{}/{}].The current PSNR is {:.3f}".format(epoch+1,args.epoch,psnr_cur)
        psnr.append(np.mean(psnr_avr))
        print('The average psnr of epoch{} is {:.2f} '.format(epoch+1,np.mean(psnr_avr)))
        if psnr_cur>best_psnr:
            best_psnr=psnr_cur
            best_epoch=epoch+1
            #再储存一个最优的权重。
            best_weights=copy.deepcopy(model.state_dict())
        if ((epoch+1)%50)==0:
            plot_psnr(psnr,epoch)
    print('The total training and test is finished.\n The best epoch is{} and the best psnr is{}'.format(best_epoch,best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
        
        
if __name__ == '__main__':
    main()
    


        
                                                      
