import argparse
#argparse的作用：
#1.提供一个和用户交互的内置模块。
#2.可通过help帮助用户查看参数。

parser = argparse.ArgumentParser(description='The args for EDSR')#创建argparse 的解析对象

#添加模型中的参数
parser.add_argument('--n_color',type=int,default=3,
                    help='The number of input channels')
parser.add_argument('--n_resblocks',type=int,default=16,
                    help='The number of residual blocks')
parser.add_argument('--n_filters',type=int,default=64,
                    help='The number of channels of feature maps')
parser.add_argument('--kernel_size',type=int,default=3,
                    help='The size of the convolutional kernel for each convolution')
parser.add_argument('--scale',type=str,default='2',
                    help='The scaling factor for image')
parser.add_argument('--rescale',type=float,default=0.1,
                    help='The residual scaling in the residual blocks')
parser.add_argument('--input_dir',type=str,default='./DIV2K/bin/DIV2K_train_LR_bicubic',
                    help='The input images directory')
parser.add_argument('--label_dir',type=str,default='./DIV2K/bin/DIV2K_train_HR',
                    help='The label images directory')
parser.add_argument('--input_dir_test',type=str,default='./DIV2K/bin/DIV2K_test_LR',
                    help='The input images_test directory')
parser.add_argument('--label_dir_test',type=str,default='./DIV2K/bin/DIV2K_test_HR',
                    help='The label images_test directory')
parser.add_argument('--batch_size',type=int,default=16,
                    help='The number of mini batch size')
parser.add_argument('--patch_size',type=int,default=96,
                    help='The number of input channels')
parser.add_argument('--num_workers',type=int,default=8,
                    help='The number of num_workers')
parser.add_argument('--epoch',type=int,default=400,
                    help='The number of epoch')
args = parser.parse_args()