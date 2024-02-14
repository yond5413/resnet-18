## make neural network
import torch
from torch import nn
#import torch.nn.functional as F
######
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
#################################
import time
## build whatever neural network like specified in prompt
## kind of weird it is like an obect oriented neural net
## using block structure to do it 
#############################
#input block
## have to relu and batch norm after every convolution
''' 
input->[64]
1st block: []
2nd block: [64->128],[128,128] [input,output]
3rd block: [128->256],[256,256]
4th block: [256->,512],[512,512]
'''
## data loading + training time practically the same
class ConvBlock(nn.Module):
    ### params
    def __init__(self, in_channel = 3, out_channel = 64, kernel_size = 3, stride=1, padding=1):
        super.__init__()
        self.conv = nn.conv2d(in_channel,out_channel,kernel_size, stride,padding)
        self.batchNorm = nn.batchnorm2d(out_channel)        
        ## batch normalization is done after every convolution
        ### and prior to each activation 
    def forward(self,x):
        out = self.conv(x)
        out = nn.ReLU(out)
        return self.batchNorm(out)
        #return self.conv(x)#self.batchNorm(self.conv(x))
#############################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding):
        super.__init__()
        ###########
        self.conv1 = ConvBlock(in_channels, out_channels,kernel_size, stride, padding)
        self.conv2 = ConvBlock(out_channels, out_channels,kernel_size, stride, padding)
        self.relu  = nn.ReLU()
        self.batchNorm  = nn.BatchNorm2d()
        #2 convolution blocks#
        ###########
    def forward(self,x):
        identity = x
        f = self.conv1(x)
        h = f+x
        ret =self.batchNorm(self.relu(h))
        return ret
        #f = NotImplemented
        ##self.conv1 then relu?
        #h = f+x
        ## h = f + x relu'ed?
        #return h
##############################
class ResNet(nn.Module):
    def __init__(self):
        super.__init__()
        ### 2 basicblocks per sub group
        ###
        ''' 
        input->[64]
        1st block: [64->64],[64,]
        2nd block: [64->128],[128,128] [input,output]
        3rd block: [128->256],[256,256]
        4th block: [256->,512],[512,512]
        '''
        self.input_layer = ConvBlock()
        ### has default parmas ^
        self.block1 = ResidualBlock(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.block2 = ResidualBlock(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.block3 = ResidualBlock(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1)
        self.block4 = ResidualBlock(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1)
        self.output_layer = NotImplemented
    def forward(self):
        pass
def dataLoad():
    pass
def train():
    pass
def test():
    pass
if __name__ == "__main__":
    print("hello world")