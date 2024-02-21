## make neural network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
###################################
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
###################################
import os
import argparse
###################################
import time
from tqdm import tqdm
'''
2/19 notes
do dataloader in main function
use tdqm lib for progress bar
'''
## build whatever neural network like specified in prompt
## kind of weird it is like an obect oriented neural net
## using block structure to do it 
####################################
#input block
## have to relu and batch norm after every convolution
''' 
input->[64]
1st block: [64->64],[64,64]
2nd block: [64->128],[128,128] [input,output]
3rd block: [128->256],[256,256]
4th block: [256->,512],[512,512]
'''
## data loading + training time practically the same
'''class ConvBlock(nn.Module):
    ### params
    def __init__(self, in_channel = 3, out_channel = 64, kernel_size = 3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size, stride,padding)
        self.batchNorm = nn.BatchNorm2d(out_channel)  
        self.relu = nn.ReLU(out_channel)      
        ## batch normalization is done after every convolution
        ### and prior to each activation 
    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)#nn.ReLU(out)
        return self.batchNorm(out)
        #return self.conv(x)#self.batchNorm(self.conv(x))'''
#############################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding):
        super().__init__()
        ###########
        #self.conv1 = ConvBlock(in_channels, out_channels,kernel_size, stride, padding)
        #self.conv2 = ConvBlock(out_channels, out_channels,kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size, stride,padding,bias = False)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size, stride=1 ,padding=padding, bias = False)
        self.relu  = nn.ReLU(out_channels)
        self.batchNorm  = nn.BatchNorm2d(out_channels)
        if stride != 1:
            self.down_sample = nn.Conv2d(in_channels,out_channels,kernel_size=(1,1), stride=stride, padding=0, bias = False)
        else:
            self.down_sample =None
        ##TODO add down sampling
        '''
        it will reduce dimension of the input/identity to make things not crash 
        '''
        #2 convolution blocks#
        ###########
       
    def forward(self,x):
        identity = x
        out1 = self.conv1(x)
        f = self.relu(self.batchNorm(out1))
        #################
        f = self.conv2(f)
        #################
        if self.down_sample:    
            identity = self.down_sample(identity)
            ## should I apply relu and batch-norm here?
        #print(f"size of tensors f: {f.size()}, identity: {identity.size()}, out1: {out1.size()}")
        h = f+identity
        ###
        ret =self.batchNorm(h)#self.batchNorm(self.relu(h))
        ret = self.relu(ret)#self.relu(h)
        return ret
##############################
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        ### 2 basicblocks per sub group
        ###
        ''' 
        input->[64]
        1st block: [64->64],[64,64]
        2nd block: [64->128],[128,128] [input,output]
        3rd block: [128->256],[256,256]
        4th block: [256->,512],[512,512]
        '''
        #(3,3) -> 3x3 
        # stride may only impact the input layer for residuals?
        self.input_layer = nn.Conv2d(in_channels = 3,out_channels=64,kernel_size=(3,3), stride = 1,padding=1)#ConvBlock()
        ### has default parmas ^
        #print("Resnet-18 model init")
        self.block1 =    ResidualBlock(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding=1)
        self.block1_b =  ResidualBlock(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding=1)
        ##############
        self.block2 = ResidualBlock(in_channels=64,out_channels=64,kernel_size=(3,3),stride=2,padding=1)
        self.block2_b = ResidualBlock(in_channels=64,out_channels=128,kernel_size=(3,3),stride=2,padding=1)
        ##############
        self.block3 = ResidualBlock(in_channels=128,out_channels=256,kernel_size=(3,3),stride=2,padding=1)
        self.block3_b = ResidualBlock(in_channels=256,out_channels=256,kernel_size=(3,3),stride=2,padding=1)
        ##############
        self.block4 = ResidualBlock(in_channels=256,out_channels=512,kernel_size=(3,3),stride=2,padding=1)
        self.block4_b = ResidualBlock(in_channels=512,out_channels=512,kernel_size=(3,3),stride=2,padding=1)
        ##############
        self.output_layer = nn.Linear(in_features= 512,out_features=10 )
    def forward(self,x):
        out1 = self.block1(self.input_layer(x))
        out1_b = self.block1_b(out1)
        #TODO
        ### need other block for the subgroups 
        out2 = self.block2(out1_b)
        out2_b = self.block2_b(out2)
        #TODO
        #####################
        out3 = self.block3(out2_b)
        out3_b = self.block3_b(out3)
        #TODO
        out4 = self.block4(out3_b)
        out4_b = self.block4_b(out4)
        #TODO
        #print(f"prior to linear layer: {out4_b.size()}")
        y = out4_b.view(out4_b.size(0),-1) ## flattening
        ### is this expected for outputlayer
        ret = self.output_layer(y)#out4_b)
        return ret
'''
Might have to make other stuff global to compare with reference 
'''
resnet = ResNet()
## Below should be Main
def Main():
    '''
    Random cropping with size 32x32 and padding 4
    Random horizontal flipping with prob 0.5
    Normalize each image's RGB with mean (0.4914,0.4822,0.4465)
    '''
    ### might be able to use reference code 
    pass

def train(epoch):
    print('\nEpoch: %d' % epoch)
    resnet.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(trainloader, desc=f'Epoch {epoch}', leave=False)
    for batch_idx, (inputs, targets) in (enumerate(progress_bar)):#enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        ## didn't use their progress bar
        progress_bar.set_postfix(loss=train_loss / (batch_idx + 1), accuracy=100. * correct / total)
    average_loss = train_loss / len(trainloader)
    accuracy = correct / total
    print(f'Training Loss: {average_loss:.4f}, Accuracy: {100 * accuracy:.2f}%')
def test(epoch):
    global best_acc
    resnet.eval()
    test_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(testloader, desc=f'Testing for Epoch {epoch}', leave=False)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):#enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = resnet(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar.set_postfix(loss=test_loss / (batch_idx + 1), accuracy=100. * correct / total)

    average_loss = test_loss / len(testloader)
    accuracy = correct / total
    print(f'Test Loss: {average_loss:.4f}, Accuracy: {100 * accuracy:.2f}%')
if __name__ == "__main__":
    print("hello world")
    
    ##################################
    '''
    -for argparser need device, cuda, data_path, dataloader workers and optimizers as str
    - optimizers ar SGD,SGD with nesterov, Adagra, Adadelta, and Adam 
    ---> use same default hypter params
    '''
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--device', default='cuda',type = str, help =  "device")
    parser.add_argument('--num_workers',default= 2, type= int, help = "dataloader workers")
    parser.add_argument('--data_path',default="./data", type= str, help = "data path")
    parser.add_argument('--opt', default ='sgd',type = str ,help = "optimzer")
    args = parser.parse_args()

    device = 'cuda' #if torch.cuda.is_available() else 'cpu'
    print(f"device info: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    ##################################
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ##################################
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ##################################
    # data loader 
    trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    ##################################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    for epoch in range(start_epoch, start_epoch+6):
        if epoch == 0:
            print("Warm-up epoch.....")
        train(epoch)
        test(epoch)
        scheduler.step()
#### could take first epoch as a warm-up and do 6 epochs
#### then compare first 5, all 6, and last 5?
#### input performed much better without batch norm

'''
c1
per batch is an iteration of an epoch
- do accuracy per batch an the full run through, per batch
- average accross the epoch  
1,2,3,4,5 plus average across is fine
'''