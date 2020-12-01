'''A simple code for training H-score'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision

#import model_utils
import matplotlib
import argparse
import time
import Image
from hscore import neg_hscore

from resnet import *

import numpy as np

import csv

import pickle
import os

import hub

n_CLASS=20
part='.\\resnet18\seg'
print(part)
class aceModel_g(nn.Module):
    def __init__(self):
        super(aceModel_g, self).__init__()
        self.fc1 = nn.Linear(n_CLASS, 1024)

    def forward(self, y):
        g = self.fc1(y)
        return g

class f_attention1(nn.Module):
    def __init__(self):
        super(f_attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
        self.softmax=nn.Softmax(dim=1)

    def get_feature(self, x):
        n_batch, C, H, W = x.size()
        out = self.conv1(x)
        out = self.softmax(out)
        return out

    def forward(self, x):
        n_batch, C, H, W = x.size()
        out = self.conv1(x)
        out = self.softmax(out)
        return out

class f_attention(nn.Module):
    def __init__(self,model):
        super(f_attention, self).__init__()
        self.layer=nn.Sequential(*list(model.children())[:-2])
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
        self.softmax=nn.Softmax(dim=1)
        self.avrpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def threshold(self,x):
        n_batch, C, H, W = x.size()
        alpha=1./(H*W)

        Z=torch.zeros(n_batch,1,7,7)
        ZT= torch.zeros(n_batch, 1, 7, 7)
        for b in range(n_batch):
            batch=x[b]
            for i in range(W):
                for j in range(H):
                    if batch[0][i][j]< alpha:
                       ZT[b][0][i][j]=1
                    else: Z[b][0][i][j]=1
        return Z, ZT

    def mat_cat(self,x_r, x_s):
        x_r=x_r.to(DEVICE)
        x_s=x_s.to(DEVICE)


        Z, ZT = self.threshold(x_s)

        Z=Z.to(DEVICE)
        ZT=ZT.to(DEVICE)

        # print(x_r.shape)
        # print(x_s.shape)
        # print(Z.shape)
        # print(ZT.shape)
        attend = x_r * Z
        attend = attend.to(DEVICE)
        unattend = x_r * ZT
        unattend = unattend.to(DEVICE)
        feature = torch.cat((attend, unattend), 1)
        feature = feature.to(DEVICE)
        return feature

    def get_feature(self, x):
        x_r = self.layer(x)
        out = self.conv1(x_r)
        x_s = self.softmax(out)
        out = self.mat_cat(x_r,x_s)
        out = self.avrpool(out)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        x_r = self.layer(x)
        out = self.conv1(x_r)
        x_s = self.softmax(out)

        return x_s











def test_Hscore(test_loader, model):
    model[0].eval()
    correct = 0
    count = 0
    accData = [[]]

    for i, (data, label) in enumerate(test_loader):
        label_1hot = torch.zeros(len(label), n_CLASS).scatter_(1, label.resize(len(label),1), 1).to(DEVICE)
        data, label = data.to(DEVICE), label.to(DEVICE)
        #f = model[0].get_feature(data)
        f = my_forward(model[0], data)
        f_zeromean = f - torch.mean(f,0)
        #g = model[1](label_1hot)
        g_val = model[1](torch.eye(n_CLASS).to(DEVICE))

        #loss = neg_hscore(f, g_val)
        #lossData.append([i, -(loss.cpu().data.numpy())])

        g_zeromean = g_val - torch.mean(g_val,0)
        logits = torch.mm(f_zeromean, torch.t(g_zeromean))

        _, predicted = torch.max(logits.data, 1)
        count += label.size(0)
        # 记录batch的acc
        accData.append([100*(label.data == predicted).float().sum().cpu().numpy()/32.0])
        correct += (predicted == label.data).float().sum()
    #data_write_csv(part+'_batch_test_acc' + '.csv', accData)

    print('Test Accuracy: %.2f' % (100 * float(correct) / count))
    return  (100 * float(correct) / count)

def my_forward(model, x):
    mo = nn.Sequential(*list(model.children())[:-2])
    feature = mo(x)
    #feature = feature.view(x.size(0), -1)
    attention = f_attention()
    out=attention(feature)
    Z, ZT=threshold(out)
    attend=feature*Z
    unattend=feature*ZT
    feature=torch.cat((attend, unattend),1)
    return feature



def train_Hscore(train_loader, model, optimizer, epoch, file_dir_path):
    model[0].train()
    count=0
    train_loss=0
    train_correct=0
    total_features = []
    total_labels = []

    lossData=[[]]
    accData=[[]]

    for i, (data, label) in enumerate(train_loader):
        label_1hot = torch.zeros(len(label), n_CLASS).scatter_(1, label.resize(len(label),1), 1).to(DEVICE)
        data, label = data.to(DEVICE), label.to(DEVICE)

        model_f=f_attention(model[0]).to(DEVICE)
        #f = model[0].get_feature(data)
        #f=my_forward(model[0],data)
        f=model_f.get_feature(data)

        g = model[1](label_1hot)
        #g_val = model[1](torch.eye(n_CLASS).to(DEVICE))
        loss=neg_hscore(f, g)
        # 记录batch loss
        lossData.append([i, -(loss.cpu().data.numpy())])

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        f_zeromean = f - torch.mean(f,0)
        g_val = model[1](torch.eye(n_CLASS).to(DEVICE))
        g_zeromean = g_val - torch.mean(g_val,0)
        logits = torch.mm(f_zeromean, torch.t(g_zeromean))

        _, predicted = torch.max(logits.data, 1)

        total_labels.append(label.data.detach().cpu().numpy())
        total_features.append(f.data.detach().cpu().numpy())

        count += label.size(0)
        train_loss += loss
        train_correct += (label.data == predicted).float().sum()
        # 记录 batch acc
        accData.append([100*(label.data == predicted).float().sum().cpu().numpy()/32.0])
        if epoch%5==0 :
            torch.save(model[0].state_dict(), '{}/Hscore_F_{}epochs.pkl'.format(file_dir_path,epoch))
            torch.save(model[1].state_dict(), '{}/Hscore_G_{}epochs.pkl'.format(file_dir_path,epoch))


    #data_write_csv(part+'_batch_train_loss'+'.csv',lossData)
    #data_write_csv(part+'_batch_train_acc'+'.csv',accData)
    #accAll.append(100*(train_correct/count).cpu().numpy())

    print("Epoch {}".format(epoch))
    print("Train loss: {}, Train acc:{:.2f}%".format(train_loss/len(train_loader), 100*train_correct/count))
    return train_loss/len(train_loader), 100*train_correct/count

# 新添
def data_write_csv(filename,datas):
    with open(filename, "a", encoding="utf-8", newline='') as w:
        writer = csv.writer(w)
        for data in datas:
            writer.writerow(data)


def train_model_with_Hscore(train_loader,test_loader,model,args,file_dir_path):
    # make mm_means
    model_f = model
    model_g = aceModel_g().to(DEVICE)

    #optimizer = optim.Adam((list(model_f.parameters())+list(model_g.parameters())),lr=args.lr)
    optimizer = optim.SGD((list(model_f.parameters())+list(model_g.parameters())), lr=args.lr, momentum=0.9,weight_decay=2e-5)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[35,45] if args.dataset == 'cifar10' else [40]) 

    loss_list=[]
    acc_list=[]

    for epoch in range(args.nb_epochs):
        scheduler.step()
        # print optimizer4nn.param_groups[0]['lr']
        start_time = time.clock()
        loss, acc=train_Hscore(train_loader, [model_f,model_g], optimizer, epoch + 1, file_dir_path)

        # 记录 loss和acc
        loss_list.append([loss.cpu().detach().numpy()])
        acc_list.append([acc.cpu().detach().numpy()])

        data_write_csv(part+'_epoch_train_loss' + '.csv', loss_list)
        data_write_csv(part + '_epoch_train_acc' +'.csv', acc_list)

        end_time = time.clock()
        print("Training Time:{:.2f}".format(end_time - start_time))
        #=test_Hscore(test_loader, [model_f,model_g])
        #loss_list.append([loss.cpu() .detach().numpy()])

        # 记录test的acc
        acc_list.append([acc])
        #data_write_csv(trainlossfile +'_test_loss_all' + '.csv', loss_list)

        #data_write_csv(part + '_epoch_test_acc' +'.csv', acc_list)


def test_CE(test_loader, model):
    model.eval()
    correct = 0
    count = 0

    accData = [[]]
    for i, (data, label) in enumerate(test_loader):
        data, label = data.to(DEVICE), label.to(DEVICE)

        logits = model(data)
        _, predicted = torch.max(logits.data, 1)
        count += label.size(0)
        correct += (predicted == label.data).float().sum()
        #print(data,logits,predicted,correct)
        # 记录batch的acc
        accData.append([i,100 * (label.data == predicted).float().sum().cpu().numpy() / 32.0])
    data_write_csv(part + '_batch_CE_test_acc' + '.csv', accData)
    print('Test Accuracy: {:.2f}%'.format(100. * correct / count))


def train_CE(train_loader, model, criterion, optimizer, epoch, file_dir_path):
    model.train()
    count=0
    train_loss=0
    train_correct=0
    for i, (data, label) in enumerate(train_loader):
        data, label = data.to(DEVICE), label.to(DEVICE)
        #label_1hot = torch.zeros(len(label), n_CLASS).scatter_(1, label.resize(len(label), 1), 1).to(DEVICE)

        lossData = [[]]
        accData = [[]]

        logits = model(data)
        loss = criterion[0](logits,label)

        # 记录batch loss
        lossData.append([i, loss.cpu().data.numpy()])

        optimizer[0].zero_grad()
        loss.backward()
        optimizer[0].step()

        _, predicted = torch.max(logits.data, 1)

        count += label.size(0)
        train_loss += loss
        train_correct += (label.data == predicted).float().sum()

        # 记录 batch acc
        accData.append([i, 100 * (label.data == predicted).float().sum().cpu().numpy() / 32.0])

        if epoch % 5 == 0:
            torch.save(model.state_dict(), '{}/crossentropy_{}epochs.pkl'.format(file_dir_path,epoch))

    data_write_csv(part + '_batch_CE_train_loss' + '.csv', lossData)
    data_write_csv(part + '_batch_CE_train_acc' + '.csv', accData)

    print("Epoch {}".format(epoch))
    print("Train loss: {}, Train acc:{:.2f}%".format(train_loss/len(train_loader), 100*train_correct/count))

def train_model_with_softmax(train_loader,test_loader,model,args,file_dir_path):
    # NLLLoss
    nllloss = nn.CrossEntropyLoss().to(DEVICE)
    criterion = [nllloss]

    # optimzer4nn
    #optimizer4nn = optim.Adam(model.parameters(), lr=args.lr)    
    optimizer4nn = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=2e-5)

    #optimizer4nn = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer4nn, [35,45] if args.dataset == 'cifar10' else [40])

    for epoch in range(args.nb_epochs):
        scheduler.step()
        start_time = time.clock()
        train_CE(train_loader, model, criterion, [optimizer4nn], epoch + 1,file_dir_path)
        end_time = time.clock()
        print("Training Time:{:.2f}".format(end_time - start_time))
        test_CE(test_loader, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--method",type=str, default='CE', choices=['CE','MM100','Hscore','centerloss','LGM','LMCL','contrastive','triplet','Hscore&CE'], help="the number of training epochs")
    parser.add_argument("-a","--arch",type=str, default='resnet18', help="the model architecture")
    parser.add_argument("-n","--nb_epochs",type=int, default=100, help="the number of training epochs")
    parser.add_argument("-b","--batch_size",type=int, default=32, help="the batch size")
    parser.add_argument("-f","--f_dim",type=int, default=512, help="the last feature dim")
    parser.add_argument("-l","--lr",type=float, default=0.001, help="the learning rate")
    parser.add_argument("-d","--da"
                             "taset",type=str, default='image', choices=['mnist','cifar10','image'], help="the dataset")
    args = parser.parse_args()

    temp_resnet=torchvision.models.resnet18(pretrained=True)
    channel_in = temp_resnet.fc.in_features
    print(temp_resnet.fc)
    # print(temp_resnet.fc.out_features)

    # temp_resnet.fc = nn.Sequential(
    #     nn.Linear(channel_in, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.4),
    #     nn.Linear(256, n_CLASS),
    #     nn.LogSoftmax(dim=1)
    # )
    temp_resnet.fc = nn.Sequential(
        nn.Linear(channel_in, 20))

    # for param in temp_resnet.parameters():
    #     param.requires_grad = False
    #
    # for param in temp_resnet.fc.parameters():
    #     param.requires_grad = True

    #print(temp_resnet)


    #set configuration
    cifar_model_list={'resnet18': temp_resnet }
    #cifar_model_list = {'resnet18': ResNet18()}
    mnist_model_list={'resnet18':ResNet18_mnist()}
    USE_DEFAULT_EPOCHS=False
    DEVICE="cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir='ckpt0826'
    file_dir_path = os.path.join(os.getcwd(),ckpt_dir,'{}_{}_SGD-lr{}_batch{}_fdim{}'.format(args.dataset,args.arch,args.lr,args.batch_size,args.f_dim))
    if not os.path.exists(file_dir_path):os.makedirs(file_dir_path)

    # build dataset

    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        #siamese_train_loader = torch.utils.data.DataLoader(SiameseDataset(train_dataset), batch_size=args.batch_size, shuffle=True)
        #triplet_train_loader = torch.utils.data.DataLoader(TripletDataset(train_dataset), batch_size=args.batch_size, shuffle=True)


        if USE_DEFAULT_EPOCHS: args.nb_epochs = 50

    elif args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root='data', train=True, download=True,transform=transform_train)
        test_dataset = datasets.MNIST(root='data', train=False, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,shuffle=True)
        if USE_DEFAULT_EPOCHS: args.nb_epochs = 40

    elif args.dataset == 'image':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

        # csv_file='./mark.csv'
        # root_dir='./seg/'

        # csv_file='./mark_.csv'
        # root_dir='./seg_single/'

        #csv_file = './mark_.csv'
        #root_dir = './jpeg_img/'


        #root_dir='./seg_single/'
        root_dir='./seg_single/' if part=='seg' else './jpeg_img/'



        train_csv_file='./train_mark.csv'
        test_csv_file='./test_mark.csv'


        train_dataset = Image.ImageDataset(train_csv_file, root_dir, transform=transform_train)
        test_dataset = Image.ImageDataset(test_csv_file, root_dir, transform=transform_train)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


        if USE_DEFAULT_EPOCHS: args.nb_epochs = 40
    # build model
    model = mnist_model_list[args.arch].to(
        DEVICE) if args.dataset == 'mnist' else cifar_model_list[args.arch].to(DEVICE)
    #model = cifar_model_list[args.arch].to(DEVICE) if args.dataset == 'cifar10' else mnist_model_list[args.arch].to(DEVICE)
    #model.load_state_dict(torch.load('{}/{}_{}epochs.pkl'.format(file_dir_path,'crossentropy',args.nb_epochs)))

    # train with specific method
    if args.method == 'CE':
        train_model_with_softmax(train_loader,test_loader,model,args,file_dir_path)
    elif args.method == 'Hscore':
        train_model_with_Hscore(train_loader,test_loader,model,args,file_dir_path)
