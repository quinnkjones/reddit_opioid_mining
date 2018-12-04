import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np 
import imageio
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

parser = argparse.ArgumentParser()

parser.add_argument('arch')
parser.add_argument('resume')
parser.add_argument('--batch_size',default = 256)

args = parser.parse_args() 

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class Add_one_fc(nn.Module):

    def __init__(self,model,in_classes, out_classes):
        super().__init__() 
        self.other_model = model
        self.fc = nn.Linear(in_classes,out_classes)

    def forward(self,input):
        return self.fc(self.other_model(input))

print("=> using pre-trained model '{}'".format(args.arch))
model = models.__dict__[args.arch](pretrained=True)

model = Add_one_fc(model,1000,2) 

if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
else:
    model = torch.nn.DataParallel(model).cuda()

if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
else:
    quit() 


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


data = './train_val_combined/'
traindir = os.path.join(data, 'train')
valdir = os.path.join(data, 'val')

train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=16, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=16, pin_memory=True)


def eval(loader):
    global normalize
    # switch to evaluate mode
    model.eval()

    if not os.path.isdir('./right'):
        os.makedirs('right')
        os.makedirs('wrong')
    count = 0
    for i, (input, target) in enumerate(val_loader):
        
        t_gpu = target.cuda(async=True)

     
        input_var = torch.autograd.Variable(input)

        target_var = torch.autograd.Variable(t_gpu)

        # compute output
        output = model(input_var)

        output = output.detach().cpu().numpy()

        imgs = torch.transpose(input,1,3)
        imgs = imgs.detach().cpu().numpy()

        output = np.argmax(output,axis=1)
        target = target.detach().cpu().numpy()
        direction = target == output

        for idx,d in enumerate(direction):
            if d:
                imageio.imwrite('./right/{}.png'.format(count),imgs[idx])
            else:
                imageio.imwrite('./wrong/{}.png'.format(count),imgs[idx])
            count += 1    

#eval(val_loader)
eval(train_loader)

