import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from sam import SAM
import timm

class Detector(nn.Module):

    def __init__(self, module, num_classes=2, rho = 0.05, learning_rate = 1e-5,momentum = 0.9, label_smoothing = 0.1):
        super(Detector, self).__init__()
        if module == 'Swinv2':
            self.net=timm.create_model('swinv2_base_window12to24_192to384_22kft1k', pretrained=True, num_classes = num_classes)#384
        else:
            self.net=EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=num_classes)#380
        self.cel=nn.CrossEntropyLoss(label_smoothing = label_smoothing)
        self.optimizer=SAM(self.parameters(),torch.optim.SGD, lr=learning_rate, momentum=momentum, rho=rho)
        
        

    def forward(self,x):
        x=self.net(x)
        return x
    
    def training_step(self,x,target):
        for i in range(2):
            pred_cls=self(x)
            if i==0:
                pred_first=pred_cls
            loss_cls=self.cel(pred_cls,target)
            loss=loss_cls
            self.optimizer.zero_grad()
            loss.backward()
            if i==0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
        
        return pred_first
