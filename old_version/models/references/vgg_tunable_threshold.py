import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math


cfg = {
    'VGG4' : [64, 'A', 128, 'A'],
    'VGG6' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}


class Threshold_relu(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    #gamma = 0.3 #Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, threshold, epoch):
        
        ctx.save_for_backward(input, threshold)
        ctx.epoch = epoch
        relu = nn.ReLU()
        out = relu(input-threshold*epoch*1e-3)*(threshold)/(threshold - threshold*epoch*2e-3)
        out[out > (threshold - threshold*epoch*1e-3)] = threshold
        
        #out[input > 0] = float(threshold*scale*0.9)   ####*scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input, threshold    = ctx.saved_tensors
        epoch = ctx.epoch
        grad_input, grad_threshold = grad_output.clone(), grad_output.clone()
        grad_inp, grad_thr = torch.zeros_like(input).cuda(), torch.zeros_like(input).cuda()
        grad_inp[input > 0] = 1.0
        grad_inp[input > threshold] = 0.0
        grad_thr[input > threshold - threshold*epoch*1e-3] = 1.0	
        #grad[input <=- 1] = 0.0
        #grad = input
        #grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad_inp*grad_input, grad_thr*grad_threshold, None
        #return last_spike*grad_input, None


class VGG_TUNABLE_THRESHOLD(nn.Module):
    def __init__(self, vgg_name='VGG16', labels=10, dataset = 'CIFAR10', kernel_size=3, dropout=0.2, default_threshold=1):
        super(VGG_TUNABLE_THRESHOLD, self).__init__()
        
        self.dataset        = dataset
        self.kernel_size    = kernel_size
        self.dropout        = dropout
        self.features       = self._make_layers(cfg[vgg_name])
        self.dropout_conv   = nn.Dropout(self.dropout)
        self.dropout_linear = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.act_func 		= Threshold_relu.apply
        self.threshold 	= {}
        self.perc = {}
        self.dis = []
        if vgg_name == 'VGG6' and dataset!= 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(512*4*4, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )
        elif vgg_name == 'VGG4' and dataset== 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(128*7*7, 1024, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            #nn.Linear(4096, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(1024, labels, bias=False)
                            )
        elif vgg_name!='VGG6' and dataset!='MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(2048, 4096, bias=True),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=True),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=True)
                            )
        elif vgg_name == 'VGG6' and dataset == 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(128*7*7, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )
        elif vgg_name!='VGG6' and dataset =='MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(512*1*1, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )
        for l in range(len(self.features)):
            if isinstance(self.features[l], nn.Conv2d):
                self.threshold['t'+str(l)] 	= nn.Parameter(torch.tensor(default_threshold))
                #percentile['t'+str(l)]  = nn.Parameter(torch.ones(9))
        prev = len(self.features)
        for l in range(len(self.classifier)-1):#-1
            if isinstance(self.classifier[l], nn.Linear):
                self.threshold['t'+str(prev+l)]	= nn.Parameter(torch.tensor(default_threshold))
                #percentile['t'+str(prev+l)]  = nn.Parameter(torch.ones(9))
        self.threshold 	= nn.ParameterDict(self.threshold)
        #self.epoch = nn.parameter(epoch)
        #self.epoch.requires_grad = False
        #self.cur = nn.ParameterDict(percentile)
        #self.cur.requires_grad = False
        
        self._initialize_weights2()

    def percentile(self, t, q):

        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result
    '''
    def relu_threshold (self, x, epoch, threshold):

        res_1 = (x<epoch*1e-3)*0
        res_2 = (epoch*1e-3<=x & x<threshold-epoch*1e-3)*((threshold/(threshold-epoch*2e-3))*x - (threshold*(epoch*1e-3)/(threshold-epoch*2e-3)))
        res_3 = (x>=threshold-epoch*1e-3)*(threshold)
        #out[out<epoch*1e-3] = 0
        #out[out>=epoch*1e-3] = (threshold/(threshold-epoch*2e-3)) - (threshold*(epoch*1e-3)/(threshold-epoch*2e-3))
        #out[out>=threshold-epoch*1e-3] = threshold

        return res_1 + res_2 + res_3
    '''    

    def forward(self, x, epoch):   #######epoch
        out_prev = x
        threshold_out = []
        for l in range(len(self.features)):
            if isinstance(self.features[l], (nn.Conv2d)):
                #self.perc[str(l)] = {}
                #for i in range(1, 100):
                #    self.perc[str(l)][str(i)] = self.percentile(self.features[l](out_prev).view(-1), i)
                    #getattr(self.cur, 't'+str(l))[int(i/10-1)] = (self.percentile(self.features[l](out_prev).view(-1), i))
                out = self.features[l](out_prev) #- getattr(self.threshold, 't'+str(l))*epoch*1e-3
                #out = self.relu_threshold(out, epoch, getattr(self.threshold, 't'+str(l)) )
                #if l==1:
                #    self.dis.extend(torch.flatten(out).tolist())
                out = self.relu(out)
                #out = out.clone()
                #out = self.act_func(out, getattr(self.threshold, 't'+str(l)), epoch)
                #out = self.relu(out)*(getattr(self.threshold, 't'+str(l))/(getattr(self.threshold, 't'+str(l)) - getattr(self.threshold, 't'+str(l))*epoch*2e-3))
                #out[out > getattr(self.threshold, 't'+str(l))-getattr(self.threshold, 't'+str(l))*epoch*1e-3] = getattr(self.threshold, 't'+str(l))
                #out[out>getattr(self.threshold, 't'+str(l))] =  getattr(self.threshold, 't'+str(l))
                out[out>self.threshold['t'+str(l)]] =  self.threshold['t'+str(l)]
                out = self.dropout_conv(out)
                out_prev = out.clone()
                threshold_out.append(self.threshold['t'+str(l)])
                #threshold_out.append(getattr(self.threshold, 't'+str(l)))
            if isinstance(self.features[l], (nn.MaxPool2d)):
                out_prev = self.features[l](out_prev)

        out_prev = out_prev.view(out_prev.size(0), -1)
        prev = len(self.features)

        for l in range(len(self.classifier)-1):
            if isinstance(self.classifier[l], (nn.Linear)):
                #self.perc[str(prev+l)] = {}
                #for i in range(1, 100):
                #    self.perc[str(prev+l)][str(i)] = self.percentile(self.classifier[l](out_prev).view(-1), i)
                #for i in range(10, 100, 10):
                #    getattr(self.cur, 't'+str(prev+l))[int(i/10-1)] = (self.percentile(self.classifier[l](out_prev).view(-1), i))
                out = self.classifier[l](out_prev) #- getattr(self.threshold, 't'+str(prev+l))*epoch*1e-3
                #self.dis.extend(torch.flatten(out).tolist())
                #out = self.relu_threshold(out, epoch, getattr(self.threshold, 't'+str(prev+l)))
                #out = self.act_func(out, getattr(self.threshold, 't'+str(prev+l)), epoch)
                #out = self.relu(out)*(getattr(self.threshold, 't'+str(prev+l))/(getattr(self.threshold, 't'+str(prev+l))-epoch*2e-3))
                #out[out>getattr(self.threshold, 't'+str(prev+l)) - epoch*getattr(self.threshold, 't'+str(prev+l))*1e-3] =  getattr(self.threshold, 't'+str(prev+l))
                out = self.relu(out)
                #out = out.clone()
                #out[out>getattr(self.threshold, 't'+str(prev+l))] =  getattr(self.threshold, 't'+str(prev+l))
                out[out>self.threshold['t'+str(prev+l)]]  =  self.threshold['t'+str(prev+l)]
                out = self.dropout_linear(out)
                out_prev = out.clone()
                #out_prev = out
                threshold_out.append(self.threshold['t'+str(prev+l)])

        out = self.classifier[l+1](out_prev)
        return out, threshold_out

    


        

        #out = self.features(x)
        #out = out.view(out.size(0), -1)
        #out = self.classifier(out)
        #return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_weights2(self):
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def _make_layers(self, cfg):
        layers = []

        if self.dataset == 'MNIST':
            in_channels = 1
        else:
            in_channels = 3
        
        for x in cfg:
            stride = 1
            
            if x == 'A':
                #layers.pop()
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=True)]
                           #nn.ReLU(inplace=True)
                           #]
                #layers += [nn.Dropout(self.dropout)]           
                in_channels = x

        
        return nn.Sequential(*layers)

def test():
    for a in cfg.keys():
        if a=='VGG6' or a=='VGG4':
            continue
        net = VGG(a)
        x = torch.randn(2,3,32,32)
        y = net(x)
        print(y.size())
    # For VGG6 change the linear layer in self. classifier from '512*2*2' to '512*4*4'    
    # net = VGG('VGG6')
    # x = torch.randn(2,3,32,32)
    # y = net(x)
    # print(y.size())
if __name__ == '__main__':
    test()
