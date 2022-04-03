



import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, dropout, threshold1, threshold2):
        super().__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.perc = {}
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.residual = nn.Sequential(
        #    nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
        #    nn.ReLU(inplace=True),
        #    self.threshold,
        #    nn.Dropout(dropout),
        #    nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
        #    )
        self.identity = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.identity = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                
            )

    def percentile(self, t, q):

        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    def forward(self, x):

        out = self.conv1(x)
        #a = self.percentile(self.features[l](out_prev).view(-1), i)
        a, b = {}, {}
        for i in range(1, 100):
            a[i] = self.percentile(self.conv1(x).view(-1), i)
        out[out>self.threshold1] = self.threshold1
        out[out<0] = 0
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + self.identity(x)
        for i in range(1, 100):
            b[i] = self.percentile(out.view(-1), i)
        out[out>self.threshold2] = self.threshold2
        out[out<0] = 0
        return out, a, b

class ResNet_Tunable_Threshold(nn.Module):
        
    def __init__(self, block, num_blocks, labels=10, dropout=0.2, default_threshold=1.0):
        
        super(ResNet_Tunable_Threshold, self).__init__()

        self.in_planes      = 64
        self.dropout        = dropout
        threshold           = {}
        self.perc = {}
        self.pre_process    = nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(self.dropout),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(self.dropout),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                #nn.ReLU(inplace=True),
                                nn.MaxPool2d(2)
                                )
        for l in range(len(self.pre_process)):
            if isinstance(self.pre_process[l],nn.Conv2d):
				#self.register_buffer('threshold[l]', torch.tensor(default_threshold, requires_grad=True))
                threshold['t'+str(l)] 	= nn.Parameter(torch.tensor(default_threshold))
                
        pos = len(self.pre_process)#
        if num_blocks == [1, 1, 1, 1]:
            total_res = 14
            for l in range(total_res):
                threshold['t'+str(pos)] = nn.Parameter(torch.tensor(default_threshold))
                pos = pos + 1
        elif num_blocks == [2, 2, 2, 2]:
            total_res = 22
            for l in range(total_res):
                threshold['t'+str(pos)] = nn.Parameter(torch.tensor(default_threshold))
                pos = pos + 1
        elif num_blocks == [3, 4, 5, 3]:
            total_res = 36
            for l in range(total_res):
                threshold['t'+str(pos)] = nn.Parameter(torch.tensor(default_threshold))
                pos = pos + 1
                
        self.threshold 	= nn.ParameterDict(threshold)
        pos = len(self.pre_process)
        
        self.layer1, self.pos1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout=self.dropout, pos=pos)
        self.layer2, self.pos2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout=self.dropout, pos=self.pos1)
        self.layer3, self.pos3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout=self.dropout, pos=self.pos2)
        self.layer4, self.pos4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout=self.dropout, pos=self.pos3)
        self.classifier     = nn.Sequential(
                                nn.Linear(2048, labels, bias=False)
                                )
        #self.layers = {1: self.layer1, 2: self.layer2, 3: self.layer3, 4:self.layer4}
        
        
        self._initialize_weights2()

    def _initialize_weights2(self):
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        
    def _make_layer(self, block, planes, num_blocks, stride, dropout, pos):
        
        if num_blocks==0:
            return nn.Sequential()
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            threshold1 = getattr(self.threshold, 't'+str(pos))
            threshold2 = getattr(self.threshold, 't'+str(pos+1))
            layers.append(block(self.in_planes, planes, stride, dropout, threshold1, threshold2))
            self.in_planes = planes * block.expansion
            pos +=2
        
        return nn.Sequential(*layers), pos

    def percentile(self, t, q):

        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    def forward(self, x):

        out_prev = x
        threshold_out = []
        j = 0

        for l in range(len(self.pre_process)):
            if isinstance(self.pre_process[l], nn.Conv2d):
                #self.perc[str(j)] = {}
                #for i in range(1, 100):
                #    self.perc[str(j)][str(i)] = self.percentile(self.pre_process[l](out_prev).view(-1), i)
                #j += 1
                out = self.pre_process[l](out_prev)
                out[out<0] = 0
                out[out>getattr(self.threshold, 't'+str(l))] =  getattr(self.threshold, 't'+str(l))
                out_prev = out.clone()
                threshold_out.append(getattr(self.threshold, 't'+str(l)))
            else:
                out_prev = self.pre_process[l](out_prev)
        pos = len(self.pre_process)
        out = out_prev
        j = 0


        for layer in self.layer1:
            
            #for i in range(1, 100):
            #    self.perc[str(j)][str(i)] = self.percentile(layer(out).view(-1), i)
            #j += 1
            out, a, b = layer(out)
            #self.perc[str(pos+j)] = {}
            #for i in range(1, 100):
            #    self.perc[str(pos+j)][str(i)] = a[i]
            #j += 1
            #self.perc[str(pos+j)] = {}
            #for i in range(1, 100):
            #    self.perc[str(pos+j)][str(i)] = b[i]
            #j += 1

        for layer in self.layer2:
            out, a, b = layer(out)
            #self.perc[str(pos+j)] = {}
            #for i in range(1, 100):
            #    self.perc[str(pos+j)][str(i)] = a[i]
            #j += 1
            #self.perc[str(pos+j)] = {}
            #for i in range(1, 100):
            #    self.perc[str(pos+j)][str(i)] = b[i]
            #j += 1
            #for i in range(1, 100):
            #    self.perc[str(j)][str(i)] = self.percentile(layer(out).view(-1), i)
            #j += 1
        for layer in self.layer3:
            out, a, b = layer(out)
            #self.perc[str(pos+j)] = {}
            #for i in range(1, 100):
            #    self.perc[str(pos+j)][str(i)] = a[i]
            #j += 1
            #self.perc[str(pos+j)] = {}
            #for i in range(1, 100):
            #    self.perc[str(pos+j)][str(i)] = b[i]
            #j += 1
            #for i in range(1, 100):
            #    self.perc[str(j)][str(i)] = self.percentile(layer(out).view(-1), i)
            #j += 1
        for layer in self.layer4:
            out, a, b = layer(out)
            #self.perc[str(pos+j)] = {}
            #for i in range(1, 100):
            #    self.perc[str(pos+j)][str(i)] = a[i]
            #j += 1
            #self.perc[str(pos+j)] = {}
            #for i in range(1, 100):
            #    self.perc[str(pos+j)][str(i)] = b[i]
            #j += 1
            #for i in range(1, 100):
            #    self.perc[str(j)][str(i)] = self.percentile(layer(out).view(-1), i)
            #j += 1
                
        #out = self.layer1(out_prev)
        #out = self.layer2(out)
        #out = self.layer3(out)
        #out = self.layer4(out)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        for l in range(pos,self.pos4):
            threshold_out.append(getattr(self.threshold, 't'+str(l)))
        return out, threshold_out
        
def ResNet12(labels=10, dropout=0.2, default_threshold=4.0):
    return ResNet_Tunable_Threshold(block=BasicBlock, num_blocks=[1,1,1,1], labels=labels, dropout=dropout, default_threshold=default_threshold)

def ResNet20(labels=10, dropout=0.2, default_threshold=4.0):
    return ResNet_Tunable_Threshold(block=BasicBlock, num_blocks=[2,2,2,2], labels=labels, dropout=dropout, default_threshold=default_threshold)

def ResNet34(labels=10, dropout=0.2, default_threshold=4.0):
    return ResNet_Tunable_Threshold(block=BasicBlock, num_blocks=[3,4,5,3], labels=labels, dropout=dropout, default_threshold=default_threshold)

def test():
    print('In test()')
    net = ResNet12()
    print('Calling y=net() from test()')
    y = net(torch.randn(1,3,32,32))
    print(y.size())

if __name__ == '__main__':
    test()
