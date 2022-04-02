#---------------------------------------------------
# Imports
#---------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import pickle
import math
import json
from collections import OrderedDict
#from matplotlib import pyplot as plt
import copy

cfg = {
	'VGG4' : [64, 'A', 128, 'A'],
    'VGG6' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}


class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 #Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, threshold, scale, last_spike):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = float(threshold)*scale*1.3
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = torch.zeros_like(input).cuda()
        grad[input < 1] = 1.0
        grad[input > -1] = 1.0	
        grad[input >=1] = 0.0
        grad[input <=- 1] = 0.0
        #grad = input
        #grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input, grad_input, None, None
        #return last_spike*grad_input, None



class VGG_SNN(nn.Module):

	def __init__(self, vgg_name, activation='Linear', labels=10, timesteps=100, leak=1.0, default_threshold = 1.0, dropout=0.2, kernel_size=3, scaling_factor=0.5, dataset='CIFAR10'):
		super().__init__()
		
		self.vgg_name 		= vgg_name
		self.act_func 		= LinearSpike.apply
		self.labels 		= labels
		self.timesteps 		= timesteps
		self.dropout 		= dropout
		self.kernel_size 	= kernel_size
		self.dataset 		= dataset 
		self.mem 			= {}
		self.mask 			= {}
		self.spike 			= {}
		self.spike_count    = {}
		self.relu = nn.ReLU()
		self.scaling_factor = scaling_factor
		self.features, self.classifier = self._make_layers(cfg[self.vgg_name])
		with open('new_factor_x_vgg16_cifar10_1','rb') as f:
			self.scale_factor = pickle.load(f)
		with open('new_factor_y_vgg16_cifar10_1','rb') as f:
			self.scale_factor_y = pickle.load(f)
		#with open('max_vgg16.json','r') as f:
		#	self.max_factor = json.load(f)
		
		self._initialize_weights2()

		threshold 	= {}
		lk 	  		= {}
		#original_threshold = {}
		#scale_x     = {}
		#scale_y     = {}
		for l in range(len(self.features)):
			if isinstance(self.features[l], nn.Conv2d):
				threshold['t'+str(l)] 	= nn.Parameter(torch.tensor(default_threshold))
				lk['l'+str(l)] 			= nn.Parameter(torch.tensor(leak))
				#scale_y['t'+str(l)] 	= nn.Parameter(torch.tensor(default_threshold))
				#scale_x['l'+str(l)]			= nn.Parameter(torch.tensor(default_scale_x))
				#scale_y['l'+str(l)]			= nn.Parameter(torch.tensor(default_scale_y))
								
				
		prev = len(self.features)
		for l in range(len(self.classifier)-1):#-1
			if isinstance(self.classifier[l], nn.Linear):
				threshold['t'+str(prev+l)] 	= nn.Parameter(torch.tensor(default_threshold))
				lk['l'+str(prev+l)] 		= nn.Parameter(torch.tensor(leak))
				#scale_x['l'+str(prev+l)] 		= nn.Parameter(torch.tensor(default_scale_x))
				#scale_y['l'+str(prev+l)] 		= nn.Parameter(torch.tensor(default_threshold))
				
		self.threshold 	= nn.ParameterDict(threshold)
		self.threshold.requires_grad = True
		self.leak 		= nn.ParameterDict(lk)
		self.leak.requires_grad=True
		#self.scale_y = nn.ParameterDict(scale_y)
		#self.scale_y.requires_grad = False
		#self.original_threshold 	= nn.ParameterDict(threshold)
		#self.original_threshold.requires_grad = False
		#self.scale_x		= nn.ParameterDict(scale_x)
		#self.scale_x.requires_grad=True
		#self.scale_y		= nn.ParameterDict(scale_y)
		#self.scale_y.requires_grad=True
		self.dropout_conv   = nn.Dropout(self.dropout)
		self.dropout_linear = nn.Dropout(0.1)
				
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



	def threshold_update(self, scaling_factor=1.0):

		# Initialize thresholds
		self.scaling_factor = scaling_factor
		j = 0
		thresholds = self.threshold
		#self.original_threshold = []
		
		for pos in range(len(self.features)):
			if isinstance(self.features[pos], nn.Conv2d):
				#if thresholds:
					#self.threshold.update({'t'+str(pos): nn.Parameter(torch.tensor(thresholds.pop(0)*self.scale_factor[j]))})
				self.threshold.update({'t'+str(pos): nn.Parameter(torch.tensor(getattr(self.threshold, 't'+str(pos))*self.scale_factor[j]))})###*self.scale_factor[i]
				#self.threshold.update({'t'+str(pos): nn.Parameter(torch.tensor(getattr(self.max_factor, str(j))))})###*self.scale_factor[i]
				#self.threshold.update({'t'+str(pos): nn.Parameter(self.scale_factor[j])})
				#self.scale_y.update({'t'+str(pos): torch.tensor(self.scale_factor_y[i])})
				j += 1
				#self.original_threshold.append(torch.tensor(getattr(self.threshold, 't'+str(pos))))
				#print('\t Layer{} : {:.2f}'.format(pos, self.threshold[pos]))

		prev = len(self.features)

		for pos in range(len(self.classifier)-1):#-1
			if isinstance(self.classifier[pos], nn.Linear):
				#if thresholds:
				#	self.threshold.update({'t'+str(prev+pos): nn.Parameter(torch.tensor(thresholds.pop(0)*self.scale_factor[j]))})
       			
				self.threshold.update({'t'+str(prev+pos): nn.Parameter(torch.tensor(getattr(self.threshold, 't'+str(prev+pos))*self.scale_factor[j]))}) #####*self.scale_factor[i]
				#self.threshold.update({'t'+str(prev+pos): nn.Parameter(torch.tensor(getattr(self.max_factor, str(j))))}) #####*self.scale_factor[i]
				#self.scale_y.update({'t'+str(prev+pos): torch.tensor(self.scale_factor_y[i])})
				#self.threshold.update({'t'+str(prev+pos): nn.Parameter(self.scale_factor[j])})
				j += 1
				#self.original_threshold.append(torch.tensor(getattr(self.threshold, 't'+str(prev+pos))))
				#print('\t Layer{} : {:.2f}'.format(prev+pos, self.threshold[prev+pos]))
    




	def _make_layers(self, cfg):
		layers 		= []
		if self.dataset =='MNIST':
			in_channels = 1
		else:
			in_channels = 3

		for x in (cfg):
			stride = 1
						
			if x == 'A':
				#layers.pop()
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False)]
							#nn.ReLU(inplace=True)
							#]
				#layers += [nn.Dropout(self.dropout)]
				in_channels = x

		features = nn.Sequential(*layers)
		
		layers = []
		if self.dataset == 'IMAGENET':
			layers += [nn.Linear(512*7*7, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]

		elif self.vgg_name == 'VGG6' and self.dataset != 'MNIST':
			layers += [nn.Linear(512*4*4, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]

		elif self.vgg_name == 'VGG4' and self.dataset== 'MNIST':
			layers += [nn.Linear(128*7*7, 1024, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			#layers += [nn.Linear(4096, 4096, bias=False)]
			#layers += [nn.ReLU(inplace=True)]
			#layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(1024, self.labels, bias=False)]
		
		elif self.vgg_name != 'VGG6' and self.dataset != 'MNIST':
			layers += [nn.Linear(512*2*2, 4096, bias=False)]
			#layers += [nn.ReLU(inplace=True)]
			#layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			#layers += [nn.ReLU(inplace=True)]
			#layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]
		
		elif self.vgg_name == 'VGG6' and self.dataset == 'MNIST':
			layers += [nn.Linear(128*7*7, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]

		elif self.vgg_name != 'VGG6' and self.dataset == 'MNIST':
			layers += [nn.Linear(512*1*1, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]


		classifer = nn.Sequential(*layers)
		return (features, classifer)

	def network_update(self, timesteps, leak):
		self.timesteps 	= timesteps
			
	def neuron_init(self, x):
		self.batch_size = x.size(0)
		self.width 		= x.size(2)
		self.height 	= x.size(3)			
		
		self.mem 	= {}
		self.spike 	= {}
		self.spike_count = {}
		self.mask 	= {}

		for l in range(len(self.features)):
								
			if isinstance(self.features[l], nn.Conv2d):
				self.mem[l] 		= torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height)
				self.spike[l] 	= torch.ones(self.mem[l].shape)*(-1000)
				self.spike_count[l] 	= torch.zeros(self.mem[l].shape)
				self.mask[l] = self.dropout_conv(torch.ones(self.mem[l].shape).cuda())
			elif isinstance(self.features[l], nn.MaxPool2d):
				self.width = self.width//self.features[l].kernel_size
				self.height = self.height//self.features[l].kernel_size
		
		prev = len(self.features)

		for l in range(len(self.classifier)):
			
			if isinstance(self.classifier[l], nn.Linear):
				self.mem[prev+l] 		= torch.zeros(self.batch_size, self.classifier[l].out_features)
				self.spike[prev+l] 		= torch.ones(self.mem[prev+l].shape)*(-1000)
				self.spike_count[prev+l] 		= torch.zeros(self.mem[prev+l].shape)
				self.mask[prev+l] = self.dropout_linear(torch.ones(self.mem[prev+l].shape).cuda())
				#if (l==len(self.classifier)-1):
				#	self.spike[prev+l]=torch.ones(self.batch_size, self.classifier[l].out_features)*(-1000)
				#	self.out1[prev+l]=torch.zeros(self.batch_size, self.classifier[l].out_features)
			
			
				
				
		
	def percentile(self, t, q):

		k = 1 + round(.01 * float(q) * (t.numel() - 1))
		result = t.view(-1).kthvalue(k).values.item()
		return result
	
	def forward(self, x, find_max_mem=False, max_mem_layer=0):
		
		self.neuron_init(x)
		max_mem=0.0
				
		for t in range(self.timesteps):
			out_prev = x
			i = 0

			for l in range(len(self.features)):
				
				if isinstance(self.features[l], (nn.Conv2d)):
					
					if find_max_mem and l==max_mem_layer:
						cur = self.percentile(self.features[l](out_prev).view(-1), 99.7)
						if (cur>max_mem):
							max_mem = torch.tensor([cur])
						break
					
					#mem_thr 		= (self.mem[l]/getattr(self.threshold, 't'+str(l))) - 1.0
					#mem_current     = self.mem[l]
					#out_current     = (mem_thr>0).float()
					#rst 			= getattr(self.threshold, 't'+str(l)) * (mem_thr>0).float()
					#self.mem[l] 	= getattr(self.leak, 'l'+str(l)) *self.mem[l] + self.features[l](out_prev) - rst
					#mem_next        = self.mem[l]
					#mem_thr_next 	= (mem_next/getattr(self.threshold, 't'+str(l))) - 1.0
					#out_next        = (mem_thr_next>0).float()
					delta_mem 		= self.features[l](out_prev) #+  getattr(self.threshold, 't'+str(l))/40
					self.mem[l] 	= getattr(self.leak, 'l'+str(l)) *self.mem[l] + delta_mem
					mem_thr 		= (self.mem[l]/ getattr(self.threshold, 't'+str(l))) - 1.0
					out             = self.act_func(mem_thr,  getattr(self.threshold, 't'+str(l)), self.scale_factor_y[i], (t-1))
					
					#self.spike[l][out1[l]==1] = 0
					
					#self.out1[l][out > 0] = 1.0
					self.spike[l] 	= self.spike[l].masked_fill(out.bool(),t-1)
					self.spike_count[l][out.bool()] 	= self.spike_count[l][out.bool()] + 1
					rst 			=  getattr(self.threshold, 't'+str(l)) * (mem_thr>0).float()
					self.mem[l] 	= self.mem[l]-rst
					#out = out * getattr(self.threshold, 't'+str(l))
					out_prev 		= out * self.mask[l]
					i += 1
					#thres = getattr(self.threshold, 't'+str(l-1))
					#derivative = torch.div((out_next - out_current),(mem_next - mem_current))

					

				elif isinstance(self.features[l], nn.MaxPool2d):
					out_prev 		= self.features[l](out_prev)
					
			
			if find_max_mem and max_mem_layer<len(self.features):
				continue

			out_prev       	= out_prev.reshape(self.batch_size, -1)
			prev = len(self.features)
			
			for l in range(len(self.classifier)-1):#-1
													
				if isinstance(self.classifier[l], (nn.Linear)):
					
					if find_max_mem and (prev+l)==max_mem_layer:
						cur = self.percentile(self.classifier[l](out_prev).view(-1),99.7)
						if cur>max_mem:
							max_mem = torch.tensor([cur])
						break

					delta_mem 			= self.classifier[l](out_prev) #+  getattr(self.threshold, 't'+str(prev+l))/40
					self.mem[prev+l] 	= getattr(self.leak, 'l'+str(prev+l)) * self.mem[prev+l] + delta_mem
					mem_thr 			= (self.mem[prev+l]/ getattr(self.threshold, 't'+str(prev+l))) - 1.0
					rst 				= getattr(self.threshold, 't'+str(prev+l)) * (mem_thr>0).float()
					self.mem[prev+l] 	= self.mem[prev+l]-rst
					out 				= self.act_func(mem_thr,  getattr(self.threshold, 't'+str(prev+l)), self.scale_factor_y[i], (t-1))
					i += 1
					self.spike[prev+l] 	= self.spike[prev+l].masked_fill(out.bool(),t-1)
					#self.out1[prev+l][out==1] = 1
					self.spike_count[prev+l][out.bool()] 	= self.spike_count[prev+l][out.bool()] + 1
					#out = out * getattr(self.threshold, 't'+str(prev+l))
					out_prev 		= out * self.mask[prev+l]
					
			# Compute the classification layer outputs
			if not find_max_mem:
				self.mem[prev+l+1] 		= self.mem[prev+l+1] + self.classifier[l+1](out_prev)

		if find_max_mem:
			return max_mem

		return self.mem[prev+l+1], self.spike_count    



'''

#---------------------------------------------------
# Imports
#---------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
from collections import OrderedDict
from matplotlib import pyplot as plt
import copy

cfg = {
	'VGG4' : [64, 'A', 128, 'A'],
    'VGG5' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512],
    'CIFARNET': [128, 256, 'A', 512, 'A', 1204, 512]
}

class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, last_spike):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input, None

class VGG_SNN(nn.Module):

	def __init__(self, vgg_name, activation='Linear', labels=10, timesteps=100, leak=1.0, default_threshold = 1.0, dropout=0.2, kernel_size=3, dataset='CIFAR10', individual_thresh=False, vmem_drop=0):
		super().__init__()
		
		self.vgg_name 		= vgg_name
		if activation == 'Linear':
			self.act_func 	= LinearSpike.apply
		elif activation == 'STDB':
			self.act_func	= STDB.apply
		self.labels 		= labels
		self.timesteps 		= timesteps
		#STDB.alpha 		 	= alpha
		#STDB.beta 			= beta 
		self.dropout 		= dropout
		self.kernel_size 	= kernel_size
		self.dataset 		= dataset
		self.individual_thresh = individual_thresh
		self.vmem_drop 		= vmem_drop
		#self.threshold 		= nn.ParameterDict()
		#self.leak 			= nn.ParameterDict()
		self.mem 			= {}
		self.mask 			= {}
		self.spike 			= {}
		
		self.features, self.classifier = self._make_layers(cfg[self.vgg_name])
		
		self._initialize_weights2()

		threshold 	= {}
		lk 	  		= {}
		if self.dataset in ['CIFAR10', 'CIFAR100']:
			width = 32
			height = 32
		elif self.dataset=='MNIST':
			width = 28
			height = 28
		elif self.dataset=='IMAGENET':
			width = 224
			height = 224

		for l in range(len(self.features)):
			if isinstance(self.features[l], nn.Conv2d):
				if self.individual_thresh:
					threshold['t'+str(l)] 	= nn.Parameter(torch.ones(self.features[l].out_channels, width, height)*default_threshold)
					lk['l'+str(l)] 			= nn.Parameter(torch.ones(self.features[l].out_channels, width, height)*leak)
				else:
					threshold['t'+str(l)] 	= nn.Parameter(torch.tensor(default_threshold))
					lk['l'+str(l)]			= nn.Parameter(torch.tensor(leak))
				#threshold['t'+str(l)] 	= nn.Parameter(torch.empty(1,1).fill_(default_threshold))
				#lk['l'+str(l)]			= nn.Parameter(torch.empty(1,1).fill_(leak))
			elif isinstance(self.features[l], nn.AvgPool2d):
				width 	= width//self.features[l].kernel_size
				height 	= height//self.features[l].kernel_size
				
				
		prev = len(self.features)
		for l in range(len(self.classifier)-1):
			if isinstance(self.classifier[l], nn.Linear):
				if self.individual_thresh:
					threshold['t'+str(prev+l)]	= nn.Parameter(torch.ones(self.classifier[l].out_features)*default_threshold)
					lk['l'+str(prev+l)] 		= nn.Parameter(torch.ones(self.classifier[l].out_features)*leak)
				else:
					threshold['t'+str(prev+l)] 	= nn.Parameter(torch.tensor(default_threshold))
					lk['l'+str(prev+l)] 		= nn.Parameter(torch.tensor(leak))
				#threshold['t'+str(prev+l)] 	= nn.Parameter(torch.empty(1,1).fill_(default_threshold))
				#lk['l'+str(prev+l)] 		= nn.Parameter(torch.empty(1,1).fill_(leak))

		#pdb.set_trace()
		self.threshold 	= nn.ParameterDict(threshold)
		self.leak 		= nn.ParameterDict(lk)
		
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

	def threshold_update(self, scaling_factor=1.0, thresholds=[]):

		# Initialize thresholds
		self.scaling_factor = scaling_factor
		
		if self.dataset in ['CIFAR10', 'CIFAR100']:
			width = 32
			height = 32
		elif self.dataset=='MNIST':
			width = 28
			height = 28
		elif self.dataset=='IMAGENET':
			width = 224
			height = 224

		for pos in range(len(self.features)):
			if isinstance(self.features[pos], nn.Conv2d):
				if thresholds:
					if self.individual_thresh:
						self.threshold.update({'t'+str(pos): nn.Parameter(torch.ones(self.features[pos].out_channels, width, height)*thresholds.pop(0)*self.scaling_factor)})
					else:
						self.threshold.update({'t'+str(pos): nn.Parameter(torch.tensor(thresholds.pop(0))*self.scaling_factor)})
				#print('\t Layer{} : {:.2f}'.format(pos, self.threshold[pos]))
			elif isinstance(self.features[pos], nn.AvgPool2d):
				width 	= width//self.features[pos].kernel_size
				height 	= height//self.features[pos].kernel_size

		prev = len(self.features)

		for pos in range(len(self.classifier)-1):
			if isinstance(self.classifier[pos], nn.Linear):
				if thresholds:
					if self.individual_thresh:
						self.threshold.update({'t'+str(prev+pos): nn.Parameter(torch.ones(self.classifier[pos].out_features)*thresholds.pop(0)*self.scaling_factor)})
					else:
						self.threshold.update({'t'+str(prev+pos): nn.Parameter(torch.tensor(thresholds.pop(0))*self.scaling_factor)})
				#print('\t Layer{} : {:.2f}'.format(prev+pos, self.threshold[prev+pos]))


	def _make_layers(self, cfg):
		layers 		= []
		if self.dataset =='MNIST':
			in_channels = 1
		else:
			in_channels = 3

		for x in (cfg):
			stride = 1
						
			if x == 'A':
				layers.pop()
				layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
			
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False),
							nn.ReLU(inplace=True)
							]
				layers += [nn.Dropout(self.dropout)]
				in_channels = x

		if self.dataset== 'IMAGENET':
			layers.pop()
			layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
			
		features = nn.Sequential(*layers)
		
		layers = []
		if self.dataset == 'IMAGENET':
			layers += [nn.Linear(512*7*7, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]

		elif self.vgg_name == 'VGG5' and self.dataset != 'MNIST':
			layers += [nn.Linear(512*4*4, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]

		elif self.vgg_name == 'VGG4' and self.dataset== 'MNIST':
			layers += [nn.Linear(128*7*7, 1024, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			#layers += [nn.Linear(4096, 4096, bias=False)]
			#layers += [nn.ReLU(inplace=True)]
			#layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(1024, self.labels, bias=False)]
		
		elif self.vgg_name != 'VGG5' and self.dataset != 'MNIST':
			layers += [nn.Linear(512*2*2, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]
		
		elif self.vgg_name == 'VGG5' and self.dataset == 'MNIST':
			layers += [nn.Linear(128*7*7, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]

		elif self.vgg_name != 'VGG5' and self.dataset == 'MNIST':
			layers += [nn.Linear(512*1*1, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(self.dropout)]
			layers += [nn.Linear(4096, self.labels, bias=False)]


		classifer = nn.Sequential(*layers)
		return (features, classifer)

	def network_update(self, timesteps, leak):
		self.timesteps 	= timesteps
		#for key, value in sorted(self.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
		#	if isinstance(leak, list) and leak:
		#		self.leak.update({key: nn.Parameter(torch.tensor(leak.pop(0)))})
	
	def neuron_init(self, x):
		self.batch_size = x.size(0)
		self.width 		= x.size(2)
		self.height 	= x.size(3)			
		
		self.mem 	= {}
		self.spike 	= {}
		self.mask 	= {}

		for l in range(len(self.features)):
								
			if isinstance(self.features[l], nn.Conv2d):
				self.mem[l] 		= torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height)
			
			elif isinstance(self.features[l], nn.ReLU):
				if isinstance(self.features[l-1], nn.Conv2d):
					self.spike[l] 	= torch.ones(self.mem[l-1].shape)*(-1000)
				elif isinstance(self.features[l-1], nn.AvgPool2d):
					self.spike[l] 	= torch.ones(self.batch_size, self.features[l-2].out_channels, self.width, self.height)*(-1000)

			elif isinstance(self.features[l], nn.Dropout):
				self.mask[l] = self.features[l](torch.ones(self.mem[l-2].shape).cuda())

			elif isinstance(self.features[l], nn.AvgPool2d):
				self.width = self.width//self.features[l].kernel_size
				self.height = self.height//self.features[l].kernel_size
		
		prev = len(self.features)

		for l in range(len(self.classifier)):
			
			if isinstance(self.classifier[l], nn.Linear):
				self.mem[prev+l] 		= torch.zeros(self.batch_size, self.classifier[l].out_features)
			
			elif isinstance(self.classifier[l], nn.ReLU):
				self.spike[prev+l] 		= torch.ones(self.mem[prev+l-1].shape)*(-1000)

			elif isinstance(self.classifier[l], nn.Dropout):
				self.mask[prev+l] = self.classifier[l](torch.ones(self.mem[prev+l-2].shape).cuda())
				
		# self.spike = copy.deepcopy(self.mem)
		# for key, values in self.spike.items():
		# 	for value in values:
		# 		value.fill_(-1000)

	def percentile(self, t, q):

		k = 1 + round(.01 * float(q) * (t.numel() - 1))
		result = t.view(-1).kthvalue(k).values.item()
		return result
	
	def custom_dropout(self, tensor, prob, conv=True):
		if prob==0:
			return tensor
		mask_retain			= F.dropout(tensor, p=prob, training=True).bool()*1
		mask_drop 			= (mask_retain==0)*1
		tensor_retain 		= tensor*mask_retain
		tensor_drop 		= tensor*mask_drop
		if conv:
			tensor_drop_sum	= tensor_drop.sum(dim=(2,3))
			retain_no 		= mask_retain.sum(dim=(2,3))
			tensor_drop_sum[retain_no==0] = 0
			retain_no[retain_no==0] = 1
			increment 		= tensor_drop_sum/(retain_no)
			increment_array = increment.repeat_interleave(tensor.shape[2]*tensor.shape[3]).view(tensor.shape)
		else:
			#pdb.set_trace()
			tensor_drop_sum = tensor_drop.sum(dim=1)
			retain_no 		= mask_retain.sum(dim=1)
			tensor_drop_sum[retain_no==0] = 0
			retain_no[retain_no==0] = 1
			increment 		= tensor_drop_sum/(retain_no)	
			increment_array = increment.repeat_interleave(tensor.shape[1]).view(tensor.shape)
			
		increment_array = increment_array*mask_retain
		new_tensor 		= tensor_retain+increment_array
		
		return new_tensor

	def forward(self, x, find_max_mem=False, max_mem_layer=0, percentile=99.7):
		
		self.neuron_init(x)
		max_mem=0.0
		# if find_max_mem:
		# 	prob=self.vmem_drop
		# else:
		# 	prob=self.vmem_drop
		#ann = [0]*len(self.features)
		#ann[1] = 1
		#pdb.set_trace()
		for t in range(self.timesteps):
			out_prev = x
			# keys = [*self.mem]
			# print('time: {}'.format(t), end=', ')
			# for l, key in enumerate(keys):
			# 	print('l{}: {:.1f}'.format(l+1, self.mem[key].max()), end=', ')
			# print()
			# input()
			for l in range(len(self.features)):
				#if l==27:
				#	pdb.set_trace()
				if isinstance(self.features[l], (nn.Conv2d)):
					
					if find_max_mem and l==max_mem_layer:
						#if t==9:
							#pdb.set_trace()
						cur = self.percentile(self.features[l](out_prev).view(-1), percentile)
						if (cur>max_mem):
							max_mem = torch.tensor([cur])
						break
					
					#mem_thr 		= (self.mem[l]/getattr(self.threshold, 't'+str(l))) - 1.0
					#rst 			= getattr(self.threshold, 't'+str(l)) * (mem_thr>0).float()
					#self.mem[l] 	= getattr(self.leak, 'l'+str(l)) *self.mem[l] + self.features[l](out_prev) - rst
					#delta_mem 		= self.custom_dropout(self.features[l](out_prev), prob=prob, conv=True)
					delta_mem 		= self.features[l](out_prev)
					self.mem[l] 	= getattr(self.leak, 'l'+str(l)) *self.mem[l] + delta_mem
					mem_thr 		= (self.mem[l]/getattr(self.threshold, 't'+str(l))) - 1.0
					rst 			= getattr(self.threshold, 't'+str(l)) * (mem_thr>0).float()
					self.mem[l] 	= self.mem[l]-rst
					#out_prev 		= self.features[l](out_prev)
					
				elif isinstance(self.features[l], nn.ReLU):
					#pdb.set_trace()
					out 			= self.act_func(mem_thr, (t-1-self.spike[l]))
					self.spike[l] 	= self.spike[l].masked_fill(out.bool(),t-1)
					out_prev  		= out.clone()

				elif isinstance(self.features[l], nn.AvgPool2d):
					out_prev 		= self.features[l](out_prev)
				
				elif isinstance(self.features[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[l]
			
			if find_max_mem and max_mem_layer<len(self.features):
				continue

			out_prev       	= out_prev.reshape(self.batch_size, -1)
			prev = len(self.features)
			#pdb.set_trace()
			for l in range(len(self.classifier)-1):
													
				if isinstance(self.classifier[l], (nn.Linear)):
					
					if find_max_mem and (prev+l)==max_mem_layer:
						#pdb.set_trace()
						cur = self.percentile(self.classifier[l](out_prev).view(-1), percentile)
						if cur>max_mem:
							max_mem = torch.tensor([cur])
						break

					#mem_thr 			= (self.mem[prev+l]/getattr(self.threshold, 't'+str(prev+l))) - 1.0
					#rst 				= getattr(self.threshold,'t'+str(prev+l)) * (mem_thr>0).float()
					#self.mem[prev+l] 	= getattr(self.leak, 'l'+str(prev+l)) * self.mem[prev+l] + self.classifier[l](out_prev) - rst
					#delta_mem 			= self.custom_dropout(self.classifier[l](out_prev), prob=prob, conv=False)
					delta_mem 			= self.classifier[l](out_prev)
					self.mem[prev+l] 	= getattr(self.leak, 'l'+str(prev+l)) * self.mem[prev+l] + delta_mem
					mem_thr 			= (self.mem[prev+l]/getattr(self.threshold, 't'+str(prev+l))) - 1.0
					rst 				= getattr(self.threshold,'t'+str(prev+l)) * (mem_thr>0).float()
					self.mem[prev+l] 	= self.mem[prev+l]-rst

				
				elif isinstance(self.classifier[l], nn.ReLU):
					out 				= self.act_func(mem_thr, (t-1-self.spike[prev+l]))
					self.spike[prev+l] 	= self.spike[prev+l].masked_fill(out.bool(),t-1)
					out_prev  			= out.clone()

				elif isinstance(self.classifier[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[prev+l]
			
			# Compute the classification layer outputs
			if not find_max_mem:
				self.mem[prev+l+1] 		= self.mem[prev+l+1] + self.classifier[l+1](out_prev)
		if find_max_mem:
			return max_mem

		return self.mem[prev+l+1]

'''

