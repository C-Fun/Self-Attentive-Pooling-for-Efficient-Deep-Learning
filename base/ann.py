import warnings
warnings.filterwarnings("ignore")

import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchviz import make_dot
from matplotlib import pyplot as plt
import pdb
import sys
import datetime
import os
import cv2
import numpy as np
import json
import pickle

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.network import Network, name_parse

# root = 'E:/留学相关/研究/RPIXEL/' # Windows
root = '/nas/home/fangc/' # Linux

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)

def compute_mac(model, dataset):

	if dataset.lower().startswith('cifar'):
		h_in, w_in = 32, 32
	elif dataset.lower().startswith('image'):
		h_in, w_in = 224, 224
	elif dataset.lower().startswith('mnist'):
		h_in, w_in = 28, 28

	macs = []
	for name, l in model.named_modules():
		if isinstance(l, nn.Conv2d):
			c_in    = l.in_channels
			k       = l.kernel_size[0]
			h_out   = int((h_in-k+2*l.padding[0])/(l.stride[0])) + 1
			w_out   = int((w_in-k+2*l.padding[0])/(l.stride[0])) + 1
			c_out   = l.out_channels
			mac     = k*k*c_in*h_out*w_out*c_out
			if mac == 0:
				pdb.set_trace()
			macs.append(mac)
			h_in    = h_out
			w_in    = w_out
			print('{}, Mac:{}'.format(name, mac))
		if isinstance(l, nn.Linear):
			mac     = l.in_features * l.out_features
			macs.append(mac)
			print('{}, Mac:{}'.format(name, mac))
		if isinstance(l, nn.AvgPool2d):
			h_in    = h_in//l.kernel_size
			w_in    = w_in//l.kernel_size
	print('{:e}'.format(sum(macs)))
	exit()

def train(epoch, loader):

	global learning_rate
	
	losses = AverageMeter('Loss')
	top1   = AverageMeter('Acc@1')

	if epoch in lr_interval:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] / lr_reduce
			learning_rate = param_group['lr']

	#if epoch in lr_interval:
	#else:
	#    for param_group in optimizer.param_groups:
	#        param_group['lr'] = param_group['lr'] / ((1000-2*(epoch-1))/(998-2*(epoch-1)))
	#        learning_rate = param_group['lr']
	
	#total_correct   = 0
	model.train()
	with tqdm(loader, total=len(loader)) as t:
		for batch_idx, (data, target) in enumerate(t):
			
			#start_time = datetime.datetime.now()

			if torch.cuda.is_available() and args.gpu:
				data, target = data.cuda(), target.cuda()

			optimizer.zero_grad()
			# output, _ = model(data)
			# output, _ = model(data, epoch)
			output = model(data)
			loss = F.cross_entropy(output,target)
			#make_dot(loss).view()
			#exit(0)
			loss.backward(inputs = list(model.parameters()))
			optimizer.step()
			#for p in model.parameters():
			#    p.data.clamp_(0)
			pred = output.max(1,keepdim=True)[1]
			correct = pred.eq(target.data.view_as(pred)).cpu().sum()
			#total_correct += correct.item()

			losses.update(loss.item(), data.size(0))
			top1.update(correct.item()/data.size(0), data.size(0))

			if batch_idx % 1 == 0:
				t.set_postfix_str("train_loss: {:.4f}, train_acc: {:.4f}".format(loss.item(), correct.item()/data.size(0)))
		
		f.write('\n Epoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}'.format(
				epoch,
				learning_rate,
				losses.avg,
				top1.avg
				)
			)

def test(epoch, loader):

	losses = AverageMeter('Loss')
	top1   = AverageMeter('Acc@1')

	with torch.no_grad():
		model.eval()
		total_loss = 0
		correct = 0
		#dis = []

		global max_accuracy, start_time
		
		for batch_idx, (data, target) in enumerate(loader):

			if torch.cuda.is_available() and args.gpu:
				data, target = data.cuda(), target.cuda()
			
			# output, thresholds = model(data, epoch)
			#output, thresholds = model(data)
			output = model(data)
			#dis.extend(act)
			loss = F.cross_entropy(output,target)
			total_loss += loss.item()
			pred = output.max(1, keepdim=True)[1]
			correct = pred.eq(target.data.view_as(pred)).cpu().sum()
			losses.update(loss.item(), data.size(0))
			top1.update(correct.item()/data.size(0), data.size(0))
			#break

		#with open('percentiles_resnet20_cifar100.json','w') as f:
		#    json.dump(percentiles, f)

		#with open('thresholds_resnet20_cifar100_new', 'wb') as fp:
		#    pickle.dump(thresholds, fp)
		
		#with open('activations','wb') as f:
		#    pickle.dump(dis, f)

		#if epoch>30 and top1.avg<0.15:
		#    f.write('\n Quitting as the training is not progressing')
		#    exit(0)

		
		if top1.avg>max_accuracy:
			max_accuracy = top1.avg
			state = {
					'accuracy'      : max_accuracy,
					'epoch'         : epoch,
					'state_dict'    : model.state_dict(),
					'optimizer'     : optimizer.state_dict()
			}

			ann_path = './trained_models_ann/'+arch_name+'/'
			try:
				os.makedirs(ann_path)
			except OSError:
				pass
			
			filename = ann_path+identifier+'.pth'
			if not args.dont_save:
				torch.save(state,filename)
		#dis = np.array(dis)
		#

		f.write(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f}, time: {}'.format(
			losses.avg, 
			top1.avg,
			max_accuracy,
			datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
			)
		)

		# print("threshold value is \n")
		# print(thresholds)
		# f.write('\n Time: {}'.format(
		#     datetime.timedelta(seconds=(datetime.datetime.now() - current_time).seconds)
		#     )
		# )

def visualize(loader, num_classes, visual_type=['directly', 'grad_cam']):
	def minmax(x):
		return (x-np.min(x))/(1e-10+np.max(x)-np.min(x))

	activation = {}
	def get_activation(name):
		def hook(model, input, output):
			activation[name] = output
		return hook

	name_list = []
	module_list = []
	for (name, module) in model.named_modules():
		if name.endswith('pool_weight'):
			module.register_forward_hook(get_activation(name))
			name_list.append(name)
			module_list.append(module)

	for batch_idx, (data, target) in enumerate(loader):

		if torch.cuda.is_available() and args.gpu:
			data, target = data.cuda(), target.cuda()

		
		output = model(data)
		pred = output.max(1,keepdim=True)[1]
		(b, c, h, w) = data.shape
		img = data[0,:,:,:].detach().cpu().numpy().transpose([1,2,0])
		img = np.uint8(255 * minmax(img))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# Directly Apply
		if 'directly' in visual_type:
			plt.figure(1)
			plt.suptitle('Directly')
			subfig_num = len(name_list)+1
			plt.subplot(100+10*subfig_num+1)
			plt.imshow(img)
			plt.title('Image')
			for (i, name) in enumerate(name_list):
				weight = activation[name]
				restore_weight = F.interpolate(weight, size=(h,w), mode='bilinear')
				avg_weight = torch.mean(restore_weight, axis=1)
				heatmap = avg_weight.detach().cpu().numpy().transpose([1,2,0])
				heatmap = np.uint8(255 * minmax(heatmap))
				heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
				heatimg = minmax(heatmap*0.9+img)
				plt.subplot(100+10*subfig_num+i+2)
				plt.imshow(heatimg)
				plt.title('Pool Layer '+str(i+1))


		# Grad Cam
		if 'grad_cam' in visual_type:
			plt.figure(2)
			plt.suptitle('Grad CAM')
			target_layers = []
			for module in module_list:
				target_layers.append(module)
				# target_layers = [module]

			cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

			target_cam = cam(input_tensor=data, targets=[ClassifierOutputTarget(target)])
			target_cam = target_cam[0, :]
			target_visual = show_cam_on_image(minmax(img), target_cam, use_rgb=True)

			pred_cam = cam(input_tensor=data, targets=[ClassifierOutputTarget(pred)])
			pred_cam = pred_cam[0, :]
			pred_visual = show_cam_on_image(minmax(img), pred_cam, use_rgb=True)

			plt.subplot(131)
			plt.imshow(img)
			plt.title('Image')
			plt.subplot(132)
			plt.imshow(target_visual)
			plt.title('Target: '+str(target.detach().cpu().numpy()[0]))
			plt.subplot(133)
			plt.imshow(pred_visual)
			plt.title('Pred: '+str(pred.detach().cpu().numpy()[0,0]))

		plt.show()





if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train ANN to be later converted to SNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
	parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
	parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
	parser.add_argument('--momentum',               default=0.9,                  type=float,       help='mometum of optimizer')
	parser.add_argument('--amsgrad',               default=True,                  type=bool,       help='amsgrad')
	parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100', 'IMAGENET', 'STL10'])
	parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
	# parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
	parser.add_argument('-a','--architecture',      default=None,            type=str,       help='network architecture')
	parser.add_argument('--im_size',                 default=None,             type=int,         help='image size')
	parser.add_argument('-rthr','--relu_threshold', default='4.0',            type=float,       help='threshold value for the RELU activation')
	parser.add_argument('-lr','--learning_rate',    default=1e-2,               type=float,     help='initial learning_rate')
	parser.add_argument('--pretrained_backbone',         default='',                 type=str,       help='pretrained model to initialize Backbone')
	parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained model to initialize ANN')
	parser.add_argument('--weight_decay',           default=0.000,               type=float,       help='weight_decay')
	parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
	parser.add_argument('--epochs',                 default=300,                type=int,       help='number of training epochs')
	parser.add_argument('--lr_interval',            default='0.45 0.70 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
	parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
	parser.add_argument('--optimizer',              default='SGD',             type=str,        help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
	parser.add_argument('--dropout',                default=0.2,                type=float,     help='dropout percentage for conv layers')
	parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
	parser.add_argument('--dont_save',              action='store_true',                        help='don\'t save training model during testing')
	parser.add_argument('--visualize',              action='store_true',                        help='visualize the attention map')

	parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')

	args=parser.parse_args()

	# os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

	# Seed random number
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	dataset         = args.dataset
	batch_size      = args.batch_size
	architecture    = args.architecture
	im_size 		= args.im_size
	learning_rate   = args.learning_rate
	pretrained_ann  = args.pretrained_ann
	epochs          = args.epochs
	lr_reduce       = args.lr_reduce
	optimizer       = args.optimizer
	dropout         = args.dropout
	momentum        = args.momentum
	kernel_size     = args.kernel_size
	threshold       = args.relu_threshold
	weight_decay    = args.weight_decay
	amsgrad         = args.amsgrad
		
	# Training settings
	#if torch.cuda.is_available() and args.gpu:
	#    torch.set_default_tensor_type('torch.cuda.FloatTensor')
	
	# Loading Dataset
	if dataset == 'CIFAR100':
		normalize   = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
		labels      = 100
	elif dataset == 'CIFAR10':
		normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
		labels      = 10
	elif dataset == 'MNIST':
		labels = 10
	elif dataset == 'IMAGENET':
		normalize   = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		labels = 1000
	elif dataset == 'STL10':
		normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
										 std=[0.5, 0.5, 0.5])
		labels = 10


	if dataset == 'CIFAR10' or dataset == 'CIFAR100':
		if im_size==None:
			im_size = 32
			transform_train = transforms.Compose([
							  transforms.RandomCrop(32, padding=4),
							  transforms.RandomHorizontalFlip(),
							  transforms.ToTensor(),
							  normalize])
		else:
			transform_train = transforms.Compose([
							  transforms.Resize(im_size),
							  transforms.RandomResizedCrop(im_size),
							  transforms.RandomHorizontalFlip(),
							  transforms.ToTensor(),
							  normalize])
		transform_test = transforms.Compose([transforms.ToTensor(), normalize])
	
	if dataset == 'CIFAR100':
		train_dataset   = datasets.CIFAR100(root=root+'/data/cifar_data', train=True, download=True,transform =transform_train)
		test_dataset    = datasets.CIFAR100(root=root+'/data/cifar_data', train=False, download=True, transform=transform_test)

	elif dataset == 'CIFAR10': 
		train_dataset   = datasets.CIFAR10(root=root+'/data/cifar_data', train=True, download=True,transform =transform_train)
		test_dataset    = datasets.CIFAR10(root=root+'/data/cifar_data', train=False, download=True, transform=transform_test)
	
	elif dataset == 'MNIST':
		train_dataset   = datasets.MNIST(root=root+'/data/mnist', train=True, download=True, transform=transforms.ToTensor()
			)
		test_dataset    = datasets.MNIST(root=root+'/data/mnist', train=False, download=True, transform=transforms.ToTensor())
	
	elif dataset == 'IMAGENET':
		traindir    = os.path.join('/m2/data/imagenet', 'train')
		valdir      = os.path.join('/m2/data/imagenet', 'val')
		if im_size==None:
			im_size = 224
			train_dataset    = datasets.ImageFolder(
								traindir,
								transforms.Compose([
									transforms.RandomResizedCrop(224),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									normalize,
								]))
		else:
			train_dataset    = datasets.ImageFolder(
								traindir,
								transforms.Compose([
									transforms.Resize(im_size),
									transforms.RandomResizedCrop(im_size),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									normalize,
								]))
		test_dataset     = datasets.ImageFolder(
							valdir,
							transforms.Compose([
								transforms.Resize(im_size),
								transforms.CenterCrop(im_size),
								transforms.ToTensor(),
								normalize,
							]))
	
	elif dataset == 'STL10':
		if im_size==None:
			im_size = 96
			transform_train = transforms.Compose([
							  transforms.RandomResizedCrop(96),
							  transforms.RandomHorizontalFlip(),
							  transforms.ToTensor(),
							  normalize])
		else:
			transform_train = transforms.Compose([
							  transforms.Resize(im_size),
							  transforms.RandomResizedCrop(im_size),
							  transforms.RandomHorizontalFlip(),
							  transforms.ToTensor(),
							  normalize])
		transform_test = transforms.Compose([
						 transforms.Resize(im_size),
						 transforms.CenterCrop(im_size),
						 transforms.ToTensor(),
						 normalize,
						 ])
		train_dataset = datasets.stl10.STL10(root=root+"/data/stl10_data", split="train", download=True, transform=transform_train)
		test_dataset = datasets.stl10.STL10(root=root+"/data/stl10_data", split="test", download=True, transform=transform_test)


	train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	
	test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

	# logs out
	values = args.lr_interval.split()
	lr_interval = []
	for value in values:
		lr_interval.append(int(float(value) * args.epochs))

	arch_name = architecture.lower()
	model_cfg = name_parse(arch_name)
	model_json = json.dumps(model_cfg, sort_keys=False, indent=4, separators=(',', ': '))

	arch_name_list = arch_name.split('-')
	backbone_name = arch_name_list[-3].split('_')[0]
	if len(arch_name_list)==3:
		folder_name = backbone_name
	elif len(arch_name_list)==6:
		folder_name = '-'.join([arch_name_list[0], backbone_name])
	else:
		folder_name = '-'.join(['unknown', backbone_name])
	log_file = './logs_new/' + folder_name + '/'
	try:
		os.makedirs(log_file)
	except OSError:
		pass

	identifier = arch_name + '_' + dataset.lower() + '_imsize' + str(im_size)
	log_file += identifier + '.log'

	if args.log:
		f = open(log_file, 'w', buffering=1)
	else:
		f = sys.stdout

	f.write('\n Run on time: {}'.format(datetime.datetime.now()))

	f.write('\n\n Architecture: {}'.format(arch_name))
	f.write('\n\n Pool Config: {}'.format(model_json))

	f.write('\n\n Arguments:')
	for arg in vars(args):
		if arg == 'lr_interval':
			f.write('\n\t {:20} : {}'.format(arg, lr_interval))
		else:
			f.write('\n\t {:20} : {}'.format(arg, getattr(args, arg)))


	# prepare model
	if args.pretrained_backbone:
		pth_file = args.pretrained_backbone
	else:
		pth_file = None

	model = Network(model_cfg, num_classes=labels, pth_file=pth_file)

	total_params = sum(p.numel() for p in model.parameters())
	print(f'{total_params:,} total parameters.')
	total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'{total_trainable_params:,} training parameters.')

	device_ids = [id for id in range(len(args.devices.split(',')))]
	model = nn.DataParallel(model, device_ids=device_ids)

	# print(model)
	for name, param in model.named_parameters():
		if param.requires_grad:
			print('Trainable:', name)

	# f.write('\n{}'.format(model))
	
	#CIFAR100 sometimes has problem to start training
	#One solution is to train for CIFAR10 with same architecture
	#Load the CIFAR10 trained model except the final layer weights
	#model.cuda()
	#model = nn.DataParallel(model.cuda(),device_ids=[0,1,2])
	#model.cuda()
	#model = nn.DataParallel(model)
	if args.pretrained_ann:
		state = torch.load(pretrained_ann, map_location='cpu')
		
		missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
		f.write('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))        
		f.write('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))
		'''
		state=torch.load(args.pretrained_ann, map_location='cpu')
		cur_dict = model.state_dict()
		for key in state['state_dict'].keys():
			if key in cur_dict:
				if (state['state_dict'][key].shape == cur_dict[key].shape):
					cur_dict[key] = nn.Parameter(state[key].data)
					f.write('\n Success: Loaded {} from {}'.format(key, pretrained_ann))
				else:
					f.write('\n Error: Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
			else:
				f.write('\n Error: Loaded weight {} not present in current model'.format(key))
		
		#model.load_state_dict(cur_dict)
		'''
		#model.load_state_dict(torch.load(args.pretrained_ann, map_location='cpu')['state_dict'])

		#for param in model.features.parameters():
		#    param.require_grad = False
		#num_features = model.classifier[6].in_features
		#features = list(model.classifier.children())[:-1] # Remove last layer
		#features.extend([nn.Linear(num_features, 1000)]) # Add our layer with 4 outputs
		#model.classifier = nn.Sequential(*features) # Replace the model classifier

		
	
	f.write('\n {}'.format(model)) 
	
	if torch.cuda.is_available() and args.gpu:
		model.cuda()
	
	if optimizer == 'SGD':
		optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
	elif optimizer == 'Adam':
		optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)
	
	f.write('\n {}'.format(optimizer))
	
	max_accuracy = 0.0
	#compute_mac(model, dataset)
	for epoch in range(1, epochs):
		start_time = datetime.datetime.now()
		if not args.test_only:
			train(epoch, train_loader)
		test(epoch, test_loader)
	if args.visualize:
		visual_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
		visualize(visual_loader, num_classes=labels)

	f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))
