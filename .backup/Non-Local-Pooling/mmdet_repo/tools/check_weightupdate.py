import torch
import torch.nn

root = '/nas/home/fangc/mmdetection/work_dirs'

def check(pth_file1, pth_file2):
	dict1 = torch.load(pth_file1)['state_dict']
	dict2 = torch.load(pth_file2)['state_dict']

	common_keys = [k for k in dict1.keys() if k in dict2.keys()]
	for k in common_keys:
		v1 = dict1[k]
		v2 = dict2[k]
		if torch.all(v1==v2):
			print('Weight is unupdated in key:', k)
	print('Done.')

if __name__ == '__main__':
	pth_file1 = root+'/mybackbone/epoch_1.pth'
	pth_file2 = root+'/mybackbone/epoch_2.pth'
	pth_file3 = root+'/mybackbone/epoch_3.pth'
	check(pth_file1, pth_file3)