# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data import ValData

from utils import validation_PSNR, generate_filelist

from MAMFNet import MAMFNet
import os
# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='PyTorch implementation of MAMFNet from Liu et al. (2024)')
parser.add_argument('-d', '--dataset-name', help='name of dataset',choices=['NH', 'dense', 'indoor','6k','our_test'], default='NH')
parser.add_argument('-t', '--test-image-dir', help='test images path', default='')
parser.add_argument('-c', '--ckpts-dir', help='ckpts path', default='')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
args = parser.parse_args()


val_batch_size = args.val_batch_size
dataset_name = args.dataset_name
# import pdb;pdb.set_trace()
# --- Set dataset-specific hyper-parameters  --- #
if dataset_name == 'NH':
    val_data_dir = './data/NH-HAZE/valid_NH/'
    ckpts_dir = './ckpts/NH/NH-20.10ssim6716.pt'
elif dataset_name == 'dense': 
    val_data_dir = './data/dataset/Dense-Haze/valid_dense'
    ckpts_dir = './ckpts/dense/DENSE-16.42ssim0.5235.pt'
elif dataset_name == 'indoor': 
    val_data_dir = ''
    ckpts_dir = ''
elif dataset_name == '6k': 
    val_data_dir = './data/6K/test/'
    ckpts_dir = './ckpts/6k/6K-30.20ssim0.9643.pt'
else:
    val_data_dir = args.test_image_dir
    ckpts_dir =  args.ckpts_dir

# prepare .txt file
if not os.path.exists(os.path.join(val_data_dir, 'val_list.txt')):
    generate_filelist(val_data_dir, valid=True)

# --- Gpu device --- #
device_ids =  [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Validation data loader --- #
val_data_loader = DataLoader(ValData(dataset_name,val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=24)


# --- Define the network --- #
net = MAMFNet()

  
# --- Multi-GPU --- # 
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(torch.load(ckpts_dir), strict=False)


# --- Use the evaluation model in testing --- #
net.eval() 
print('--- Testing starts! ---') 
start_time = time.time()
val_psnr, val_ssim = validation_PSNR(net, val_data_loader, device, dataset_name, save_tag=True)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))
