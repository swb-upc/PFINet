import os
import time

import torch
from natsort import natsorted


def getCurrentFileName(dir):
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    return dir+'/'+t+'_log.txt'

def save_model(model,optimizer,dir,epoch):
    # checkpoint_files = natsorted(os.listdir(dir))
    # if len(checkpoint_files) >=10:
    #     c = checkpoint_files.pop(0)
    #     os.remove(dir+'/'+c)
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}, dir+'/checkpoint_{}.pth.tar'.format(epoch))