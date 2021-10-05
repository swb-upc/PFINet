import os

from yacs.config import CfgNode

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
from tqdm import tqdm
from Core.train_scheduler.LRScheduler import MyScheduler
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset,dataloader
import numpy as np
from configs.default import *
import random
import argparse
from Core.Models import *
from data import get_train_loader, get_test_loader
from tools.train_utils import *
import logging
import pprint

from Core.loss import TripletLoss,CMTripletLoss,CrossEntropyLabelSmooth,CenterLoss,ClusterLoss,RangeLoss,hetero_loss
class Trainer:
    def __init__(self,cfg):
        self.cfg = cfg
        self.epoch = 1
        #set logger
        self.log_dir = os.path.join('logs',cfg.dataset,cfg.model,cfg.info)
        self.checkpoint_dir = os.path.join('checkpoints',cfg.dataset,cfg.model,cfg.info)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir,exist_ok=True)
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        log_file = getCurrentFileName(self.log_dir)
        logging.basicConfig(format="%(asctime)s %(message)s",
                            filename=log_file,
                            filemode="w")

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.stream_handler)
        self.logger.info(pprint.pformat(cfg))
        #model
        if cfg.model == "PFINet":
            model = PFINet(self.cfg)

        def get_parameter_number(net):
            total_num = sum(p.numel() for p in net.parameters())
            trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
            return {'Total': total_num, 'Trainable': trainable_num}
        print('#==========模型参数量====================#')
        print(get_parameter_number(model))
        print('#========================================#')
        #Parallel training
        model = torch.nn.DataParallel(model)
        self.model = model.cuda()

        #data
        self.train_loader = get_train_loader(dataset=cfg.dataset,
                                        root=cfg.data_root,
                                        sample_method=cfg.sample_method,
                                        batch_size=cfg.batch_size,
                                        p_size=cfg.p_size,
                                        k_size=cfg.k_size,
                                        random_flip=cfg.random_flip,
                                        random_crop=cfg.random_crop,
                                        random_erase=cfg.random_erase,
                                        color_jitter=cfg.color_jitter,
                                        padding=cfg.padding,
                                        image_size=cfg.image_size,
                                        num_workers=4,
                                        RGB_to_Infrared=cfg.RGB_to_Infrared,
                                        R_channel=cfg.R_channel)
        self.gallery_loader, self.query_loader = get_test_loader(dataset=cfg.dataset,
                                                       root=cfg.data_root,
                                                       batch_size=cfg.batch_size,
                                                       image_size=cfg.image_size,
                                                       num_workers=4,
                                                       RGB_to_Infrared=cfg.RGB_to_Infrared)
        #optimizer
        assert cfg.optimizer in ['adam','sgd']
        if cfg.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),lr=cfg.lr,weight_decay=cfg.wd)
        elif cfg.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(),lr=cfg.lr,weight_decay=cfg.wd,momentum=0.9,nesterov=True)

        #resume
        if cfg.resume:
            checkpoint = torch.load(cfg.resume)
            self.model.module.load_state_dict(checkpoint['model'])
            #self.optimizer.load_state_dict(checkpoint['optimizer'])
            #self.optimizer.param_groups[0]['lr'] = self.cfg.lr
            self.epoch = checkpoint['epoch'] + 1

        #lr_scheduler
        assert cfg.lr_scheduler in ['MultiStepLR','Consine']
        if cfg.lr_scheduler == 'MultiStepLR':
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                               milestones=cfg.lr_step,
                                                               gamma=0.1)
        if cfg.lr_scheduler == "Consine":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,T_max=20,eta_min=cfg.lr*0.01)

        self.warmup_scheduler = MyScheduler(optimizer=self.optimizer,
                                                target_iteration=cfg.warmup,
                                                target_lr=cfg.lr,
                                                after_scheduler=self.lr_scheduler)

    def train(self):
        while True:
            self.model.train()
            self.warmup_scheduler(self.epoch)
            lr = self.optimizer.param_groups[0]['lr']
            t = tqdm(self.train_loader,ncols=100)
            totle_loss = 0.0
            for i,data in enumerate(t):
                images, labels, cam_ids, img_paths, img_ids = data
                
                loss, _ = self.model(images.cuda(),labels.cuda(),cam_ids.cuda())
                totle_loss += loss.detach().sum(dim=-1)
                t.set_description('epoch:%d iter:%d lr:%f loss:%f' % (self.epoch, i + 1, lr, totle_loss / (i+1)))
                self.optimizer.zero_grad()
                loss.sum(dim=-1).backward()
                self.optimizer.step()

            if self.epoch >= self.cfg.start_eval:
                if self.epoch <= self.cfg.num_epoch and self.epoch % self.cfg.eval_interval == 0:
                    self.eval(self.epoch,totle_loss / (i + 1))

            self.epoch += 1

    def eval(self,epoch,loss):
        self.model.eval()
        query_feat = []
        query_ids = []
        query_cams = []
        query_paths = []
        gallery_feat = []
        gallery_ids = []
        gallery_cams = []
        gallery_paths = []
        with torch.no_grad():
            for data in self.query_loader:
                images, labels, cam_ids, paths = data[:4]

                feat = self.model(images.cuda(),labels.cuda(),cam_ids=cam_ids.cuda())

                query_feat.append(feat.cpu())
                query_ids.append(labels)
                query_cams.append(cam_ids)
                query_paths.append(paths)
            for data in self.gallery_loader:
                torch.cuda.empty_cache()
                images, labels, cam_ids, paths = data[:4]

                feat = self.model(images.cuda(), labels.cuda(), cam_ids=cam_ids.cuda())

                gallery_feat.append(feat.cpu())
                gallery_ids.append(labels)
                gallery_cams.append(cam_ids)
                gallery_paths.append(paths)
        q_feats = torch.cat(query_feat, dim=0)
        q_ids = torch.cat(query_ids, dim=0).numpy()
        q_cams = torch.cat(query_cams, dim=0).numpy()
        q_img_paths = np.concatenate(query_paths, axis=0)
        g_feats = torch.cat(gallery_feat, dim=0)
        g_ids = torch.cat(gallery_ids, dim=0).numpy()
        g_cams = torch.cat(gallery_cams, dim=0).numpy()
        g_img_paths = np.concatenate(gallery_paths, axis=0)
        self.logger.info('===========eval:epoch-{}-loss-{}============='.format(epoch, loss))
        if self.cfg.dataset == 'sysu':
            perm = sio.loadmat(os.path.join(self.cfg.data_root, 'exp', 'rand_perm_cam.mat'))[
                'rand_perm_cam']
            mAP, r1, r5, r10, r20 = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=self.cfg.rerank)
            self.logger.info('single-All     Map:{:.4f} R1:{:.4f} R5:{:.4f} R10:{:.4f} R20:{:.4f}'.format(mAP,r1,r5,r10,r20))
            mAP, r1, r5, r10, r20 = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, rerank=self.cfg.rerank)
            self.logger.info('Multi-All      Map:{:.4f} R1:{:.4f} R5:{:.4f} R10:{:.4f} R20:{:.4f}'.format(mAP, r1, r5, r10, r20))
            mAP, r1, r5, r10, r20 = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, rerank=self.cfg.rerank)
            self.logger.info('single-indoor  Map:{:.4f} R1:{:.4f} R5:{:.4f} R10:{:.4f} R20:{:.4f}'.format(mAP, r1, r5, r10, r20))
            mAP, r1, r5, r10, r20 = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, rerank=self.cfg.rerank)
            self.logger.info('Multi-indoor   Map:{:.4f} R1:{:.4f} R5:{:.4f} R10:{:.4f} R20:{:.4f}'.format(mAP, r1, r5, r10, r20))
            save_model(self.model.module,self.optimizer,self.checkpoint_dir,epoch)
        elif self.cfg.dataset == 'regdb':
            mAP, r1, r5, r10, r20 = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=self.cfg.rerank)
            self.logger.info('infrared to visible  Map:{:.4f} R1:{:.4f} R5:{:.4f} R10:{:.4f} R20:{:.4f}'.format(mAP, r1, r5, r10, r20))
            mAP, r1, r5, r10, r20 = eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=self.cfg.rerank)
            self.logger.info('visible to infrared  Map:{:.4f} R1:{:.4f} R5:{:.4f} R10:{:.4f} R20:{:.4f}'.format(mAP, r1, r5, r10, r20))
            save_model(self.model.module, self.optimizer,self.checkpoint_dir,epoch)

if __name__ == '__main__':

    #configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int,default=0,choices=[0,1])
    parser.add_argument('--dataset',type=str,default="sysu",choices=['sysu','regdb'])
    parser.add_argument('--model',type=str,default='PFINet',choices=['PFINet'])
    parser.add_argument('--resume',default=False)
    args = parser.parse_args()

    #set random seed
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    #enable cudnn backend
    torch.backends.cudnn.benchmar = True
    #load configuration
    cfg = PFINet_cfg
    cfg.merge_from_file('configs/'+args.model+'/cfg.yml')
    cfg.model = args.model
    cfg.resume = args.resume
    cfg.dataset = args.dataset
    dataset_cfg = dataset_cfg.get(cfg.dataset)
    for k,v in dataset_cfg.items():
        cfg[k] = v
    cfg.batch_size = cfg.p_size * cfg.k_size
    if cfg.Learning_W_RGB:
        cfg.RGB_to_Infrared = False
    cfg.freeze()
    trainer = Trainer(cfg)
    if not args.test:
        trainer.train()
    else:
        trainer.eval(100,loss=0.0)


