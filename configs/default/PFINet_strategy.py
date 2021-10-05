from yacs.config import CfgNode

strategy_cfg = CfgNode()
strategy_cfg.dataset =  'sysu'
strategy_cfg.model = 'Transformer'
strategy_cfg.info = 'pure Transformer'
strategy_cfg.RGB_to_Infrared = False
strategy_cfg.R_channel = False

strategy_cfg.fp16 = False
strategy_cfg.rerank = False

#model
        #L = R * 299/1000 + G * 587/1000 + B * 114/1000
strategy_cfg.w_RGB = [[[0.299]],[[0.587]],[[0.144]]]
strategy_cfg.Learning_W_RGB = False


# dataset
strategy_cfg.sample_method = 'identity_random'
strategy_cfg.image_size = [256,128]
strategy_cfg.batch_size = 16
strategy_cfg.p_size = 4
strategy_cfg.k_size = 4


# optimizer
strategy_cfg.lr = 0.00035
strategy_cfg.optimizer = 'adam'
strategy_cfg.num_epoch = 1000
strategy_cfg.lr_step = [200, 400]
strategy_cfg.lr_scheduler = 'MultiStepLR'
strategy_cfg.warmup = 0
strategy_cfg.wd = 5e-4


# augmentation
strategy_cfg.random_flip = True
strategy_cfg.random_crop = True
strategy_cfg.random_erase = False
strategy_cfg.color_jitter = False
strategy_cfg.padding = 10

# train epochs
strategy_cfg.log_period = 150
strategy_cfg.start_eval = 10
strategy_cfg.eval_interval = 10
strategy_cfg.mutual_learning = True

#loss
strategy_cfg.Triplet_loss = True
strategy_cfg.center_loss = False
strategy_cfg.cluster_loss = False
strategy_cfg.CMTriplet_loss = False
strategy_cfg.hetero_loss = True
strategy_cfg.id_loss = True
strategy_cfg.range_loss = False
strategy_cfg.CenterTripletLoss = False
strategy_cfg.Triplet_margin = 0.3
strategy_cfg.CMTriplet_margin = 0.3
strategy_cfg.CenterTriplet_margin = 0.7
strategy_cfg.CircleLoss = True

#output
strategy_cfg.feature_dim = 2048


#resume
strategy_cfg.resume = False

