from torch.optim.lr_scheduler import _LRScheduler,StepLR


class MyScheduler(object):
    def __init__(self,optimizer,target_iteration,target_lr,init_lr=None,after_scheduler=None):
        self.optimizer = optimizer
        self.target_iteration =  target_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.curren_lr = 0.0
        if init_lr:
            self.init_lr = init_lr
        else:
            self.init_lr = target_lr * 0.1
    def __call__(self, epoch):
        if epoch <= self.target_iteration:
            self.warmup_lr(epoch)
        else:
            if self.after_scheduler is not None:
                self.after_scheduler.step(epoch - self.target_iteration)

    def warmup_lr(self,epoch):
        lr = self.init_lr + (epoch-1) * (self.target_lr - self.init_lr) / self.target_iteration
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    def get_lr(self):
        return self.curren_lr
