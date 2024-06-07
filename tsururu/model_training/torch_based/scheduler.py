import torch
from torch.optim.lr_scheduler import _LRScheduler


class LRAdjScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_schedule_type, args, last_epoch=-1):
        self.lr_schedule_type = lr_schedule_type
        self.args = args
        super(LRAdjScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1

        if self.lr_schedule_type == "type1":
            lr = self.args.learning_rate * (0.5 ** ((epoch - 1) // 1))
        elif self.lr_schedule_type == "type2":
            lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
            lr = lr_adjust.get(epoch, self.base_lrs[0])
        elif self.lr_schedule_type == "3":
            lr = self.args.learning_rate if epoch < 10 else self.args.learning_rate * 0.1
        elif self.lr_schedule_type == "4":
            lr = self.args.learning_rate if epoch < 15 else self.args.learning_rate * 0.1
        elif self.lr_schedule_type == "5":
            lr = self.args.learning_rate if epoch < 25 else self.args.learning_rate * 0.1
        elif self.lr_schedule_type == "6":
            lr = self.args.learning_rate if epoch < 5 else self.args.learning_rate * 0.1
        else:
            raise ValueError(f"Unknown lr_schedule_type: {self.lr_schedule_type}")

        return [lr for _ in self.optimizer.param_groups]
