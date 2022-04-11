import math


class ScheduledOptimizer:
    def __init__(self, optimizer, config):
        self._optimizer = optimizer
        self.init_lr = config['learning_rate']
        self.num_training_steps = config['epochs'] * config['iters_per_epoch']
        self.num_warmup_steps = config['warmup_epochs'] * config['iters_per_epoch']

        self.cur_step = 0

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        self._update_learning_rate()
        self._optimizer.step()

    def state_dict(self):
        return {'cur_step': self.cur_step, 'optimizer': self._optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.cur_step = state_dict['cur_step']
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _update_learning_rate(self):
        self.cur_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr_scale(self):
        if self.cur_step < self.num_warmup_steps:
            return float(self.cur_step) / float(self.num_warmup_steps)
        progress = float(self.cur_step - self.num_warmup_steps) / \
                   float(self.num_training_steps - self.num_warmup_steps)
        return 0.5 * (1. + math.cos(math.pi * progress))
