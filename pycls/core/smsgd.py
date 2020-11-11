import torch
from torch.optim.optimizer import Optimizer, required
import copy


class SMSGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, sm=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, sm=sm)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SMSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SMSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            sm = group['sm']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if p.dim() == 4:
                    if p.size(2) == 3 and p.size(3) == 3:
                        # center surround function
                        d_p = self.center_surround(p, d_p, sm)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss

    @torch.no_grad()
    def center_surround(self, p, d_p, weight_decay):

        p_center = torch.nn.functional.pad(p[:, :, 1:2, 1:2], (1, 1, 1, 1))
        p_surround = p - p_center

        d_p_center = torch.nn.functional.pad(d_p[:, :, 1:2, 1:2], (1, 1, 1, 1))
        d_p_surround = d_p - d_p_center

        d_p_center = d_p_center.add(-torch.nn.functional.relu(-p_center), alpha=weight_decay)
        d_p_surround = d_p_surround.add(torch.nn.functional.relu(p_surround), alpha=weight_decay/8)

        d_p = d_p_center + d_p_surround

        return d_p