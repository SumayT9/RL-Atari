import torch
import math

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state["step"] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p).share_memory_()
        
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                exp_avg.mul_(beta1).add_(other=grad, alpha=(1 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(tensor1=grad, tensor2=grad, value=(1 - beta2))


                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(tensor1=exp_avg, tensor2=denom, value=(-step_size))
        return loss
                

