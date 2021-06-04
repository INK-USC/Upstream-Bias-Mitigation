from torch.autograd import Function
from torch.nn import Module

def get_rev_grad_func(grad_rev_strength):
    class _RevGradFunc(Function):
        @staticmethod
        def forward(ctx, input_):
            ctx.save_for_backward(input_)
            output = input_
            return output

        @staticmethod
        def backward(ctx, grad_output):  # pragma: no cover
            grad_input = None
            if ctx.needs_input_grad[0]:
                grad_input = - grad_rev_strength * grad_output
            return grad_input
    revgrad = _RevGradFunc.apply
    return revgrad

class RevGrad(Module):
    def __init__(self, grad_rev_strength, *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """

        super().__init__(*args, **kwargs)
        self.grad_rev_strength = grad_rev_strength
        self.rev_grad_func = get_rev_grad_func(grad_rev_strength)

    def forward(self, input_):
        output = self.rev_grad_func(input_)
        return output

