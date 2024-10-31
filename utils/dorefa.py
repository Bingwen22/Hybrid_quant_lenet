import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def uniform_quantize(k):
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    class QuantizeFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if k == 32:
                output = input
            elif k == 1:
                output = torch.sign(input)
            else:
                max_value = float(2 ** k - 1)
                output = torch.round(input * max_value) / max_value
            return output

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input

    return QuantizeFunction.apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32, "w_bit must be <= 8 or == 32"
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x  # No quantization for 32-bit
        elif self.w_bit == 1:
            epsilon = 1e-6  # Small constant to avoid division by zero
            E = torch.mean(torch.abs(x)).detach() + epsilon
            weight_q = self.uniform_q(x / E) * E  # Binary quantization
        else:
            max_w = torch.max(torch.abs(torch.tanh(x))).detach()
            weight = torch.tanh(x) / 2 / max_w + 0.5
            weight_q = max_w * (2 * self.uniform_q(weight) - 1)

        return weight_q


class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        assert a_bit <= 8 or a_bit == 32, "a_bit must be <= 8 or == 32"
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k=a_bit)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            activation_q = self.uniform_q(torch.clamp(x, 0, 1))
        return activation_q


def conv2d_Q_fn(w_bit, a_bit):
    class Conv2d_Q(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=True):
            super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               padding, dilation, groups, bias)
            self.w_bit = w_bit
            self.a_bit = a_bit
            self.w_quantize_fn = weight_quantize_fn(w_bit=w_bit)
            self.a_quantize_fn = activation_quantize_fn(a_bit=a_bit)

        def forward(self, input):
            weight_q = self.w_quantize_fn(self.weight)
            activation_q = self.a_quantize_fn(input)
            return F.conv2d(activation_q, weight_q, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    return Conv2d_Q


def linear_Q_fn(w_bit, a_bit):
    class Linear_Q(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super(Linear_Q, self).__init__(in_features, out_features, bias)
            self.w_bit = w_bit
            self.a_bit = a_bit
            self.w_quantize_fn = weight_quantize_fn(w_bit=w_bit)
            self.a_quantize_fn = activation_quantize_fn(a_bit=a_bit)

        def forward(self, input):
            weight_q = self.w_quantize_fn(self.weight)
            activation_q = self.a_quantize_fn(input)
            return F.linear(activation_q, weight_q, self.bias)

    return Linear_Q
