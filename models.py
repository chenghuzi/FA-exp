from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb


def infinite_deformed_sin(x):
    return x + torch.sin(x)


def infinite_quarter_circle(x):
    base = torch.floor(x)
    evens = torch.sqrt(1 - (torch.ceil(x) - x)**2)
    odds = 1 - torch.sqrt(1 - (x - torch.floor(x))**2)
    return torch.floor(x) + (base % 2 == 0) * evens + (base % 2 == 1) * odds


class FaLinearFunction(torch.autograd.Function):
    def forward(ctx,
                x,
                weight,
                bias,
                fa,
                fa1,
                fa2,
                fai,
                feedback_option="BP",
                nfa_amp=0,
                dw_this_layer=True):

        ctx.feedback_option = feedback_option
        ctx.dw_this_layer = dw_this_layer
        ctx.fai = fai
        ctx.nfa_amp = nfa_amp

        output = x.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(x, weight, bias, fa, fa1, fa2,
                              output.data.detach())

        return output

    def backward(ctx, grad_output):
        x, weight, bias, fa, fa1, fa2, output = ctx.saved_tensors

        # calculate error signals for the input node
        if ctx.needs_input_grad[0]:
            # Back Prop (BP)
            if ctx.feedback_option == "BP":
                grad_input = grad_output.mm(weight)
            # Feedback Alignment (FA)
            elif ctx.feedback_option == "FA":
                grad_input = grad_output.mm(fa)
            # Target Feedback
            elif ctx.feedback_option == "TF":
                # ipdb.set_trace()
                # ˝target = output - grad_output
                # grad_output = grad_output - grad_output.mean(
                #     dim=-1).unsqueeze(1)
                grad_input = grad_output.mm(fa)
                grad_input = grad_input - grad_input.mean(dim=0).unsqueeze(0)

            # Noisy Feedback Alignment
            elif ctx.feedback_option == "NFA":
                # Inject noise into the feedback matrix.
                grad_input = grad_output.mm(fa + ctx.nfa_amp *
                                            (torch.rand_like(fa) - 0.5))

            # use a Feedback Network(NF) to replace B matrix in Feedback alignment
            elif ctx.feedback_option == "NF":
                # now this is just a simple mapping but it could be changed later.
                _h = torch.tanh(grad_output.mm(fa1))
                grad_input = _h.mm(fa2)

            # Just Feedback Scale(FS) the original error
            elif ctx.feedback_option == "FS":
                fsm = torch.ones_like(fa).to(fa.device)
                grad_input = grad_output.mm(fsm)
                # normalize
                grad_input /= ((grad_input**2).sum(-1).sqrt().unsqueeze(1) +
                               1e-10)
                scales = (grad_output**2).sum(-1).sqrt().unsqueeze(1)
                grad_input *= scales * 0.9

                # next we need to control the amplitude

            # this one aims to maximize the mutual information
            elif ctx.feedback_option == "FI":

                def gen_loss(fm):
                    grad_in_normalized = grad_output.matmul(fm)

                    grad_in_normalized = (
                        grad_in_normalized - grad_in_normalized.min()
                    ) / (grad_in_normalized.max() - grad_in_normalized.min())
                    grad_in_normalized = grad_in_normalized - grad_in_normalized.mean(
                        dim=0).unsqueeze(0)

                    cov_grad_in = grad_in_normalized.T.mm(grad_in_normalized)
                    loss = torch.log(torch.linalg.det(cov_grad_in))
                    return loss

                norm_fai = (ctx.fai['fai'].data**2).sum().sqrt()

                perturbation = torch.autograd.functional.jacobian(
                    gen_loss, ctx.fai['fai'].data)
                norm_perturbation = (perturbation**2).sum().sqrt()
                perturbation = perturbation * norm_fai / norm_perturbation

                ctx.fai['fai'].data = ctx.fai[
                    'fai'].data + perturbation * ctx.fai['lr']
                grad_input = grad_output.mm(ctx.fai['fai'].data)

            # feedback random
            elif ctx.feedback_option == "FR":
                norm_fai = (ctx.fai['fai'].data**2).sum().sqrt()

                perturbation = torch.normal(
                    torch.zeros_like(ctx.fai['fai'].data),
                    torch.ones_like(ctx.fai['fai'].data))
                norm_perturbation = (perturbation**2).sum().sqrt()
                perturbation = perturbation * norm_fai / norm_perturbation

                ctx.fai['fai'].data = (1 - ctx.fai['lr']) * ctx.fai[
                    'fai'].data + perturbation * ctx.fai['lr']
                grad_input = grad_output.mm(ctx.fai['fai'].data)
        else:
            grad_input = None

        # calculate error signals for parameter, i.e., weights here
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(x)
        else:
            grad_weight = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        else:
            grad_bias = None

        # whether to turn on learning or not
        # print(ctx.feedback_option, ctx.dw_this_layer)
        if ctx.dw_this_layer:
            grad_weight = grad_weight
        else:
            grad_weight = None

        true_grads = (grad_input, grad_weight, grad_bias)

        null_grads = tuple((None for _ in range(7)))

        return true_grads + null_grads


class FALinear(torch.nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        dw_this_layer=True,
        feedback_option="BP",
        feedback_lr=5e-1,
        nfa_amp=0,
    ):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.nfa_amp = nfa_amp

        self.weight = torch.nn.Parameter(
            torch.empty(output_features, input_features))
        torch.nn.init.kaiming_normal_(self.weight)
        # Feedback alignment
        self.fa = torch.nn.Parameter(torch.empty(output_features,
                                                 input_features),
                                     requires_grad=False)
        torch.nn.init.kaiming_normal_(self.fa)

        # self.fap = torch.nn.Parameter(torch.empty(output_features,
        #                                           input_features),
        #                               requires_grad=False)
        # torch.nn.init.kaiming_normal_(self.fap)

        # Feedback Network
        m = 50
        self.fa1 = torch.nn.Parameter(torch.empty(output_features, m),
                                      requires_grad=False)
        self.fa2 = torch.nn.Parameter(torch.empty(m, input_features),
                                      requires_grad=False)
        torch.nn.init.kaiming_normal_(self.fa1)
        torch.nn.init.kaiming_normal_(self.fa2)

        # Feedback information
        self.fai = torch.nn.Parameter(torch.empty(output_features,
                                                  input_features),
                                      requires_grad=True)
        torch.nn.init.kaiming_normal_(self.fai)

        # self.fai = torch.nn.Linear(output_features, input_features)
        self.feedback_lr = feedback_lr

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter("bias", None)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -0.1, 0.1)

        self.dw_this_layer = dw_this_layer
        # target feedback is equvalent to FA in terms of propagating errors
        # if feedback_option == "TF":
        #     self.feedback_option = "FA"
        # else:
        #     pass
        self.feedback_option = feedback_option

    def forward(self, input):
        return FaLinearFunction.apply(
            input,
            self.weight,
            self.bias,
            self.fa,
            self.fa1,
            self.fa2,
            {
                "fai": self.fai,
                "lr": self.feedback_lr,
            },
            self.feedback_option,
            self.nfa_amp,
            self.dw_this_layer,
        )

    def extra_repr(self):
        return "input_features={}, output_features={}, bias={}".format(
            self.input_features, self.output_features, self.bias is not None)


class ConvBackbone(torch.nn.Module):
    def __init__(self, n_hidden=120) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.to_hidden = nn.LazyLinear(self.n_hidden)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        return self.to_hidden(x)


class NnConstructor:
    # @classmethod
    @staticmethod
    def construct_sigmoid():
        return lambda: torch.nn.Sigmoid()

    @staticmethod
    def construct_softmax():
        return lambda: torch.nn.Softmax(-1)

    # @classmethod
    @staticmethod
    def construct_hardtanh(min_val=-1, max_val=1):
        return lambda: torch.nn.Hardtanh(min_val=min_val, max_val=max_val)

    @staticmethod
    def construct_bounded_tanh(scale=1):
        return lambda: (lambda x: torch.tanh(x) * scale)

    # @classmethod
    @staticmethod
    def construct_conv_backbone(d_hidden):
        return lambda: ConvBackbone(d_hidden)

    @staticmethod
    def construct_fixpoint():
        return lambda: (lambda x: x)


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1, fix=True):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))
        if fix:
            self.scale.requires_grad = False

    def forward(self, input):
        return input * self.scale


class FADeepFullNet(nn.Module):
    def __init__(
        self,
        d_i,
        d_o,
        d_hiddens,
        feedback_option,
        dw_decoder_on,
        output_act_f: Callable,
        backbone_net_f: Optional[Callable] = None,
        feedback_lr=1e-4,
        fa_output_bias=True,
        allow_scale_after_act=False,
        nfa_amp=0,
    ) -> None:
        super().__init__()
        self.dw_decoder_on = dw_decoder_on
        self.feedback_option = feedback_option

        self.dims_encoder = [d_i] + [int(dh) for dh in d_hiddens.split("-")]
        self.len_dims_encoder = len(self.dims_encoder) - 1
        layers = {}
        maxv = 1
        for layer_idx, (din, dout) in enumerate(
                zip(self.dims_encoder[:-1], self.dims_encoder[1:])):
            layers[f"layer-{layer_idx}"] = nn.Sequential(
                nn.Linear(din, dout),
                nn.Tanh(),
                ScaleLayer(maxv, fix=not allow_scale_after_act),
            )
            # if layer_idx == self.len_dims_encoder - 1:
            #     layers[f"layer-{layer_idx}"] = nn.Sequential(
            #         nn.Linear(din, dout),
            #         nn.Tanh(),
            #         ScaleLayer(maxv, fix=not allow_scale_after_act),
            #     )
            # else:
            #     layers[f"layer-{layer_idx}"] = nn.Sequential(
            #         nn.Linear(din, dout),
            #         nn.Tanh(),
            #     )

        if backbone_net_f:
            self.backbone_net = backbone_net_f()
        else:
            self.backbone_net = None

        self.encoder_layers = nn.ModuleDict(layers)
        # self.batch_normalization = nn.BatchNorm1d(dout)
        self.decoder = FALinear(
            self.dims_encoder[-1],
            d_o,
            dw_this_layer=dw_decoder_on,
            feedback_option=feedback_option,
            feedback_lr=feedback_lr,
            bias=fa_output_bias,
            nfa_amp=nfa_amp,
        )

        self.decoder_act = output_act_f()

    def encode(self, x, retain_act_grad=False):
        if self.backbone_net:
            x = self.backbone_net(x)
        encoder_activity = [x]

        for layer_idx in self.encoder_layers.keys():
            encoder_activity.append(self.encoder_layers[layer_idx](
                encoder_activity[-1]))
            if retain_act_grad:
                encoder_activity[-1].retain_grad()

        return encoder_activity[-1], encoder_activity

    def decode(self, x):
        return self.decoder_act(self.decoder(x))

    def clear_decoder_grads(self):
        for param in self.decoder.parameters():
            param.grad = torch.zeros_like(param.data)

    def clear_encoder_grads(self):
        if self.backbone_net:
            for param in self.backbone_net.parameters():
                param.grad = torch.zeros_like(param.data)
        for param in self.encoder_layers.parameters():
            param.grad = torch.zeros_like(param.data)

    def forward(self, x, retain_act_grad=False):
        # if self.backbone_net:
        #     x = self.backbone_net(x)
        # encoder_activity = [x]
        # # encoding
        # for layer_idx in self.encoder_layers.keys():
        #     encoder_activity.append(self.encoder_layers[layer_idx](
        #         encoder_activity[-1]))

        #     if retain_act_grad:
        #         encoder_activity[-1].retain_grad()
        # # decoding

        encoded, encoder_activity = self.encode(
            x,
            retain_act_grad=retain_act_grad,
        )
        encoder_activity.append(self.decode(encoded))
        return encoder_activity[-1], encoder_activity

    def get_grad(self):
        if self.decoder.weight.grad is None:
            decoder_grad = np.zeros(self.decoder.weight.data.shape).reshape(-1)
        else:
            decoder_grad = self.decoder.weight.grad.data.cpu().numpy().reshape(
                -1)

        return {"decoder": decoder_grad}


class FAConvNet(torch.nn.Module):
    def __init__(
        self,
        d_out,
        n_hidden,
        dw_fa_layer=True,
        feedback_option="NF",
        output_bias=False,
    ):
        super().__init__()
        self.feedback_option = feedback_option
        self.dw_fa_layer = dw_fa_layer
        self.n_hidden = n_hidden

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.lin1 = torch.nn.LazyLinear(self.n_hidden)
        self.lin2 = FALinear(
            n_hidden,
            d_out,
            dw_this_layer=dw_fa_layer,
            feedback_option=feedback_option,
            bias=output_bias,
        )

        self.act = torch.nn.Tanh()

    def forward(self, x, retain_act_grad=False):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)

        h = self.act(self.lin1(x))
        if retain_act_grad:
            h.retain_grad()

        x = self.lin2(h)
        return x, {"h": h}

    def get_grad(self):
        ipdb.set_trace()
        pass


class FA2LayerNet(torch.nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 n_hidden,
                 dw_fa_layer=True,
                 feedback_option="NF",
                 output_bias=False):
        super().__init__()
        self.feedback_option = feedback_option
        self.dw_fa_layer = dw_fa_layer

        self.lin1 = torch.nn.Linear(d_in, n_hidden)
        self.lin2 = FALinear(
            n_hidden,
            d_out,
            dw_this_layer=dw_fa_layer,
            feedback_option=feedback_option,
            bias=output_bias,
        )

        self.act = torch.nn.Tanh()

    @property
    def name(self):
        return type(
            self
        ).__name__ + f"-dw_fa_layer:{self.dw_fa_layer}, feedback_option:{self.feedback_option}"

    def forward(self, x, retain_act_grad=False):
        h = self.act(self.lin1(x))

        if retain_act_grad:
            h.retain_grad()

        y = self.lin2(h)
        y.retain_grad()
        return y, {"h": h}

    def get_grad(self):
        return {"lin1": self.lin1.weight.grad.data.cpu().numpy().reshape(-1)}