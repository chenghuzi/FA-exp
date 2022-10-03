import itertools
import pickle
from collections import defaultdict

import ipdb  # noqa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import trange

from models import FA2LayerNet, FAConvNet
from utils import GaussianXYDataset, errorfill, get_itm_info


def draw_curve(info, ylabel, order, fname, figsize=(8, 4)):
    conds = list(info[0].keys())
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    axes[0].set_title("$\Delta W_o$ On")
    axes[1].set_title("$\Delta W_o$ Off")
    axes[0].set_ylabel(ylabel)

    for cond in conds:
        turn_on_dw2, feedback_option = cond
        if turn_on_dw2:
            ax = axes[0]
        else:
            ax = axes[1]

        curves = []
        for info_trial in info:
            curves.append(info_trial[cond][order].reshape(1, -1))
            pass
        curves = np.concatenate(curves, axis=0)  # repeat, epochs
        epochs = np.arange(curves.shape[1]) + 1

        epoch_loss_mean = curves.mean(axis=0)
        epoch_loss_std = curves.std(axis=0)
        errorfill(
            epochs,
            epoch_loss_mean,
            epoch_loss_std,
            ax=ax,
            label=feedback_option,
        )
        ax.legend()
        ax.set_xlabel("Epochs")
    plt.tight_layout()
    plt.savefig(fname)


def draw_loss(info, fname="loss.png", figsize=(8, 4)):
    draw_curve(info, "Loss", 0, fname, figsize=figsize)


def draw_mi(info, fname="mi.png", figsize=(8, 4)):
    draw_curve(info, "$I(H;Y)$", 2, fname, figsize=figsize)


def draw_angle(info, fname="angle.png", figsize=(8, 4)):
    draw_curve(info, "Angle", 3, fname, figsize=figsize)


def train_model(
    model,
    n_epochs,
    loss_func,
    device,
    learning_rate,
    momentum,
    train_data_loader,
    metric_func,
    batch_interval=3,
    cal_itm_on_epoch=True,
):

    model.to(device=device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )
    loss_epochs = []
    metric_epochs = []
    itm_epochs = []

    grad_h_epochs = []
    act_h_epochs = []
    targets_epochs = []

    pbar = trange(n_epochs)
    for epoch in pbar:

        grad_h_list = []
        act_h_list = []
        targets_list = []

        loss_list = []
        metric_list = []
        model.train()
        optimizer.zero_grad()
        for batch_idx, (data, target) in enumerate(train_data_loader):
            target = target.to(device)
            output, info = model(data.to(device), retain_act_grad=True)

            # if (model.dw_fa_layer, model.feedback_option) == (False, "FA"):
            # if model.feedback_option == "FA":
            #     # output.grad.data.mm(model.lin2.fa.data).data.cpu().numpy()-info['h'].grad.data.cpu().numpy()
            #     loss = loss_func(output.data.detach(), target)
            #     output.backward(0.1 * loss * target)
            #     # ipdb.set_trace()

            # else:
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

            grad_h_list.append(info['h'].grad.data.cpu().numpy())
            act_h_list.append(info['h'].data.cpu().numpy())
            targets_list.append(target.data.cpu().numpy())

            metric = metric_func(output, target)
            loss_list.append(loss.item())
            metric_list.append(metric.item())

            if batch_idx % batch_interval == 0:
                pbar.set_description(
                    f"Epoch {epoch+1:5d} Loss={np.mean(loss_list):2.4f}, Metric={np.mean(metric_list):2.2f}"
                )

        loss_epochs.append(loss_list)
        metric_epochs.append(metric_list)

        grad_h_epochs.append(grad_h_list)
        act_h_epochs.append(act_h_list)
        targets_epochs.append(targets_list)

        if cal_itm_on_epoch:
            grad_h_list = np.concatenate(grad_h_list, axis=0)
            act_h_list = np.concatenate(act_h_list, axis=0)
            targets_list = np.concatenate(targets_list, axis=0)
            itm_epochs.append(
                [get_itm_info(act_h_list, targets_list, grad_h_list)])

        else:
            itm = []
            for h_, y_, g_h_ in zip(act_h_list, targets_list, grad_h_list):
                itm.append(get_itm_info(h_, y_, g_h_))
            itm_epochs.append(itm)

    return loss_epochs, metric_epochs, itm_epochs, act_h_epochs, grad_h_epochs, targets_epochs


def train_on_gaussian(device, n_epochs=10):
    res_gaussian = {}
    batch_size = 256
    n_batch = 10
    dim_x = 20
    dim_y = 1

    train_loader = torch.utils.data.DataLoader(
        GaussianXYDataset(
            np.random.normal(0, 1, (n_batch * batch_size, dim_x)),
            np.random.normal(0, 1, (n_batch * batch_size, dim_y))),
        batch_size=batch_size,
        shuffle=False,
    )

    turn_on_dw2s = (True, False)
    feedback_options = ["BP", "FA"]
    conditions = itertools.product(turn_on_dw2s, feedback_options)

    for condition in conditions:
        print(condition)
        turn_on_dw2, feedback_option = condition

        loss_func = torch.nn.MSELoss()

        n_hidden = 64
        learning_rate = 1e-3
        momentum = 0.1

        model = FA2LayerNet(
            dim_x,
            dim_y,
            n_hidden,
            dw_fa_layer=turn_on_dw2,
            feedback_option=feedback_option,
        )

        def _get_acc(output, target):
            return torch.nn.functional.mse_loss(output.data, target)

        loss_epochs, metric_epochs, itm_epochs, act_h_epochs, _, _ = train_model(
            model,
            n_epochs,
            loss_func,
            device,
            learning_rate,
            momentum,
            train_loader,
            _get_acc,
            cal_itm_on_epoch=False,
        )
        res_gaussian[condition] = (
            loss_epochs,
            metric_epochs,
            itm_epochs,
            act_h_epochs,
        )

    return res_gaussian


def train_on_mnist(device, n_epochs=5):
    res_mnist = {}
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/',
                                   train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307, ), (0.3081, ))
                                   ])),
        batch_size=1024,
        shuffle=False,
    )
    turn_on_dw2s = (True, False)
    feedback_options = ["BP", "FA"]
    conditions = itertools.product(turn_on_dw2s, feedback_options)

    for condition in conditions:
        print(condition)
        turn_on_dw2, feedback_option = condition
        loss_func = torch.nn.CrossEntropyLoss()
        d_out = 10
        n_hidden = 128
        learning_rate = 1e-3
        momentum = 0.1

        model = FAConvNet(
            d_out,
            n_hidden,
            dw_fa_layer=turn_on_dw2,
            feedback_option=feedback_option,
        )

        def _get_acc(output, target):
            pred = torch.argmax(output, dim=-1)
            return (pred == target).type(torch.FloatTensor).mean()

        loss_epochs, metric_epochs, itm_epochs, act_h_epochs, _, _ = train_model(
            model,
            n_epochs,
            loss_func,
            device,
            learning_rate,
            momentum,
            train_loader,
            _get_acc,
            cal_itm_on_epoch=False,
        )
        res_mnist[condition] = (
            loss_epochs,
            metric_epochs,
            itm_epochs,
            act_h_epochs,
        )
    return res_mnist


def post_processing(res):
    turn_on_dw2s = (True, False)
    feedback_options = ["BP", "FA"]
    conditions = itertools.product(turn_on_dw2s, feedback_options)

    info = {}
    for condition in conditions:

        loss_epochs = np.array(res[condition][0])
        loss_curve = loss_epochs.mean(axis=1)

        metric_epochs = np.array(res[condition][1])
        metric_curve = metric_epochs.mean(axis=1)

        itm_epochs = res[condition][2]
        mi_hys = []
        angles = []
        residuals = []
        for itm in itm_epochs:
            mi_hys.append(np.mean([itm_single['mi_hy'] for itm_single in itm]))
            angles.append(np.mean([itm_single['angle'] for itm_single in itm]))
            residuals.append(
                np.mean([itm_single['residual'] for itm_single in itm]))
        mi_hys_curve = np.array(mi_hys)
        angles_curve = np.array(angles)
        residuals_curve = np.array(residuals)

        info[condition] = (loss_curve, metric_curve, mi_hys_curve,
                           angles_curve, residuals_curve)

    return info


if __name__ == '__main__':
    seed = 12
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda'

    # gaussian
    n_repeat = 5
    info_gaussians = []
    for _idx in range(n_repeat):
        res_gaussian = train_on_gaussian(device)
        info_gaussians.append(post_processing(res_gaussian))

    draw_loss(info_gaussians, fname="figs/exp1_gussian_loss.png")
    draw_mi(info_gaussians, fname="figs/exp1_gussian_mi.png")
    draw_angle(info_gaussians, fname="figs/exp1_gussian_angle.png")

    # # mnist
    # n_repeat = 10
    # info_mnist = []
    # for _idx in range(n_repeat):
    #     res_mnist = train_on_mnist(device)
    #     info_mnist.append(post_processing(res_mnist))

    # draw_loss(info_mnist, fname="figs/exp1_mnist_loss.png")
    # draw_mi(info_mnist, fname="figs/exp1_mnist_mi.png")
    # draw_angle(info_mnist, fname="figs/exp1_mnist_angle.png")
