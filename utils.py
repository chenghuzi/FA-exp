import colorsys

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch

import mi


def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def model2condition(model):
    return (model.dw_decoder_on, model.feedback_option)


def errorfill(x,
              y,
              yerr,
              color=None,
              alpha_fill=0.25,
              ax=None,
              linestyle='-',
              label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, linestyle=linestyle, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def draw_loss(res, figsize=(8, 4), fname="exp1.png"):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].set_title("dW On")
    axes[1].set_title("dW Off")
    for cond in res.keys():
        turn_on_dw2, feedback_option = cond
        if turn_on_dw2:
            ax = axes[0]
        else:
            ax = axes[1]

        epoch_loss, metric_list = res[cond][0]
        epochs = np.arange(len(epoch_loss))

        epoch_loss_mean = np.array(
            [np.mean(loss_list) for loss_list in epoch_loss])
        epoch_loss_std = np.array(
            [np.std(loss_list) for loss_list in epoch_loss])
        errorfill(
            epochs,
            epoch_loss_mean,
            epoch_loss_std,
            ax=ax,
            label=feedback_option,
        )
        ax.legend()
    plt.tight_layout()
    plt.savefig(fname)


class GaussianXYDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys) -> None:
        super().__init__()
        assert len(xs) == len(ys)
        self.xs = torch.from_numpy(xs).type(torch.FloatTensor)
        self.ys = torch.from_numpy(ys).type(torch.FloatTensor)

    def __len__(self, ):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


def g_reg_H(H_, Y_):
    assert H_.shape[0] == Y_.shape[0]

    eye = np.eye(Y_.shape[0])

    ALPHA = np.linalg.inv(H_.T @ H_)
    BETA = H_ @ ALPHA @ H_.T

    Q = Y_.T @ H_ @ ALPHA
    P = -2 * (Y_.T @ (eye - BETA) @ H_ @ ALPHA + Q)

    a = eye - BETA
    b = 2 * BETA @ (eye - BETA)

    second_term = b @ Y_ @ Q

    g = a @ Y_ @ P + second_term

    g_approx = a @ Y_ @ P

    info = {
        "a": a,
        "b": b,
        ####
        "ALPHA": ALPHA,
        "BETA": BETA,
        ####
        "P": P,
        "Q": Q,
        "second_term": second_term,
        "g": g,
        "g_approx": g_approx,
    }
    return g, info


class XORDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xs = torch.from_numpy(np.array([
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 1],
        ])).type(torch.FloatTensor)
        self.ys = torch.from_numpy(np.array([
            -1,
            -1,
            +1,
            +1,
        ])).type(torch.FloatTensor)

    def __len__(self, ):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


def get_norm(x):
    return np.sqrt((x**2).sum())


def get_degree(x, y, eps=1e-18):
    dp = x.dot(y)
    cos_v = dp / (get_norm(x) * get_norm(y) + eps)
    rad = np.arccos(np.clip(cos_v, -1, 1))
    degree = 180. * rad / np.pi
    return degree


def get_itm_info(H, Y, g_H, entropy_binsize=0.1):

    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    res = {}
    # get the residual
    if np.linalg.det(H.T @ H) == 0:
        residual = 9999
        angle = 90
    else:
        theta = np.linalg.inv(H.T @ H) @ H.T @ Y
        err = H @ theta - Y
        residual = err.T @ err

        g_H_regression, g_H_regression_info = g_reg_H(H, Y)
        angle = get_degree(
            -g_H_regression_info['g'].reshape(-1),
            g_H.reshape(-1),
        )

    # mi.binned_entropy(x, binsize=entropy_binsize)
    mi_hy = mi.binned_mi(H, Y, entropy_binsize)

    res['mi_hy'] = mi_hy
    res['angle'] = angle
    res['residual'] = residual

    return res


def extract_intervals(sign_list):
    if len(sign_list) <= 2:
        return []

    signf_t_idxes = np.argwhere(sign_list == True).reshape(-1)
    if len(signf_t_idxes) <= 2:
        return []
    intervals = []

    interval_left = signf_t_idxes[0]
    interval_right = None
    for t_post_idx, is_continuous in enumerate(signf_t_idxes[1:] -
                                               signf_t_idxes[:-1] == 1):

        if is_continuous == True:
            interval_right = signf_t_idxes[t_post_idx + 1]
        else:
            intervals.append((interval_left, interval_right))
            interval_left = signf_t_idxes[t_post_idx + 1]
            interval_right = None
    if is_continuous == True:
        intervals.append((interval_left, interval_right))

    return intervals