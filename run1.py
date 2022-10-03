"For fig.1"

import itertools
import time

import click
import ipdb  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pathos.multiprocessing as pmp
import sklearn.linear_model as linear_model
import torch
from tqdm import trange
import mi
from models import FADeepFullNet
from utils import errorfill
from tasks import MNISTTask, IBTask, IBRegTask, NnConstructor
from training import train_deep_full_net_worker
device = "cuda"


def train_task(task, cores=6, n_repeats=3, seed_base=None):
    results = {}
    feedback_options = ("FA", "BP")
    dw_on_outputs = (False, True)

    conditions = list(itertools.product(feedback_options, dw_on_outputs))

    trials = []
    return_reg_related = True
    allow_scale_after_act = False
    for feedback_option, dw_on_output in conditions:
        print(feedback_option, dw_on_output)
        for repeat_idx in range(n_repeats):
            trials.append((
                task,
                device,
                return_reg_related,
                seed_base,
                (
                    feedback_option,
                    dw_on_output,
                    allow_scale_after_act,
                    repeat_idx,
                ),
            ))

    # debug
    if cores == 1:
        res_all_list = list(map(train_deep_full_net_worker, trials))
    else:
        with pmp.Pool(cores) as p:
            res_all_list = p.map(train_deep_full_net_worker, trials)
    # results = dict(results)
    for cond_key in conditions:
        results[cond_key] = [
            res[1] for res in res_all_list if res[0][:2] == cond_key
        ]

    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(2, 2, figsize=[10, 8], sharex=True, sharey='row')
    for feedback_option, dw_on_output in conditions:
        if dw_on_output is True:
            ax_loss = axes[0, 0]
            ax_metric = axes[1, 0]
            # ax_mi = axes[2, 0]
        else:
            ax_loss = axes[0, 1]
            ax_metric = axes[1, 1]
            # ax_mi = axes[2, 1]

        loss_list = np.array(
            [r[0] for r in results[(feedback_option, dw_on_output)]])
        metric_list = np.array(
            [r[1] for r in results[(feedback_option, dw_on_output)]])
        i_hy_list = np.array(
            [r[2] for r in results[(feedback_option, dw_on_output)]])

        loss_reg_list = np.array(
            [r[3] for r in results[(feedback_option, dw_on_output)]])
        metric_reg_list = np.array(
            [r[4] for r in results[(feedback_option, dw_on_output)]])

        # loss
        errorfill(
            np.arange(loss_list.shape[1]) + 1,
            loss_list.mean(axis=0),
            loss_list.std(axis=0),
            ax=ax_loss,
            label=feedback_option,
        )
        errorfill(
            np.arange(loss_reg_list.shape[1]) + 1,
            loss_reg_list.mean(axis=0),
            loss_reg_list.std(axis=0),
            ax=ax_loss,
            linestyle='dashed',
            label=feedback_option + "(reg)",
        )
        # metric
        errorfill(
            np.arange(metric_list.shape[1]) + 1,
            metric_list.mean(axis=0),
            metric_list.std(axis=0),
            ax=ax_metric,
            label=feedback_option,
        )
        errorfill(
            np.arange(metric_reg_list.shape[1]) + 1,
            metric_reg_list.mean(axis=0),
            metric_reg_list.std(axis=0),
            ax=ax_metric,
            linestyle='dashed',
            label=feedback_option + "(reg)",
        )

        # errorfill(
        #     np.arange(i_hy_list.shape[1]) + 1,
        #     i_hy_list.mean(axis=0),
        #     i_hy_list.std(axis=0),
        #     ax=ax_mi,
        #     label=feedback_option,
        # )

    for ax in axes.flat:
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        if i_hy_list.shape[1] >= 1000:
            ax.set_xscale('log')

        ax.legend(frameon=False, prop={'size': 12})

    # axes[0, 0].legend()

    axes[0, 0].set_title(r"$\Delta W_O$ ON")
    axes[0, 1].set_title(r"$\Delta W_O$ OFF")

    axes[1, 0].set_xlabel("Epochs")
    axes[1, 1].set_xlabel("Epochs")

    axes[0, 0].set_ylabel(r"Loss")
    axes[1, 0].set_ylabel(f"{task.task_metric_name}")
    plt.tight_layout()
    plt.savefig(f"figs/fig.1.{type(task).__name__}.png")


@click.command()
@click.option("--debug/--no-debug", default=False)
def main(debug):
    seed = 192
    torch.manual_seed(seed)
    np.random.seed(seed)
    if debug:
        cores = 1
        n_repeats = 1
    else:
        cores = 6
        n_repeats = 10

    # task = IBTask()
    # print(type(task).__name__)
    # train_task(task, n_repeats=3)

    t0 = time.time()
    task = IBTask(1000)

    # task.output_act_f = NnConstructor.construct_fixpoint()
    # task.d_hiddens = "6"
    print(type(task).__name__)
    train_task(task, cores=cores, n_repeats=n_repeats, seed_base=seed)
    t1 = time.time()
    print(f"{t1-t0:3.2f} Secs")


if __name__ == '__main__':
    main()
