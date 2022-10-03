"""To show that FA performance could increase if we add Scaling layer.
"""
import itertools
import time

import click
import ipdb  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pathos.multiprocessing as pmp
import torch
from tasks import IBTask
from training import train_deep_full_net_worker
from utils import errorfill

device = "cuda"


def train_task(task, cores=6, n_repeats=3, figsize=[8, 4], seed_base=None):
    results = {}
    feedback_options = ("FA", )
    dw_on_outputs = (True, )
    allow_scale_after_acts = (True, False)

    conditions = list(
        itertools.product(feedback_options, dw_on_outputs,
                          allow_scale_after_acts))

    trials = []
    return_reg_related = True
    for feedback_option, dw_on_output, allow_scale_after_act in conditions:
        print(feedback_option, dw_on_output, allow_scale_after_act)
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
            res[1] for res in res_all_list if res[0][:-1] == cond_key
        ]

    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

    for feedback_option, dw_on_output, allow_scale_after_act in conditions:
        assert feedback_option == "FA"
        assert dw_on_output is True

        # if dw_on_output is True:
        ax_loss = axes[0]
        ax_metric = axes[1]

        loss_list = np.array([
            r[0] for r in results[(feedback_option, dw_on_output,
                                   allow_scale_after_act)]
        ])

        metric_list = np.array([
            r[1] for r in results[(feedback_option, dw_on_output,
                                   allow_scale_after_act)]
        ])

        # loss
        label = f"{feedback_option}, Allow Scaling: {allow_scale_after_act}"

        errorfill(
            np.arange(loss_list.shape[1]) + 1,
            loss_list.mean(axis=0),
            loss_list.std(axis=0),
            ax=ax_loss,
            label=label,
        )

        errorfill(
            np.arange(metric_list.shape[1]) + 1,
            metric_list.mean(axis=0),
            metric_list.std(axis=0),
            ax=ax_metric,
            label=label,
        )

    for ax in axes.flat:
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

    ax_loss.legend(loc='upper right', frameon=False, prop={'size': 12})

    axes[0].set_xlabel("Epochs")
    axes[1].set_xlabel("Epochs")

    axes[0].set_ylabel(r"Loss")
    axes[1].set_ylabel(f"{task.task_metric_name}")

    plt.tight_layout()
    plt.savefig(f"figs/fig.2.scaling.{type(task).__name__}.png")


@click.command()
@click.option("--debug/--no-debug", default=False)
def main(debug):
    seed = 1927
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    if debug:
        cores = 1
        n_repeats = 1
    else:
        cores = 6
        n_repeats = 30

    task = IBTask()
    print(type(task).__name__)
    t0 = time.time()
    train_task(task,
               cores=cores,
               n_repeats=n_repeats,
               figsize=[8, 4],
               seed_base=seed)
    t1 = time.time()
    print(f"{t1-t0:3.2f} Secs")


if __name__ == '__main__':
    main()
