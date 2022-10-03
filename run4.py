"""Test TF on two regression task.


example:
1. run the ib regression task
python run4.py train  --task_name ibreg
2. run the mnist regression task
python run4.py train  --task_name mnistreg
3. visualization
python run4.py viz
"""

import dill as pickle
import itertools
from pathlib import Path
import time

import click
import ipdb  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pathos.multiprocessing as pmp
import multiprocessing as mp
import torch
from utils import errorfill
from tasks import IBRegTask, MNISTRegTask
from exp_utils import save_exp_config, setup_result_dir
from training import train_deep_full_net_worker_tf_modification
device = "cuda"


@click.group()
def cli():
    click.echo("Run4")


def visualize(results_all: dict, result_dir):
    n_cols = len(results_all)
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=[3 * n_cols, 6],
        sharex='col',
    )
    for task_idx, task_name in enumerate(results_all.keys()):
        _task, _results = results_all[task_name]
        conditions = list(_results.keys())

        for feedback_option, dw_on_output in conditions:
            assert dw_on_output is True
            if n_cols == 1:
                ax_loss = axes[0]
                ax_metric = axes[1]
                ax_mi = axes[2]
            else:
                ax_loss = axes[0, task_idx]
                ax_metric = axes[1, task_idx]
                ax_mi = axes[2, task_idx]

            ax_loss.set_title(type(_task).__name__)
            ax_loss.set_ylabel(r"Loss")

            ax_metric.set_ylabel(f"{_task.task_metric_name}")
            ax_mi.set_ylabel("$I(H;Y)$")
            ax_mi.set_xlabel("Epochs")

            loss_list = np.array(
                [r[0] for r in _results[(feedback_option, dw_on_output)]])
            metric_list = np.array(
                [r[1] for r in _results[(feedback_option, dw_on_output)]])
            i_hy_list = np.array(
                [r[2] for r in _results[(feedback_option, dw_on_output)]])

            # loss
            label = feedback_option

            errorfill(
                np.arange(loss_list.shape[1]) + 1,
                loss_list.mean(axis=0),
                loss_list.std(axis=0),
                ax=ax_loss,
                label=label,
            )
            # metric
            errorfill(
                np.arange(metric_list.shape[1]) + 1,
                metric_list.mean(axis=0),
                metric_list.std(axis=0),
                ax=ax_metric,
                label=label,
            )
            # MI
            errorfill(
                np.arange(i_hy_list.shape[1]) + 1,
                i_hy_list.mean(axis=0),
                i_hy_list.std(axis=0),
                ax=ax_mi,
                label=label,
            )

    for ax in axes.flat:
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        ax.legend(frameon=False, prop={'size': 12})

    plt.tight_layout()
    plt.savefig(result_dir / "fig.tf_comparison_algorithms.png")
    plt.savefig("figs/fig.tf_comparison_algorithms.png")


def train_task(task,
               cores=6,
               n_repeats=3,
               extra_tf_modification=100,
               seed_base=None):
    results = {}
    dw_on_outputs = (True, )
    feedback_options = (
        "TF",
        "BP",
        "FA",
    )

    conditions = list(itertools.product(feedback_options, dw_on_outputs))

    trials = []
    return_reg_related = False

    for feedback_option, dw_on_output in conditions:
        print(feedback_option, dw_on_output)
        for repeat_idx in range(n_repeats):
            allow_scale_after_act = feedback_option != "BP"
            trials.append((
                task,
                device,
                return_reg_related,
                seed_base,
                (
                    feedback_option,
                    dw_on_output,
                    extra_tf_modification,
                    allow_scale_after_act,
                    repeat_idx,
                ),
            ))
    # debug
    if cores == 1:
        res_all_list = list(
            map(train_deep_full_net_worker_tf_modification, trials))
    else:
        with pmp.Pool(cores) as p:
            res_all_list = p.map(train_deep_full_net_worker_tf_modification,
                                 trials)
    print("dsd")
    # results = dict(results)
    for cond_key in conditions:
        results[cond_key] = [
            res[1] for res in res_all_list if res[0][:2] == cond_key
        ]
    return task, results


@cli.command()
@click.option("--task_name")
@click.option("--debug/--no-debug", default=False)
def train(task_name, debug):

    result_dir, exp_id = setup_result_dir()
    save_exp_config(result_dir, {'debug': debug})

    seed = 1927
    torch.manual_seed(seed)
    np.random.seed(seed)
    if debug:
        torch.set_printoptions(sci_mode=False)
        cores = 1
        n_repeats = 1
    else:
        cores = 4
        n_repeats = 10

    if task_name == 'ibreg':
        _task = IBRegTask(1000)
        extra_tf_modification = 10
        _cores = 1
    elif task_name == 'mnistreg':
        _task = MNISTRegTask(30)
        extra_tf_modification = 100
        _cores = cores

    t0 = time.time()

    res_all = {}
    res_all[type(_task).__name__] = train_task(
        _task,
        cores=_cores,
        extra_tf_modification=extra_tf_modification,
        n_repeats=n_repeats,
        seed_base=seed,
    )
    data_dir = Path("data") / "run4"
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / f"{type(_task).__name__}.pkl", "wb") as f:
        pickle.dump(res_all, f)

    t1 = time.time()
    print(f"{t1-t0:3.2f} Secs")
    #


@cli.command()
@click.option("--resdir", default='run4')
def viz(resdir):
    debug = False
    result_dir, exp_id = setup_result_dir()
    save_exp_config(result_dir, {'debug': debug})
    res_all = {}
    data_dir = Path("data") / resdir
    for dp in data_dir.glob("*.pkl"):
        print(dp.stem)
        res_all.update(pickle.load(open(dp, "rb")))

    visualize(res_all, result_dir)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    cli()
