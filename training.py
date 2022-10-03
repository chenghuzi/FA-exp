import itertools  # noqa
import time  # noqa
import ipdb  # noqa
import numpy as np
import pathos.multiprocessing as pmp  # noqa
import sklearn.linear_model as linear_model
import torch
from tqdm import trange
import mi
from models import FADeepFullNet
import multiprocessing as mp


def train_model_full(
    feedback_option,
    dw_on_output,
    extra_tf_modification,
    model,
    optimizer,
    training_loader,
    loss_func,
    metric_func,
    n_epochs,
    epoch_interval=5,
    mi_on_batch=True,
    mi_bin_size=1.0 / 10,
    return_reg_related=False,
    device="cpu",
):
    epoch_losses = []
    epoch_metrics = []
    epoch_i_hys = []
    # reg related
    epoch_loss_regs = []
    epoch_metric_regs = []

    pbar = trange(n_epochs)
    for epoch in pbar:

        epoch_loss = []
        epoch_metric = []
        epoch_i_hy = []
        epoch_i_dhy = []

        H_nps = []
        y_batch_nps = []

        # reg related
        epoch_loss_reg = []
        epoch_metric_reg = []

        for x_batch, y_batch in training_loader:
            y_batch_np = y_batch.data.cpu().numpy()
            # Notice here the y_batch denotes direct output.
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # y_label_batch = y_label_batch.to(device)
            y_pred, activities = model(x_batch, retain_act_grad=True)
            H = activities[-2]
            H_np = H.data.cpu().numpy()

            # we assume the metric is always between 0 and 1 and the higher the better
            metric = metric_func(y_pred.data.detach(), y_batch)
            # ipdb.set_trace()
            # For target feedback we directly propagate target information back.

            if feedback_option in ['TF']:
                # ipdb.set_trace()
                if extra_tf_modification > 0:
                    # entropy increase
                    eye = torch.eye(H.shape[0]).to(device)
                    H_norm_sq = (H**2).sum(dim=1).unsqueeze(1)
                    cov_m = H @ H.T / torch.sqrt(H_norm_sq @ H_norm_sq.T)
                    loss_entropy_H = extra_tf_modification * (1 - max(
                        metric, 0)) * torch.nn.functional.mse_loss(cov_m, eye)
                    loss_entropy_H.backward(retain_graph=True)
                    pass

                # this is just for encoder
                scale = 1e-2
                fb_signal = scale * (1 - max(metric, 0)) * y_batch
                y_pred.backward(fb_signal)
                model.clear_decoder_grads()

                # this is just for the decoder
                y_pred_2 = model.decode(H.data.detach())
                loss = loss_func(y_pred_2, y_batch)

                loss.backward()
                # if extra_tf_modification:
                # if False:
                #     # directly compute the output weights
                #     # linear relationship is assumed
                #     reg = linear_model.LinearRegression()
                #     reg.fit(H_np, y_batch_np)
                #     model.decoder.weight.data = torch.from_numpy(
                #         reg.coef_).to(device)
                #     if model.decoder.bias is not None:
                #         model.decoder.bias.data = torch.from_numpy(
                #             reg.intercept_).to(device)
                # else:
                #     loss.backward()

            else:
                # otherwise we just propagate the loss.
                loss = loss_func(y_pred, y_batch)
                loss.backward()

            if dw_on_output is False:
                model.clear_decoder_grads()

            optimizer.step()

            i_dhy = mi.binned_mi(H.grad.data.cpu().numpy(),
                                 y_batch_np,
                                 binsize=mi_bin_size)
            epoch_i_dhy.append(i_dhy)

            H_nps.append(H_np)
            # measure mutual information here.
            if mi_on_batch:
                i_hy = mi.binned_mi(H_np, y_batch_np, binsize=mi_bin_size)
                epoch_i_hy.append(i_hy)
            else:
                y_batch_nps.append(y_batch_np)

            epoch_loss.append(loss.item())
            epoch_metric.append(metric.item())

            if return_reg_related:
                reg = linear_model.LinearRegression(fit_intercept=True)
                reg.fit(H_np, y_batch_np)
                pred_reg = reg.predict(H_np)
                loss_reg = loss_func(pred_reg, y_batch)
                metric_reg = metric_func(pred_reg, y_batch)
                epoch_loss_reg.append(loss_reg.item())
                epoch_metric_reg.append(metric_reg.item())

        epoch_losses.append(epoch_loss)
        epoch_metrics.append(epoch_metric)

        if return_reg_related:
            epoch_loss_regs.append(epoch_loss_reg)
            epoch_metric_regs.append(epoch_metric_reg)

        H_nps = np.concatenate(H_nps, axis=0)
        if not mi_on_batch:
            y_batch_nps = np.concatenate(y_batch_nps, axis=0)
            epoch_i_hy = [
                mi.binned_mi(H_nps, y_batch_nps, binsize=mi_bin_size)
            ]

        epoch_i_hys.append(epoch_i_hy)

        if epoch % epoch_interval == 0:
            pbar.set_description(
                f"Epoch {epoch+1}, Loss: {np.mean(epoch_loss):1.3f}, Metric: {np.mean(epoch_metrics):1.3f} | "
                f"I(H;Y): {np.mean(epoch_i_hy):1.3f}, I(dH;Y): {np.mean(epoch_i_dhy):1.3f} | "
                f"maxH: {H_nps.max():1.3f},minH: {H_nps.min():1.3f}, stdH: {H_nps.std():1.3f}"
            )

    if return_reg_related:
        return (
            np.array(epoch_losses).mean(axis=1),
            np.array(epoch_metrics).mean(axis=1),
            np.array(epoch_i_hys).mean(axis=1),
            np.array(epoch_loss_regs).mean(axis=1),
            np.array(epoch_metric_regs).mean(axis=1),
        )
    else:
        return (
            np.array(epoch_losses).mean(axis=1),
            np.array(epoch_metrics).mean(axis=1),
            np.array(epoch_i_hys).mean(axis=1),
        )


def train_model_no_tf_modify(
    feedback_option,
    dw_on_output,
    model,
    optimizer,
    training_loader,
    loss_func,
    metric_func,
    n_epochs,
    epoch_interval=5,
    mi_on_batch=True,
    mi_bin_size=1.0 / 10,
    return_reg_related=False,
    device="cpu",
):
    extra_tf_modification = False,
    return train_model_full(
        feedback_option,
        dw_on_output,
        extra_tf_modification,
        model,
        optimizer,
        training_loader,
        loss_func,
        metric_func,
        n_epochs,
        epoch_interval=epoch_interval,
        mi_on_batch=mi_on_batch,
        mi_bin_size=mi_bin_size,
        return_reg_related=return_reg_related,
        device=device,
    )


def train_deep_full_net_worker(args):
    # results,
    task, device, return_reg_related, seed_base, key = args
    feedback_option, dw_on_output, allow_scale_after_act, repeat_id = key
    seed = hash(key) % 2**10

    if seed_base is not None:
        seed = seed + seed_base
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(key, "is training")
    model = FADeepFullNet(
        d_i=task.dim_in,
        d_o=task.dim_out,
        d_hiddens=task.d_hiddens,
        feedback_option=feedback_option,
        dw_decoder_on=dw_on_output,
        output_act_f=task.output_act_f,
        backbone_net_f=task.backbone_net,
        nfa_amp=task.nfa_amp,
        allow_scale_after_act=allow_scale_after_act,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=task.lr)

    res = train_model_no_tf_modify(
        feedback_option,
        dw_on_output,
        model,
        optimizer,
        task.training_loader,
        task.loss_func,
        task.metric_func,
        n_epochs=task.n_epochs,
        mi_on_batch=task.mi_on_batch,
        return_reg_related=return_reg_related,
        device=device,
    )

    print(feedback_option, dw_on_output, repeat_id, "finished")
    return key, res


def train_deep_full_net_worker_tf_modification(args):
    # results,
    task, device, return_reg_related, seed_base, key = args
    feedback_option, dw_on_output, extra_tf_modification, allow_scale_after_act, repeat_id = key
    seed = hash(key) % 2**10

    if seed_base is not None:
        seed = seed + seed_base
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(key, "is training")
    model = FADeepFullNet(
        d_i=task.dim_in,
        d_o=task.dim_out,
        d_hiddens=task.d_hiddens,
        feedback_option=feedback_option,
        dw_decoder_on=dw_on_output,
        output_act_f=task.output_act_f,
        backbone_net_f=task.backbone_net,
        nfa_amp=task.nfa_amp,
        allow_scale_after_act=allow_scale_after_act,
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=task.lr)

    res = train_model_full(
        feedback_option,
        dw_on_output,
        extra_tf_modification,
        model,
        optimizer,
        task.training_loader,
        task.loss_func,
        task.metric_func,
        n_epochs=task.n_epochs,
        mi_on_batch=task.mi_on_batch,
        return_reg_related=return_reg_related,
        device=device,
    )

    print(feedback_option, dw_on_output, repeat_id, "finished")
    return key, res
