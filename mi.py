"""Mutual information measure"""

# Simplified MI computation code from https://github.com/ravidziv/IDNNs
import numpy as np


def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(
        np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids,
                                                 return_index=False,
                                                 return_inverse=True,
                                                 return_counts=True)
    return np.asarray(unique_counts /
                      float(sum(unique_counts))), unique_inverse


def bin_calc_information(inputdata, layerdata, num_of_bins):
    p_xs, unique_inverse_x = get_unique_probs(inputdata)

    bins = np.linspace(-1, 1, num_of_bins, dtype='float32')
    digitized = bins[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bins) -
                     1].reshape(len(layerdata), -1)
    p_ts, _ = get_unique_probs(digitized)

    H_LAYER = -np.sum(p_ts * np.log(p_ts))
    H_LAYER_GIVEN_INPUT = 0.
    for xval in unique_inverse_x:
        p_t_given_x, _ = get_unique_probs(
            digitized[unique_inverse_x == xval, :])
        H_LAYER_GIVEN_INPUT += -p_xs[xval] * np.sum(
            p_t_given_x * np.log(p_t_given_x))
    return H_LAYER - H_LAYER_GIVEN_INPUT


def bin_calc_information2(labelixs, layerdata, binsize):
    # This is even further simplified, where we use np.floor instead of digitize
    def get_h(d):
        digitized = np.floor(d / binsize).astype('int')
        p_ts, _ = get_unique_probs(digitized)
        return -np.sum(p_ts * np.log(p_ts))

    H_LAYER = get_h(layerdata)
    H_LAYER_GIVEN_OUTPUT = 0
    for label, ixs in labelixs.items():
        H_LAYER_GIVEN_OUTPUT += ixs.mean() * get_h(layerdata[ixs, :])
    return H_LAYER, H_LAYER - H_LAYER_GIVEN_OUTPUT
    # H_LAYER_GIVEN_INPUT = 0.
    # for xval in unique_inverse_x:
    #     p_t_given_x, _ = get_unique_probs(digitized[unique_inverse_x == xval, :])
    #     H_LAYER_GIVEN_INPUT += - p_xs[xval] * np.sum(p_t_given_x * np.log(p_t_given_x))
    # print('here', H_LAYER_GIVEN_INPUT)
    # return H_LAYER - H_LAYER_GIVEN_INPUT


nats2bits = 1.0 / np.log(2)


def binned_entropy(data, binsize):
    digitized = np.floor(data / binsize).astype(int)
    p_ts, _ = get_unique_probs(digitized)
    return -np.sum(p_ts * np.log(p_ts)) * nats2bits


def binned_mi(data_x, data_y, binsize):
    data_xy = np.concatenate([data_x, data_y], axis=1)
    return binned_entropy(data_x, binsize) + binned_entropy(
        data_y, binsize) - binned_entropy(data_xy, binsize)
