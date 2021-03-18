import numpy as np

veclen = np.vectorize(len)
def bin_masks(values, bins):
    return np.array([(values>=b0)*(values<b1) for b0, b1 in zip(bins[:-1], bins[1:])])
def none_val(value, none_value):
    return value if value is not None else none_value
