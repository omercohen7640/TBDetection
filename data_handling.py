import pandas as pd
import constants
import numpy as np
np.random.seed(10)

def get_sheet(idx, scaling=True):
    """read specific data sheet and avaraging the samples"""
    df = pd.read_excel(constants.DATASET, sheet_name=idx, header=1, index_col=0)
    df = df[constants.FEATURES_LIST]
    # summing duplicated samples
    df_odd = df[1::2]
    df_even = df[0::2]
    df_odd.set_index(df_even.index, inplace=True)
    df_sum = df_even.select_dtypes(np.number) + df_odd.select_dtypes(np.number)
    df_sum[df_sum.select_dtypes(np.number).columns] = df_sum.select_dtypes(np.number).div(2)
    return df_sum.to_numpy()


def data_union(matrices_list, labels_hash,normalize='scaling'):
    """unify the matrices to one matrix and return vector of labels (healthy or not)"""
    u_mat = np.vstack(matrices_list)
    if normalize=='scaling':
        max = u_mat.max(axis=0)
        min = u_mat.min(axis=0)
        u_mat = u_mat - np.tile(min, (u_mat.shape[0], 1))
        u_mat = u_mat / np.tile(max, (u_mat.shape[0], 1))
    if normalize=='centralize':
        u_mat = u_mat -u_mat.mean()
    labels = []
    for i in range(len(labels_hash)):
        labels += list((labels_hash[i]*np.ones(matrices_list[i].shape[0])).astype(int))
    return u_mat, np.array(labels)


def sample(u_mat, labels, n_samples):
    idx = np.random.choice(len(labels), n_samples)  # sample from the discrete dist of [0,2)
    idx_test = np.array([i for i in range(constants.N) if not i in idx])
    return u_mat[idx, :], u_mat[idx_test, :], (np.array(labels))[idx], (np.array(labels))[idx_test]


