import pandas as pd
import constants
import numpy as np
np.random.seed(100)

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


def data_union(matrices_list,normalize='scaling',milo=False):
    """unify the matrices to one matrix and return vector of labels (healthy or not)"""
    labels_hash = constants.MILO_SHEETS if milo else constants.LABELS_OF_SHEETS
    u_mat = np.vstack(matrices_list)
    if normalize == 'scaling':
        max = u_mat.max(axis=0)
        min = u_mat.min(axis=0)
        u_mat = u_mat - np.tile(min, (u_mat.shape[0], 1))
        u_mat = u_mat / np.tile(max, (u_mat.shape[0], 1))
    if normalize == 'centralize':
        u_mat = u_mat - u_mat.mean()
    labels = []
    for i in range(len(labels_hash)):
        labels += list((labels_hash[i]*np.ones(matrices_list[i].shape[0])).astype(int))
    p = np.random.permutation(len(labels))
    labels = np.array(labels)
    return u_mat[p], labels[p]


def sample(u_mat, labels, n_samples, total_samples, milo=False):
    if not milo:
        TB_idx = np.where(labels < constants.HEALTHY)[0]
        nonTB_idx = np.where(labels >= constants.HEALTHY)[0]
        TB_idx_train = np.random.choice(TB_idx, size=constants.MILO_TRAIN_SIZE, replace=False)
        nonTB_idx_train = np.random.choice(nonTB_idx, size=constants.MILO_TRAIN_SIZE, replace=False)
        TB_idx_test = np.random.choice(np.array([i for i in TB_idx if i not in TB_idx_train]), size=constants.MILO_TB_TEST_SIZE, replace=False)
        nonTB_idx_test = np.random.choice(np.array([i for i in nonTB_idx if i not in nonTB_idx_train]), size=constants.MILO_HEALTHY_TEST_SIZE, replace=False)
        train_idx = np.concatenate((TB_idx_train, nonTB_idx_train), axis=0)
        test_idx = np.concatenate((TB_idx_test, nonTB_idx_test), axis=0)
    else:
        train_idx = np.random.choice(len(labels), n_samples, replace=False)  # sample from the discrete dist of [0,2)
        test_idx = np.array([i for i in range(total_samples) if i not in train_idx])
    return u_mat[train_idx, :], u_mat[test_idx, :], (np.array(labels))[train_idx], (np.array(labels))[test_idx]


def get_data(normalize='none', milo=False):
    matrix_list = []
    sheet_range = constants.MILO_SHEETS if milo else range(constants.N_SHEETS)
    for sheet in sheet_range:
        matrix_list.append(get_sheet(idx=sheet))
    x, y = data_union(matrices_list=matrix_list, normalize=normalize, milo=milo)
    return x, y