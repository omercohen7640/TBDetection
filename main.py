import pydiffmap.diffusion_map

import data_handling
import constants
import lightgbm as gbm
import matplotlib.pyplot as plt
import numpy as np
from DiffMap import DiffMap as diff
from sklearn.svm import SVC as svm
from sklearn.manifold import TSNE as tsne


def eigenvalues_plot(dm):
    evalues = dm.eigenValues
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.title.set_text('Eigenvalues descending order')
    ax.set_xlabel('n')
    ax.axvline(11, color='r', lw='0.7')
    ax.axvline(23, color='r', lw='0.7')
    ax.text(11, 0.5, 'Eigenvalue 11', rotation=90)
    ax.text(23, 0.5, 'Eigenvalue 23', rotation=90)
    ax.set_ylabel('magnitude')
    ax.scatter(np.arange(evalues.shape[0]), evalues)
    # plt.show()


def scatter_plot(_x_map, y, divide_sheets=False,milo=constants.MILO):
    if _x_map.shape[1] > 2:
        x_reduced = tsne().fit_transform(_x_map)
    else:
        x_reduced = _x_map
    fig = plt.figure()
    ax = fig.add_subplot()
    scatter = ax.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y,)
    # ax.set_xlim([-20, 20])
    # ax.set_ylim([-20, 20])
    ax.title.set_text(f'train set mapped dim={_x_map.shape[1]}')
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_xlim([-10,10])
    ax.set_ylim([-10, 10])
    if _x_map.shape[1] <= 2:
         ax.set_xlim([-0.005, 0.005])
    legend = ax.legend(*scatter.legend_elements())
    if divide_sheets:
        labels = [constants.LABELS_OF_SHEETS_FULL[l] for l in constants.MILO_SHEETS] if milo else constants.LABELS_OF_SHEETS_FULL
        legend = ax.legend(handles=scatter.legend_elements()[0], labels=labels)

    ax.add_artist(legend)
    # plt.show()


def execute_diffmap(_x_sampled, epsilon=constants.EPSILON, t=constants.T):
    dm = diff(_x_sampled, t=t, epsilon=epsilon, sigma=0.01)
    dm.compute_eigen()
    _x_map = dm.get_map(dim=constants.DIM)
    return _x_map, dm


def train_classifier(type, _x_map, _y_sampled, _x_test, _y_test, dm):
    _y_test_binary = (_y_test < constants.HEALTHY).astype(int) # 1 = TB, 0 = Healthy
    _y_sampled_binary = (_y_sampled < constants.HEALTHY).astype(int)
    x_test_map = dm.nystrom_out_of_sample(_x_test)
    if type == 'tree':
        classifier = gbm.LGBMClassifier()
        classifier.fit(_x_map, _y_sampled_binary, eval_metric='logloss')
        pred = classifier.predict(x_test_map)
    elif type == 'svm':
        classifier = svm(kernel='linear')
        classifier.fit(_x_map, _y_sampled_binary)
        pred = classifier.predict(x_test_map)
    print(f'Total accuracy: {(np.equal(pred, _y_test_binary)).sum()/_y_test.shape[0]}')
    # print(f'Total FP: {((1-np.equal(pred[pred==1],_y_test_binary[pred==1]).astype(int)).sum())/(_y_test_binary[_y_test_binary==0].shape[0])}')
    # print(f'Total FN: {(1 - np.equal(pred[pred == 0], _y_test_binary[pred == 0]).astype(int)).sum() / _y_test_binary[_y_test_binary == 1].shape[0]}')
    TP = np.equal(pred[_y_test_binary == 1],_y_test_binary[_y_test_binary==1]).astype(int).sum()
    P = _y_test_binary[_y_test_binary==1].shape[0]
    TN = np.equal(pred[_y_test_binary == 0], _y_test_binary[_y_test_binary == 0]).astype(int).sum()
    N = _y_test_binary[_y_test_binary == 0].shape[0]
    FP = (1 - np.equal(pred[pred==1],_y_test_binary[pred==1]).astype(int)).sum()
    FN = (1 - np.equal(pred[pred == 0], _y_test_binary[pred == 0]).astype(int)).sum()
    print(f'Total sensitivity (TPR): {TP/P}')
    print(f'Total specificity (TNR): {TN/N}')
    print(f'Total PPV: {TP/(TP+FP)}')
    print(f'Total NPV: {TN/(TN+FN)}')
    for l in constants.LABELS_OF_SHEETS:
        _y_test_binary_one_label = _y_test_binary[np.where(_y_test == l)]
        pred_one_label = pred[np.where(_y_test == l)]
        print(f'Accuracy for label {constants.LABELS_OF_SHEETS_FULL[l]} is {np.equal(pred_one_label,_y_test_binary_one_label).sum()/pred_one_label.shape[0]}')





if __name__ == '__main__':
    x, y = data_handling.get_data(milo=constants.MILO)
    x_sampled, x_test, y_train, y_test = data_handling.sample(x, y, n_samples=int(constants.N_TUNNING), total_samples=int(constants.N))
    x_map, dm = execute_diffmap(x_sampled, epsilon='median')
    # plot
    eigenvalues_plot(dm)
    scatter_plot(x_map, y_train, divide_sheets=True)
    train_classifier(type='tree',_x_map=x_map, _y_sampled=y_train, _x_test=x_test, _y_test=y_test, dm=dm)
    plt.show()

