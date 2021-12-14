import pydiffmap.diffusion_map

import data_handling
import constants
import lightgbm as gbm
import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
from DiffMap import DiffMap as diff
import optuna

if __name__ == '__main__':
    matrix_list = []
    for sheet in range(constants.N_SHEETS):
        matrix_list.append(data_handling.get_sheet(idx=sheet))
    x, y = data_handling.data_union(matrices_list=matrix_list, labels_hash=constants.LABELS_OF_SHEETS, normalize=False)
    x_sampled, x_test, y_sampled, y_test = data_handling.sample(x, y, constants.N_SAMPLES)
    dm = diff(x_sampled, y, t=1, epsilon=constants.epsilon, sigma=0.01)
    dm.compute_eigen()
    x_map = dm.get_map(dim=2)
    plt.scatter(np.arange(dm.eigenValues.shape[0]), dm.eigenValues)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    scatter = ax.scatter(x_map[:, 0], x_map[:, 1], c=y_sampled)
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)

    classifier = gbm.LGBMClassifier()
    classifier.fit(x_map, y_sampled, eval_metric='logloss')

    x_test_map = dm.nystrom_out_of_sample(x_test)
    pred=classifier.predict(x_test_map)
    print(((y_test==pred).sum())/len(pred))


