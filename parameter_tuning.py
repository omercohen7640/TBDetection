import optuna
import data_handling
import constants
import numpy as np
import os
import main
import datetime
import logging


def objective(trial, x, y):
    x_train, x_val, y_train, y_val = data_handling.sample(x, y, int(constants.N*0.6), int(constants.N_TUNNING))
    # epsilon = trial.suggest_float(name="epsilon", low=10e-2, high=10e5, log=True)
    t = trial.suggest_int(name='t', low=1, high=5)
    x_map, dm = main.execute_diffmap(_x_sampled=x_train, epsilon='median', t=t)
    return main.train_classifier('tree', x_map, y_train, x_val, y_val, dm)


if __name__ == "__main__":
    exp_path = os.path.join(os.getcwd(), 'experiments', str(datetime.datetime.now()))
    os.mkdir(exp_path)
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler(os.path.join(exp_path,"exp_log.txt"), mode="w"))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    # optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    x, y = data_handling.get_data()
    func = lambda trial: objective(trial, x, y)
    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=1000)
    print(study.best_trial)
