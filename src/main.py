import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import seaborn as sns
import pandas as pd

from src import utils

def fit_model_batch(data_generator, model, loss_function, calc_orderparams, n_samples=100, reg_param=1.0, lr=0.1,
                    tol=1e-8, max_iter=20000, patience=10, verbose_interval=1000, visualize=False, seed=100, device="cpu"):
    """
    Fit the model with a stopping criterion based on the change in the loss function and a maximum iteration count.

    Parameters
    ----------
    visualize : bool, optional
        If True, plot the learning curves (order parameters) over iterations.
    """
    utils.fix_seed(seed)

    # 学習曲線を保存
    learning_curves = []

    # 学習データを準備
    X_data, y_data = data_generator.generate_dataset(n_samples)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 学習
    model.train()
    prev_loss = float('inf')
    no_improvement_count = 0

    for iteration in range(max_iter):

        optimizer.zero_grad()
        y_pred = model(X_data)
        loss = loss_function(y_pred, y_data, model, reg_param, online=False)
        loss.backward()
        optimizer.step()


        # 収束判定 (Random)
        loss_value = (loss).item()/n_samples
        if abs(prev_loss - loss_value) < tol:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        if no_improvement_count >= patience:
            # 秩序変数を計算
            orderparams = calc_orderparams(data_generator, model) + [loss_value]
            # orderparams = calc_orderparams(data_generator, model)
            learning_curves.append(orderparams)
            print(f"【{n_samples/data_generator.d:.2f}】Convergence reached at iteration {iteration + 1}, stopping training.")
            break
        prev_loss = loss_value

        # Logを表示
        if (iteration + 1) % verbose_interval == 0:
            # 秩序変数を計算
            orderparams = calc_orderparams(data_generator, model) + [loss_value]
            # orderparams = calc_orderparams(data_generator, model)
            learning_curves.append(orderparams)
            print(f"Iteration {iteration + 1}/{max_iter}, Loss: {loss_value:.4f}, orderparams: {orderparams}")

    # 秩序変数を計算
    orderparams = calc_orderparams(data_generator, model) + [loss_value]
    # orderparams = calc_orderparams(data_generator, model)
    learning_curves.append(orderparams)

    # 可視化
    if visualize:
        iterations = range(len(learning_curves))
        num_orderparams = len(learning_curves[0])

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        for i in range(num_orderparams):
            orderparam_values = [op[i] for op in learning_curves]
            ax.plot(iterations, orderparam_values, label=f"OrderParam {i+1}")

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Order Parameters")
        ax.set_title("Learning Curves Over Iterations")
        ax.grid(True, ls="--")
        ax.legend()
        plt.show()

    return model, learning_curves



def run_experiments_alphas(data_generator, model_class, loss_function, calc_orderparams, alpha_values, reg_param=1.0,
                           lr=0.01, tol=1e-8, max_iter=20000, patience=100, verbose_interval=1000, learning_visualize=False, visualize=True, seed=100, device="cpu"):
    """
    Runs the `fit_model` for various values of alpha and returns the final order parameters for each alpha.

    Parameters
    ----------
    visualize : bool, optional
        If True, plot the final order parameters for all experiments.
    """

    utils.fix_seed(seed)
    d=data_generator.d

    final_orderparams_list = []

    for alpha in alpha_values:

        # 学習データ数を設定
        n_samples = int(d*alpha)

        # モデルを初期化
        model_alpha = model_class(d=data_generator.d).to(device)

        # モデルを学習
        _, learning_curves = fit_model_batch(
                                            data_generator=data_generator,
                                            model=model_alpha,
                                            loss_function=loss_function,
                                            calc_orderparams=calc_orderparams,
                                            n_samples=n_samples,
                                            reg_param=reg_param,
                                            lr=lr,
                                            tol=tol,
                                            max_iter=max_iter,
                                            patience=patience,
                                            verbose_interval=verbose_interval,
                                            seed=seed,
                                            visualize=learning_visualize,
                                            device=device
                                            )
        # パラメータを保存
        final_orderparams_list.append(learning_curves[-1])

    # 可視化
    if visualize:
        num_orderparams = len(final_orderparams_list[0])

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

        for i in range(num_orderparams):
            orderparam_values = [op[i] for op in final_orderparams_list]
            ax.plot(alpha_values, orderparam_values, label=f"OrderParam {i+1}", marker='o')

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Order Parameters")
        ax.set_title("Order Parameters for Different Alpha Values")
        ax.grid(True, ls="--")
        ax.legend()
        plt.show()

    return final_orderparams_list


def run_experiments_alphas_seeds(data_generator, model_class, loss_function, calc_orderparams, alpha_values, reg_param=1.0,
                                 lr=0.01, tol=1e-8, max_iter=50000, patience=100, verbose_interval=5000,
                                 num_trials=5, seed_list=None, visualize=False, device="cpu"):

    if seed_list is None:
        seed_list = [100 + i for i in range(num_trials)]

    all_trials_orderparams = []

    for trial, seed in enumerate(seed_list):

        print(f"【TRIAL {trial + 1}, SEED {seed}】")

        final_orderparams_list = run_experiments_alphas(
                                                        data_generator,
                                                        model_class,
                                                        loss_function,
                                                        calc_orderparams,
                                                        alpha_values=alpha_values,
                                                        reg_param=reg_param,
                                                        lr=lr,
                                                        tol=tol,
                                                        max_iter=max_iter,
                                                        patience=patience,
                                                        verbose_interval=verbose_interval,
                                                        visualize=visualize,
                                                        learning_visualize=False,
                                                        seed=seed,
                                                        device=device
                                                        )

        all_trials_orderparams.append(final_orderparams_list)

    # Shape: (num_trials, num_alphas, num_orderparams)
    all_trials_orderparams = np.array(all_trials_orderparams)

    # 可視化
    if visualize:
        num_orderparams = all_trials_orderparams.shape[2]

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

        for i in range(num_orderparams):

            ax.errorbar(alpha_values,
                        np.mean(all_trials_orderparams[:, :, i], axis=0),
                        yerr=np.std(all_trials_orderparams[:, :, i], axis=0),
                        label=f"OrderParam {i+1}",
                        marker='o',
                        capsize=5)

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Final Order Parameters (mean ± std)")
        ax.set_title("Order Parameters for Different Alpha Values with Error Bars")
        ax.grid(True, ls="--")
        ax.legend()
        plt.show()

    return all_trials_orderparams


def fit_model_online(data_generator, model, loss_function, optimizer, calc_orderparams, reg_param=1.0,
                     n_epochs=100, verbose_interval=10, visualize=False, seed=100):

    # 乱数シードを固定
    utils.fix_seed(seed)

    # 学習曲線を格納
    learning_curves = []

    d = data_generator.d

    # 初期状態 (TODO: 初期状態を適切な範囲するコードを書く)
    # X_sample, y_sample = data_generator.generate_sample()
    X_sample, y_sample = data_generator.generate_dataset(1)
    y_pred = model(X_sample)
    loss = loss_function(y_pred, y_sample, model, reg_param, online=True)
    orderparams = calc_orderparams(data_generator, model)
    print(f"Init, Loss: {loss.item():.4f}, orderparams: {orderparams}")
    learning_curves.append(orderparams)

    # Training setup
    model.train()

    for epoch in range(n_epochs):

        x_sample, y_sample = data_generator.generate_sample()
        y_pred = model(x_sample.T)
        loss = loss_function(y_pred, y_sample, model, reg_param, online=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 秩序変数を計算
        orderparams = calc_orderparams(data_generator, model)
        learning_curves.append(orderparams)

        # Logを表示
        if (epoch + 1) % verbose_interval == 0:
            print(f"Epoch {int((epoch + 1)/d)}/{int(n_epochs/d)}, Loss: {loss.item():.4f}, orderparams: {orderparams}")

    # 可視化
    if visualize:

        # alpha配列を作成
        iterations = [i / d for i in range(len(learning_curves))]
        num_orderparams = len(learning_curves[0])

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        for i in range(num_orderparams):
            orderparam_values = [op[i] for op in learning_curves]
            ax.plot(iterations, orderparam_values, label=f"OrderParam {i+1}")

        ax.set_xlabel(r"$t$")
        ax.set_ylabel("Order Parameters")
        ax.set_title("Learning Curves Over Iterations (scaled by d)")
        ax.grid(True, ls="--")
        ax.legend()
        plt.show()

    return model, learning_curves



def run_experiments_online_seeds(data_generator, model, loss_function, optimizer, calc_orderparams, 
                                 n_epochs=100, reg_param=1.0, lr=0.1, num_trials=5, verbose_interval=10, 
                                 learning_visualize=True, visualize=True, seed_list=None, device="cpu"):

    # seedリスト作成
    if seed_list is None:
        seed_list = [100 + i for i in range(num_trials)]

    all_learning_curves = []

    for trial, seed in enumerate(seed_list):

        print(f"【TRIAL {trial + 1}, SEED {seed}】")

        model_trial = copy.deepcopy(model).to(device)
        # Online学習
        
        _, learning_curves = fit_model_online(
            data_generator,
            model_trial,
            loss_function,
            optimizer,
            calc_orderparams,
            n_epochs=n_epochs,
            reg_param=reg_param,
            verbose_interval=verbose_interval,
            visualize=learning_visualize,
            seed=seed
        )

        all_learning_curves.append(learning_curves)

    all_learning_curves = np.array(all_learning_curves)  # Shape: (num_seeds, n_epochs, num_orderparams)

    # 可視化
    if visualize:
        num_orderparams = all_learning_curves.shape[2]
        iterations = [i / data_generator.d for i in range(n_epochs + 1)]

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        for i in range(num_orderparams):

            ax.plot(iterations,
                     np.mean(all_learning_curves[:, :, i], axis=0),
                     label=f"OrderParam {i+1}")
            ax.fill_between(iterations,
                            np.mean(all_learning_curves[:, :, i], axis=0) - np.std(all_learning_curves[:, :, i], axis=0),
                            np.mean(all_learning_curves[:, :, i], axis=0) + np.std(all_learning_curves[:, :, i], axis=0),
                            alpha=0.3)

        ax.set_xlabel(r"$t$")
        ax.set_ylabel("Order Parameters")
        ax.set_title("Average Learning Curves with Variance (multiple seeds)")
        ax.grid(True, ls="--")
        ax.legend()
        plt.show()

    return all_learning_curves