import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import wandb


_IGNORE_DATASET_ERROR = False


def retrieve_values_from_name(fname):
    return re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", fname)


def iterate_artifacts(project, target_cfgs=None):
    target_cfgs = target_cfgs or {}
    api = wandb.Api()
    runs = api.runs(project, filters=target_cfgs)
    for run in runs:
        for file in run.files():
            yield run, file


def download_from_wandb(resdir, n_qubits):
    project = 'IsingBP'
    grads_results = {}
    target_cfgs = {'config.n_qubits': n_qubits, 'config.g': 2, 'config.h': 0}

    print(f'Downloading experiment results from {project}')
    print(f'| Results directory : {resdir}')
    print(f'| Target constraints: {target_cfgs}')
    for run, file in iterate_artifacts(project, target_cfgs):
        if file.name.startswith('grads') \
                and file.name.endswith('.npy') \
                and run.state == 'finished':
            try:
                filepath = file.download(resdir / 'tmp' / run.name)
            except wandb.apis.CommError as e:
                print(str(e))
                print('| conflict run name: ', run.name)
                print('| conflict file name:', file)
                print('| Skip this!')
                continue
                # raise e
            print(f'File downloaded: {filepath.name}', end=' ')
            fname = Path(filepath.name).name
            assert fname == file.name

            name = f'{run.config["n_qubits"]} Qubits'
            vals = retrieve_values_from_name(fname)
            n_layers = int(vals[1])  # vals = Q, L, BS, g, h
            print(f'| {name} and {n_layers} Layers', end=' ')
            if name not in grads_results:
                grads_results[name] = {}
            if n_layers not in grads_results[name]:
                grads_results[name][n_layers] = []

            grads = np.load(filepath.name)
            print(f'| n_samples, n_dims = {grads.shape}')
            grads_results[name][n_layers].append(grads)

    print('Merging the obtained results...')
    for name in grads_results:
        for n_layers, res in grads_results[name].items():
            rname = f'{name}_{n_layers}L_g{target_cfgs["config.g"]}_h{target_cfgs["config.h"]}'
            print(f'{rname}: ', end='')
            res = np.vstack(res)
            print(f'n_samples, n_dims={res.shape}')
            np.save(resdir / f'{rname}.npy', res)
    print('Done')


def load_df(resdir, label, n_samples=5000):
    records = []
    for f in resdir.iterdir():
        if f.is_file() and f.name.startswith(label):
            grads = np.load(f)
            print(f.name, 'n_samples', len(grads))
            if len(grads) < n_samples:
                print(f'{f.name} has only {len(grads)} samples,'
                      f'while we need {n_samples} samples to draw.')
                if not _IGNORE_DATASET_ERROR:
                    raise RuntimeError(
                        f'{f.name} has only {len(grads)} samples,'
                        f'while we need {n_samples} samples to draw.'
                    )
            else:
                np.random.seed(43)  # fix random seed
                np.random.shuffle(grads)
                grads = grads[:n_samples]

            grad_var = grads.var(axis=0)
            grad_var_mean_over_param = grad_var.mean()
            grad_var_min_over_param = grad_var.min()
            grad_var_max_over_param = grad_var.max()

            grads_norm = np.linalg.norm(grads, axis=1)
            grads_norm_max = grads_norm.max()
            grads_norm_min = grads_norm.min()
            grads_norm_first_quartile = np.quantile(grads_norm, 0.25)
            grads_norm_third_quartile = np.quantile(grads_norm, 0.75)
            grads_norm_mean = grads_norm.mean()
            grads_norm_var = grads_norm.var()

            records.append(
                dict(
                    n_layers=int(retrieve_values_from_name(f.name)[1]),
                    grad_mean=grads.mean(axis=0)[0],
                    grad_var=grads.var(axis=0)[0],
                    grad_var_mean_over_param=grad_var_mean_over_param,
                    grad_var_min_over_param=grad_var_min_over_param,
                    grad_var_max_over_param=grad_var_max_over_param,
                    grad_norm_mean=grads_norm_mean,
                    grad_norm_var=grads_norm_var,
                    grad_norm_max=grads_norm_max,
                    grad_norm_min=grads_norm_min,
                    grad_norm_first_quartile=grads_norm_first_quartile,
                    grad_norm_third_quartile=grads_norm_third_quartile
                )
            )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(by='n_layers')
    return df


def draw_grad_var_with_variance(resdir, n_qubits_list, linestyles, n_samples=5000,
                                xscale='log', yscale='log'):
    for i, n_qubits in enumerate(n_qubits_list):
        label = f'{n_qubits} Qubits'
        df = load_df(resdir, label, n_samples=n_samples)
        plt.plot(df.n_layers, df.grad_var_mean_over_param, linestyles[i],
                 linewidth=1.2, alpha=1.,
                 markersize=5,
                 label=label)
        plt.fill_between(
            df.n_layers,
            df.grad_var_min_over_param,
            df.grad_var_max_over_param,
            alpha=0.35)

    plt.xscale(xscale)
    plt.yscale(yscale)
    # plt.xlim(0, 260)
    plt.xlabel(r'$L$', fontsize=13)
    plt.ylabel(r'$\mathrm{Var}\,(\partial_{\theta_i} \, (E(\mathbf{\theta}) - E_0) )$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='lower right')

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/ising_bp_grad_var.pdf')
    plt.show()


def draw_grad_norm_with_shading(resdir, n_qubits_list, linestyles, n_samples=5000,
                                xscale='log', yscale='log'):
    for i, n_qubits in enumerate(n_qubits_list):
        label = f'{n_qubits} Qubits'
        df = load_df(resdir, label, n_samples=n_samples)
        plt.plot(df.n_layers, df.grad_norm_mean, linestyles[i],
                 linewidth=1.2, alpha=1.,
                 markersize=5,
                 label=label)
        plt.fill_between(
            df.n_layers,
            df.grad_norm_first_quartile,
            df.grad_norm_third_quartile,
            alpha=0.35)

    plt.xscale(xscale)
    plt.yscale(yscale)
    # plt.xlim(0, 260)
    plt.xlabel(r'$L$', fontsize=13)
    plt.ylabel(r'$\|\| \, \nabla_{\mathbf{\theta}} \, (E(\mathbf{\theta}) - E_0) \, \|\|$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='upper left')

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/ising_bp_grad_norm.pdf')
    plt.show()


def main():
    global _IGNORE_DATASET_ERROR
    _IGNORE_DATASET_ERROR = True
    resdir = Path(f'results_ising_bp/{datetime.now().strftime("%Y%m%d")}')
    linestyles = ['-o', '-.o', '--o', ':o']
    n_samples = 5000
    n_qubits_list = [4, 6, 8, 10]
    for i, n_qubits in enumerate(n_qubits_list):
        download_from_wandb(resdir, n_qubits)

    draw_grad_var_with_variance(
        resdir, n_qubits_list, linestyles,
        n_samples=n_samples, xscale='log', yscale='log')
    draw_grad_norm_with_shading(
        resdir, n_qubits_list, linestyles,
        n_samples=n_samples, xscale='log', yscale='log')


if __name__ == '__main__':
    main()
