import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import wandb


def retrieve_values_from_name(fname):
    return re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", fname)


def iterate_artifacts(project, target_cfgs=None):
    api = wandb.Api()
    runs = api.runs(project)
    for run in runs:
        cfg = run.config
        is_target = True
        if target_cfgs:
            for k, v in target_cfgs.items():
                if cfg[k] != v:
                    is_target = False
        if is_target:
            for file in run.files():
                yield run, file


def download_from_wandb(resdir, n_qubits):
    project = 'IsingBP'
    grads_results = {}
    target_cfgs = dict(n_qubits=n_qubits, g=2.0, h=0.0)

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
            rname = f'{name}_{n_layers}L_g{target_cfgs["g"]}_h{target_cfgs["h"]}'
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
            if len(grads) < n_samples:
                raise RuntimeError(
                    f'{f.name} has only {len(grads)} samples,'
                    f'while we need {n_samples} samples to draw.'
                )
            np.random.seed(43)  # fix random seed
            np.random.shuffle(grads)
            grads = grads[:n_samples]

            grads_norm = np.linalg.norm(grads, axis=1)
            grads_norm_max = grads_norm.max()
            grads_norm_min = grads_norm.min()
            grads_norm_first_quantile = np.quantile(grads_norm, 0.25)
            grads_norm_third_quantile = np.quantile(grads_norm, 0.75)
            grads_norm_mean = grads_norm.mean()
            grads_norm_var = grads_norm.var()

            records.append(
                dict(
                    n_layers=int(retrieve_values_from_name(f.name)[1]),
                    grad_mean=grads.mean(axis=0)[0],
                    grad_var=grads.var(axis=0)[0],
                    grad_norm_mean=grads_norm_mean,
                    grad_norm_var=grads_norm_var,
                    grad_norm_max=grads_norm_max,
                    grad_norm_min=grads_norm_min,
                    grads_norm_first_quantile=grads_norm_first_quantile,
                    grads_norm_third_quantile=grads_norm_third_quantile
                )
            )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(by='n_layers')
    return df


def draw_grad_var(resdir, n_qubits_list, linestyles, n_samples=5000):
    for i, n_qubits in enumerate(n_qubits_list):
        label = f'{n_qubits} Qubits'
        df = load_df(resdir, label, n_samples=n_samples)
        plt.plot(df.n_layers, df.grad_var, linestyles[i],
                 linewidth=1.2, alpha=1.,
                 markerfacecolor='none', markersize=5,
                 label=label)
    plt.yscale('log')
    plt.xlim(0, 260)
    plt.xlabel(r'$L$', fontsize=13)
    plt.ylabel(r'$\mathrm{Var}\,(\partial_{\theta_i} \, (E_{\mathbf{\theta}} - E_0) )$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='lower right')

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/ising_bp_grad_var.pdf')
    plt.show()


def draw_grad_norm_shaded_version(resdir, n_qubits_list, linestyles, n_samples=5000):
    for i, n_qubits in enumerate(n_qubits_list):
        label = f'{n_qubits} Qubits'
        df = load_df(resdir, label, n_samples=n_samples)
        plt.plot(df.n_layers, df.grad_norm_mean, linestyles[i],
                 linewidth=1.2, alpha=1.,
                 markerfacecolor='none', markersize=5,
                 label=label)
        plt.fill_between(df.n_layers, df.grad_norm_min, df.grad_norm_max, alpha=0.35)

    plt.xlim(0, 260)
    plt.xlabel(r'$L$', fontsize=13)
    plt.ylabel(r'$\|\| \, \partial \, (E_{\mathbf{\theta}} - E_0) \, \|\|$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='upper left')

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/ising_bp_grad_norm.pdf')
    plt.show()


def draw_grad_norm(resdir, n_qubits_list, linestyles, n_samples=5000):
    for i, n_qubits in enumerate(n_qubits_list):
        label = f'{n_qubits} Qubits'
        df = load_df(resdir, label, n_samples=n_samples)
        err_high = df.grads_norm_third_quantile - df.grad_norm_mean
        err_low = df.grad_norm_mean - df.grads_norm_first_quantile
        plt.errorbar(
            df.n_layers, df.grad_norm_mean,
            fmt=linestyles[i],
            linewidth=1.2, alpha=1.,
            markerfacecolor='none', markersize=5,
            yerr=[err_low, err_high],
            capsize=2.3,
            label=label
        )
    plt.xlim(0, 260)
    plt.xlabel(r'$L$', fontsize=13)
    plt.ylabel(r'$\|\| \, \partial \, (E_{\mathbf{\theta}} - E_0) \, \|\|$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='upper left')

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/ising_bp_grad_norm.pdf')
    plt.show()


def main():
    resdir = Path(f'results_ising_bp/{datetime.now().strftime("%Y%m%d")}')
    linestyles = ['-o', '-.x', '-->', ':^']
    n_qubits_list = [4, 6, 8, 10]
    # n_qubits_list = [6]
    for i, n_qubits in enumerate(n_qubits_list):
        download_from_wandb(resdir, n_qubits)
    draw_grad_var(resdir, n_qubits_list, linestyles, n_samples=1000)
    draw_grad_norm(resdir, n_qubits_list, linestyles, n_samples=1000)


if __name__ == '__main__':
    main()
