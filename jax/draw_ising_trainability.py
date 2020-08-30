import re
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

matplotlib.rcParams['mathtext.fontset'] = 'stix'

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def retrieve_values_from_name(fname):
    return re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", fname)


def download_from_wandb(resdir):
    project = 'IsingModel'
    target_cfgs = {
        'config.g': 2,
        'config.h': 0,
        'config.lr': 0.05,
        'config.scheduler_name': 'constant',
    }
    print(f'Downloading experiment results from {project}')
    print(f'| Results directory : {resdir}')
    print(f'| Target constraints: {target_cfgs}')

    api = wandb.Api()
    runs = api.runs(project, filters=target_cfgs)
    records = []
    for run in runs:
        if run.state == 'finished':
            print(run.name)
            history = run.history()
            best_step = history.loss.argmin()
            min_energy_gap = np.abs(history.loss[best_step])  # |E(\theta) - E0|
            fidelity = history['fidelity/ground'][best_step]
            loss_threshold = 1e-4
            hitting_time = float('inf')
            for i, row in history.iterrows():
                if row['loss'] < loss_threshold:
                    hitting_time = i
                    break
            records.append(
                dict(
                    n_qubits=run.config['n_qubits'],
                    n_layers=run.config['n_layers'],
                    min_energy_gap=min_energy_gap,
                    fidelity=fidelity,
                    hitting_time=hitting_time
                )
            )
            print(records[-1])
    df = pd.DataFrame.from_records(records)
    if not resdir.exists():
        resdir.mkdir(exist_ok=True, parents=True)
    df.to_pickle(resdir / f'minloss.pkl')
    print('Done')
    return df


def retrieve_min_and_max(res, column):
    x = res.n_layers.unique()
    x.sort()
    y_mean = []
    y_min = []
    y_max = []
    for l in x:
        r = res[res.n_layers == l]
        y_mean.append(r[column].mean())
        y_min.append(r[column].min())
        y_max.append(r[column].max())
    y_mean = np.array(y_mean)
    y_min = np.array(y_min)
    y_max = np.array(y_max)
    return x, y_mean, y_min, y_max


def draw_optimization_energy_gap(df, linestyles):
    n_qubits_list = df.n_qubits.unique()
    n_qubits_list.sort()
    for i, n_qubits in enumerate(n_qubits_list):
        label = f'{n_qubits} Qubits'
        res = df[df.n_qubits == n_qubits]
        x, y_mean, y_min, y_max = retrieve_min_and_max(res, 'min_energy_gap')
        plt.plot(x, y_mean, linestyles[i],
                 linewidth=1.2, alpha=1.,
                 markersize=5,
                 label=label)
        plt.fill_between(x, y_min, y_max, alpha=0.35)

    plt.yscale('log')
    # plt.xlim(0, 155)
    plt.xlabel(r'$L$', fontsize=13)
    plt.ylabel(r'$| E(\mathbf{\theta}^*) - E_0 |$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='upper right')

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/ising_opt_energy_gap.pdf')
    plt.show()


def draw_fidelity(df, linestyles):
    n_qubits_list = df.n_qubits.unique()
    n_qubits_list.sort()
    for i, n_qubits in enumerate(n_qubits_list):
        label = f'{n_qubits} Qubits'
        res = df[df.n_qubits == n_qubits]
        x, y_mean, y_min, y_max = retrieve_min_and_max(res, 'fidelity')
        plt.plot(x, y_mean, linestyles[i],
                 linewidth=1.2, alpha=1.,
                 markersize=5,
                 label=label)
        plt.fill_between(x, y_min, y_max, alpha=0.35)

    # plt.yscale('log')
    # plt.xlim(0, 155)
    plt.xlabel(r'$L$', fontsize=13)
    plt.ylabel(r'$|\,\langle \psi(\mathbf{\theta^*})\, |\, \phi \rangle\, |^2$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='lower right')

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/ising_opt_fidelity.pdf')
    plt.show()


def draw_convergence_speed(df, linestyles):
    n_qubits_list = df.n_qubits.unique()
    n_qubits_list.sort()
    for i, n_qubits in enumerate(n_qubits_list):
        label = f'{n_qubits} Qubits'
        res = df[df.n_qubits == n_qubits]
        x, y_mean, y_min, y_max = retrieve_min_and_max(res, 'hitting_time')
        plt.plot(x, y_mean, linestyles[i],
                 linewidth=1.2, alpha=1.,
                 markersize=5,
                 label=label)
        plt.fill_between(x, y_min, y_max, alpha=0.35)

    # plt.yscale('log')
    # plt.xlim(0, 155)
    plt.xlabel(r'$L$', fontsize=13)
    plt.ylabel(r'$t^*$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='upper right')

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/ising_opt_convergence.pdf')
    plt.show()


def main():
    # Draw L vs. min_loss
    # datapath = 'results_ising_expressibility/20200830/minloss.pkl'
    datapath = None
    if datapath:
        df = pd.read_pickle(datapath)
    else:
        resdir = Path(f'results_ising_expressibility/{datetime.now().strftime("%Y%m%d")}')
        df = download_from_wandb(resdir)

    linestyles = ['-o', '-.o', '--o', ':o']
    draw_optimization_energy_gap(df, linestyles)
    draw_fidelity(df, linestyles)
    draw_convergence_speed(df, linestyles)


if __name__ == '__main__':
    main()
