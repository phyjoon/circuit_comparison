import ast
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def retrieve_values_from_name(fname):
    return re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", fname)


def download_from_wandb(resdir):
    project = 'SYK4Model'
    # target_cfgs = {
    #     'config.lr': 0.05,
    #     'config.scheduler_name': 'constant',
    #     'config.seed_SYK': 1
    # }
    print(f'Downloading experiment results from {project}')
    print(f'| Results directory : {resdir}')
    # print(f'| Target constraints: {target_cfgs}')

    api = wandb.Api()
    # runs = api.runs(project, filters=target_cfgs)
    run_ids = TARGET_RUN_IDS.split('\n')
    records = []
    visited = set()
    for run_id in run_ids:
        if run_id in visited:
            raise ValueError(f'There is a duplicated run id {run_id}.')
        run = api.run(f'vqc-quantum/{project}/{run_id.strip()}')
        visited.add(run_id)
        if run.state == 'finished':
            print(run.name)
            if 'eigenvalues' not in run.config:
                print(f'| Skip this run because eigenvalues info is not in config.')
                continue
            history = run.history()
            eigvals_str = run.config['eigenvalues'].replace('\n', '')
            eigvals_str = re.sub(' +', ',', eigvals_str)
            try:
                ground_state_energy = ast.literal_eval(eigvals_str)[0]
            except ValueError as e:
                print(str(e))
                print(f'Parsing Error: eigvals_str: {eigvals_str}')
                print(f'Retry to parse the first element')
                # Some runs logs eigenvalues in the following format.
                #   [-5.69803132e-02+0.00000000e+00j ... 1.10259914e-16-4.19720017e-16j]
                # Due to dots let us parse the first element and then get its real part.
                v_str = eigvals_str.split(',')[0].strip('[')
                print(f' - Retried string: {v_str}')
                ground_state_energy = ast.literal_eval(v_str).real
            best_step = history.loss.argmin()
            min_energy_gap = np.abs(history.loss[best_step] - ground_state_energy)  # |E(\theta) - E0|
            fidelity = history['fidelity/ground'][best_step]
            if run.config["n_qubits"] % 4 == 2:  # SYK4 is degenerated.
                fidelity += history['fidelity/next_to_ground'][best_step]
            loss_threshold = 1e-4
            hitting_time = float('inf')
            for i, row in history.iterrows():
                if np.abs(row['loss'] - ground_state_energy) < loss_threshold:
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

    # plt.xscale('log')
    plt.yscale('log')
    # plt.xlim(0, 155)
    plt.xlabel(r'$L$', fontsize=13)
    plt.ylabel(r'$E(\mathbf{\theta}^*) - E_0$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='upper right')

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/syk_opt_energy_gap.pdf')
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

    # plt.xscale('log')
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
    plt.savefig('fig/syk_opt_fidelity.pdf')
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
    plt.savefig('fig/syk_opt_convergence.pdf')
    plt.show()


def main():
    # Draw L vs. min_loss
    resdir = Path(f'results_syk_vqe/{datetime.now().strftime("%Y%m%d")}')
    # datapath = resdir / 'minloss.pkl'
    datapath = None
    if datapath:
        df = pd.read_pickle(datapath)
    else:
        resdir = Path(resdir)
        df = download_from_wandb(resdir)

    linestyles = ['-o', '-.o', '--o', ':o']
    draw_optimization_energy_gap(df, linestyles)
    draw_fidelity(df, linestyles)
    draw_convergence_speed(df, linestyles)


TARGET_RUN_IDS = """51vzyt5w
1iiwwc2s
2yk3sqgr
1jg2ugm9
1yswb81l
3ss3r7xi
18qaij9g
9hr5qpqg
396lm5vb
38z0jgt1
1c8f7jwa
33vp1cqv
2h71q1me
10rpvaap
3es61gp1
3ujou9nf
2jip7auo
38giwbt9
3473v16h
28tut4fn
2h7o9sr1
3md7d2g8
25c4kvc6
2s8xi4ni
1rgkqhgr
3kez2ssw
2bp76ggs
2dn3p80o
266lgh7i
89radulq
1k0p0mcv
2eyxnpay
1wofm4qi
3gnvc1f2
e27u581w
202sw0yw
qjrcwtbf
x5879m11
37mpqzw0
xxh8qu4y
1vo577mm
29d8h2k3
23re9wkz"""


if __name__ == '__main__':
    main()
