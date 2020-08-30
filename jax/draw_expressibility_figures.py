import re
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'stix'


def parse_name(name):
    parsed = re.findall(r"\d+", name)
    n_qubits, n_layers = int(parsed[0]), int(parsed[1])
    if 'BS' in name:
        seed = int(parsed[3])
        lr = float('.'.join(parsed[4:]))
    else:
        seed = int(parsed[2])
        lr = float('.'.join(parsed[3:]))
    return dict(n_qubits=n_qubits, n_layers=n_layers, seed=seed, lr=lr)


def sample_mean(res):
    x = res.n_layers.unique()
    x.sort()
    y_mean = []
    y_std = []
    for l in x:
        r = res[res.n_layers == l]
        y_mean.append(r.min_loss.mean())
        y_std.append(r.min_loss.std())
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    return x, y_mean, y_std


def compute_min_and_max(res):
    x = res.n_layers.unique()
    x.sort()
    y_mean = []
    y_min = []
    y_max = []
    for l in x:
        r = res[res.n_layers == l]
        y_mean.append(r.min_loss.mean())
        y_min.append(r.min_loss.min())
        y_max.append(r.min_loss.max())
    y_mean = np.array(y_mean)
    y_min = np.array(y_min)
    y_max = np.array(y_max)
    return x, y_mean, y_min, y_max


def plot_expressibility_fill_between(results):
    linestyles = ['-o', '-.x', '-->', ':^']
    for i, label in enumerate(
            sorted(results, key=lambda s: int(s.split(' ')[0]))):
        res = results[label]
        x, y_mean, y_min, y_max = compute_min_and_max(res)
        plt.plot(x, y_mean, linestyles[i],
                 linewidth=1.2, alpha=1.,
                 markerfacecolor='none', markersize=5,
                 label=label)
        plt.fill_between(x, y_min, y_max, alpha=0.35)

    plt.yscale('log')
    plt.xlim(0, 155)
    plt.xlabel(r'$L$', fontsize=13)
    plt.ylabel(r'$\varepsilon_m \ / \ 2^{n}$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='right')

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/expressibility.pdf')
    plt.show()


def load_df(logdir):
    # TODO(jdk): Let us download via wandb API.
    resdir = Path(logdir)
    df = None
    for f in resdir.glob('*.csv'):
        _df = pd.read_csv(f, index_col=0, na_values=['undefined'])
        if df is None:
            df = _df
        else:
            df = df.join(_df)
        print(f, _df.shape, df.shape)
    return df


def main():
    df = load_df('results_expressibility/200828')
    min_values = df.min()  # eq. (2)

    results = {}
    for name, min_loss in min_values.iteritems():
        result = parse_name(name)
        result['min_loss'] = min_loss
        label = f"{result['n_qubits']} Qubits"
        if label not in results:
            results[label] = []
        results[label].append(result)
    for label in results:
        res = pd.DataFrame.from_records(results[label])
        results[label] = res
        print('n_qubits:', res.n_qubits.unique())
        print('n_layers:', res.n_layers.unique())
        for l, d in res.groupby(by=['n_layers']):
            print(f'n_layers: {l:3d},\tn_samples: {len(d)}')

    plot_expressibility_fill_between(results)


if __name__ == '__main__':
    main()
