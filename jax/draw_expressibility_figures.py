import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


def plot_expressibility_scatter_graph(results):
    markers = ['+', 'x']
    linestyles = ['-', '-.']
    for i, label in enumerate(sorted(results)):
        res = results[label]
        x, y_mean, y_std = sample_mean(res)
        plt.plot(x, y_mean, linestyles[i],
                 linewidth=1.15, alpha=0.95,
                 label=f'{label} (mean)')
        plt.scatter(res.n_layers, res.min_loss,
                    marker=markers[i], s=50,
                    alpha=0.42, label=f'{label} (samples)')

    plt.yscale('log')
    plt.xlabel(r'Number of layers, $L$', fontsize=12)
    plt.ylabel(r'Expressibility,  $\epsilon$', fontsize=12)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend()

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/expressibility.png')
    plt.show()


def plot_expressibility_sample_mean_graph(results):
    markers = ['o', 'x', '+', '>']
    linestyles = ['-', '-.', '--', ':']
    for i, label in enumerate(sorted(results)):
        res = results[label]
        x, y_mean, y_std = sample_mean(res)
        plt.errorbar(x, y=y_mean, yerr=y_std, label=label,
                     marker=markers[i], fillstyle='none',
                     linestyle=linestyles[i], linewidth=1.5,
                     elinewidth=1., fmt='o', capsize=2)
    plt.yscale('log')
    plt.xlabel(r'Number of layers, $L$', fontsize=12)
    plt.ylabel(r'Expressibility,  $\epsilon$', fontsize=12)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend()

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/expressibility-ver2.png')
    plt.show()


def main():
    # TODO(jdk): Let us download via wandb API.
    filepath = 'wandb_expressibility_minloss_20200822.csv'
    df = pd.read_csv(filepath, index_col=0, na_values=['undefined'])
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

    plot_expressibility_scatter_graph(results)


if __name__ == '__main__':
    main()
