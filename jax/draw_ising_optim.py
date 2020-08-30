import matplotlib
import matplotlib.pyplot as plt

import wandb


matplotlib.rcParams['mathtext.fontset'] = 'stix'

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def main():
    n_qubits = 8
    n_layers_list = [32, 64, 80, 96]
    project = 'IsingModel'
    target_cfgs = {
        'config.n_qubits': n_qubits,
        'config.n_layers': {"$in": n_layers_list},
        'config.g': 2,
        'config.h': 0,
        'config.lr': 0.05,
        'config.seed': 99,
        'config.scheduler_name': 'constant',
    }
    print(f'Downloading experiment results from {project}')
    print(f'| Target constraints: {target_cfgs}')

    api = wandb.Api()
    runs = api.runs(project, filters=target_cfgs)

    history = {}
    for run in runs:
        if run.state == 'finished':
            print(run.name)
            n_layers = run.config['n_layers']
            h = run.history()
            # Theoretically E(\theta) >= E_0 and fidelity <= 1.
            # If it is negative, it must be a precision error.
            h['loss'] = h['loss'].clip(lower=0.)
            h['fidelity/ground'] = h['fidelity/ground'].clip(upper=1.)
            history[n_layers] = h
    print('Download done')
    assert set(history.keys()) == set(n_layers_list)

    linestyles = ['-', '-.', '--', ':']
    linewidths = [1.2, 1.2, 1.3, 1.4]

    xlim = 0, 500

    plt.subplot(211)
    for i, n_layers in enumerate(n_layers_list):
        h = history[n_layers]
        plt.plot(h._step, h.loss, linestyles[i],
                 color=color_list[i],
                 linewidth=linewidths[i],
                 alpha=1.,
                 markersize=5,
                 label=f'L={n_layers}')
    plt.xlim(*xlim)
    plt.yscale('log')
    plt.ylabel(r'$E(\mathbf{\theta}) - E_0$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    # plt.legend(loc='upper right')

    plt.subplot(212)
    for i, n_layers in enumerate(n_layers_list):
        h = history[n_layers]
        plt.plot(h._step, h['fidelity/ground'], linestyles[i],
                 color=color_list[i],
                 linewidth=linewidths[i],
                 alpha=1.,
                 markersize=5,
                 label=f'L={n_layers}')

    plt.xlim(*xlim)
    plt.xlabel('Optimization Steps', fontsize=13)
    plt.ylabel(r'$|\,\langle \psi(\mathbf{\theta^*})\, |\, \phi \rangle\, |^2$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('fig/ising_optimization.pdf')
    plt.show()


if __name__ == '__main__':
    main()
