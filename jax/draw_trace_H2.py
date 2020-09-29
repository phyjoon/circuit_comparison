import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

matplotlib.rcParams['mathtext.fontset'] = 'stix'

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# The numeric values are obtained by using another framework.
syk_tr_h2 = [
    (10, 245.97),
    (11, 546.295),
    (12, 1207.7),
    (13, 2652.74),
    (14, 5789.44),
    (15, 12559.8),
    (16, 27100.4),
    (17, 58190.6),
    (18, 124403.),
    (19, 264916.),
    (20, 562150.),
    (21, 1.18909 * 1e6),
    (22, 2.508 * 1e6),
    (23, 5.27602 * 1e6),
    (24, 1.10727 * 1e7),
    (25, 2.31876 * 1e7),
    (26, 4.84612 * 1e7),
    (27, 1.01097 * 1e8),
    (28, 2.10547 * 1e8),
    (29, 4.37808 * 1e8),
    (30, 9.09053 * 1e8),
    (31, 1.885 * 1e9),
    (32, 3.90383 * 1e9),
    (33, 8.07537 * 1e9),
    (34, 1.66862 * 1e10),
    (35, 3.44437 * 1e10),
    (36, 7.10301 * 1e10),
    (37, 1.46346 * 1e11),
    (38, 3.01266 * 1e11),
    (39, 6.1968 * 1e11),
    (40, 1.27366 * 1e12),
    (41, 2.61592 * 1e12),
    (42, 5.36906 * 1e12),
    (43, 1.10126 * 1e13),
    (44, 2.25741 * 1e13),
    (45, 4.62463 * 1e13),
    (46, 9.46887 * 1e13),
    (47, 1.9377 * 1e14),
    (48, 3.96325 * 1e14),
    (49, 8.10222 * 1e14),
    (50, 1.65559 * 1e15)
]

ising_tr_h2 = [
    (7, 4480),
    (8, 10240),
    (9, 23040),
    (10, 51200),
    (11, 112640),
    (12, 245760),
    (13, 532480),
    (14, 1146880),
    (15, 2457600),
    (16, 5242880),
    (17, 11141120)
]


def curve(x, a, b):
    return a * (2 ** x) * (x ** b)


def draw_ising_trH2():
    h2 = np.array(ising_tr_h2)
    x, y = h2[:, 0], h2[:, 1]
    p, _ = scipy.optimize.curve_fit(curve, x, y)
    print(p)
    plt.plot(x, y, 'o',
             markersize=5,
             color=color_list[0], label='Ising')
    plt.plot(x, curve(x, *p), '-.',
             markersize=5,
             linewidth=1.2,
             color=color_list[0], label='Ising (fitting)')
    plt.yscale('log')
    plt.xlabel(r'$n$', fontsize=13)
    plt.ylabel(r'$\mathrm{Tr}\, (\, \mathcal{H}^2)$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='lower right')

    fitted_curve = r'$\mathrm{Tr}(\mathcal{H}^2) \approx %.1f \, n^{%.1f} \, 2^n$' % (p[0], p[1])
    ax = plt.gca()
    ax.text(
        0.11, 0.76,
        fitted_curve,
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            edgecolor='none',
            facecolor='white',
            alpha=0.7
        )
    )

    plt.tight_layout()
    plt.savefig('fig/trH2_ising_tight.pdf', bbox_inches='tight')
    # plt.show()


def draw_syk_trH2():
    h2 = np.array(syk_tr_h2[:15])
    x, y = h2[:, 0], h2[:, 1]
    p, _ = scipy.optimize.curve_fit(curve, x, y)
    print(p)
    plt.plot(x, y, 'o',
             markersize=5,
             color=color_list[1], label='SYK')
    plt.plot(x, curve(x, *p), ':',
             markersize=5,
             linewidth=1.2,
             color=color_list[1], label='SYK (fitting)')
    plt.yscale('log')
    plt.xlabel(r'$n$', fontsize=13)
    plt.ylabel(r'$\mathrm{Tr}\, (\, \mathcal{H}^2)$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='lower right')

    fitted_curve = r'$\mathrm{Tr}(\mathcal{H}^2) \approx %.3f \, n^{%.1f} \, 2^n$' % (p[0], p[1])
    ax = plt.gca()
    ax.text(
        0.55, 0.42,
        fitted_curve,
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            edgecolor='none',
            facecolor='white',
            alpha=0.7
        )
    )


def main():
    draw_ising_trH2()
    draw_syk_trH2()
    axes = plt.gca()
    axes.xaxis.set_major_locator(plt.MultipleLocator(2))
    # axes.spines['right'].set_visible(False)
    # axes.spines['top'].set_visible(False)
    plt.savefig('fig/trace_H2.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
