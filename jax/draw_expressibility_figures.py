from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

matplotlib.rcParams['mathtext.fontset'] = 'stix'


FIRST_N_STEPS = 500


def download_from_wandb(resdir):
    project = 'expressibility'
    print(f'Downloading experiment results from {project}')
    print(f'| Results directory : {resdir}')

    api = wandb.Api()
    run_ids = TARGET_RUN_IDS.split('\n')
    records = []
    n_skipped = 0
    visited = set()
    for run_id in run_ids:
        if not run_id:
            n_skipped += 1
            continue
        # Pick the lastly resumed cases if there exist.
        run_id = run_id.split(',')[-1]
        if run_id in visited:
            raise ValueError(f'There is a duplicated run id {run_id}.')
        run = api.run(f'vqc-quantum/{project}/{run_id.strip()}')
        visited.add(run_id)
        print(run.name)
        history = run.history()
        min_loss = history[history._step < FIRST_N_STEPS].min_loss.min()
        records.append(
            dict(
                n_qubits=run.config['n_qubits'],
                n_layers=run.config['n_layers'],
                min_loss=min_loss,
            )
        )
        print(records[-1])
    df = pd.DataFrame.from_records(records)
    if not resdir.exists():
        resdir.mkdir(exist_ok=True, parents=True)
    df.to_pickle(resdir / f'minloss.pkl')
    print(f'Download done: there are {n_skipped} experimental cases to run.')
    return df


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


def draw_expressibility(df, linestyles):
    n_qubits_list = df.n_qubits.unique()
    n_qubits_list.sort()
    for i, n_qubits in enumerate(n_qubits_list):
        label = f'{n_qubits} Qubits'
        res = df[df.n_qubits == n_qubits]
        x, y_mean, y_min, y_max = retrieve_min_and_max(res, 'min_loss')
        plt.plot(x, y_mean, linestyles[i],
                 linewidth=1.2, alpha=1.,
                 markersize=5,
                 label=label)
        plt.fill_between(x, y_min, y_max, alpha=0.35)
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0, 1000)
    plt.xlabel(r'$L$', fontsize=13)
    plt.ylabel(r'$\varepsilon_m \ / \ 2^{n}$', fontsize=13)
    plt.grid(True, c='0.5', ls=':', lw=0.5)
    plt.legend(loc='upper right')

    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig/expressibility.pdf')
    plt.show()


def main():
    resdir = Path(f'results_expressibility/{datetime.now().strftime("%Y%m%d")}')
    # datapath = resdir / 'minloss.pkl'
    datapath = None
    if datapath:
        df = pd.read_pickle(datapath)
    else:
        df = download_from_wandb(resdir)

    linestyles = ['-o', '-.o', '--o', ':o']
    draw_expressibility(df, linestyles)


TARGET_RUN_IDS = """19d5fj2f
1cxwk3nj
2euuz66g
12ahow7d
2mmdzcjq
2xykygv5
3056nzvc
3ofj27wn
oxw2y6gi
1aikla9e
1y01z4mw
3vdhpfb1
16ochjn1
1j683pws
lpwdhuwb
lvxlo5qz
3l97xvs4
20wso2t9
282o1gh0
5e5fbwf7
1w3iazzv
khzy5xz9
3fmsok62
18oqkx1t
179w38e8
1iprijyc
2jflm2wb
3i3g67ax
2s03r34l
kfcuigzy
1q561gn9
lz87393x
wxkqgm1r
1kzl38jq
cyica5ea
vp44e11v
2ryab27a
2tacdq9j
3omvr6lv
6nwow2n6
i48m0pzw
2cj3twe9
1lmcfbnl
3aqpcn1j
1dm5qc2f
ew28htxf
20xwqyab
1pe2z1ts
35e6lkb9
2rzsi1t0
wbz7vpqy
19lcuqgj
248aw00o
1yult6xl
24c2zv90
1z9z2cmb
q6dirawb
1yzm3e8s
150fvtsz
34ktxxwf
2tc4gcc2
3nk7r23l
1gkbou4v
34idtg1n
uwq3nz59
3l0p45p1
11h5obr1
1vfcdqf2
3in9dyya
7jztzb1j
1ej48cfn
3m5y3x6z
2htxpyub
23k6gh8i
7nktbedw
2s75jg35
1aw20j8s
3ffjzcha
2s0mmv7k
jfk5w55r
ab8oozmz
3d3emz6w
n7q3rag5
21qogn69
1z3d8hee
3c7yizht
1akoz83x
2ekpha5y
2mjtzx9t
2cbn55tz
1d4wkyn7
3hd8w7y3
3jwnpox1
2ryz0q8c
2ifshl6q
2wkdxr6c
y3qg8zyw
26utj3ta
100xxc3r
3cdqlcuv
30uh39bx
3gphbqtv
11gny80k
2bzt0j0u
2feu94tr
14pb56jl
3t6acpla
2aqmbmmz
l6vg2gp3
q905bdmy
ekw9rvan
1thkzurr
2qoxf29o
1pnd2dzc
3bhfwpo3
2at37h5i
3vre9lka
cv3mqdd8
hifz1s03
11ein29o
2yeqozfu
13pnc3sf
qepwb6m8
e8oag12h
1tgt05sk
1vwkygdw
2ap6yg7q
10u9arac
x6rsbp9z
3qvk2203
3rm15s0p
2agngo9w
2owl25qc
1eozgfys
119lihks
1fo63ftx
1a3jyzx3
28g62ahs
32aorvtt
351bcv1x
38jxau1k
kj66e5ui
3rx8w7lj
xaiods01
2hr5dko2
2kn3b4tt
35kezqji
3q7bvlbw
10squ2ln
18j3ycpi
2gseewey
101psl3z
2ybm12w2
1rapmb1x
1vr1x65d
6ozgwp9d
1zwq6321
13b2hapm
191y2zag
2f6calbh
dyv1qhxj
20sqgavh
288chjzz
1eff252l
3avkrbj2
3nnk9id0
2bkrl3vl
8l3xoq0q
2q6ka4s5
2zbsyhvl
5nnq2d7k
2pjfaogm
wedk3c8t
1nk1hgut
3ehsu85m
pf2o3wiy
a2996dk6
2679olgc
5inepzz6
f2njtbi8
ok8uy4mm
5vhh4qjm
2zvrb78q
1of2asz6
2fg1utb7
184s1xvb
3mussr5z
24w6ajfb
3qlts1xi
pv6kys3q
fih9yx9o
27ztesv2
belhpbb0
78cc04ax
iudbubeb
1yx7w9ax
3osu9gg7
363xlkr6
s4sh3bwd
2k1its9t
3mej929x
90sx3k8d
3icplrcc
1j7ql88f
29qusdk1
22nlgyf7
1xd2lmh8
2jibtn0i
1ylg9oic
30ef7knl
3buf8a2c
1xirwg83
132dwq8p
2ctodbxo
34p89c7l
239ig7ll
2bzvde55
364mcjzd
6f1p8uzr
s2rvwjdu
3it7z6b3
150udlop
pz9c4wih
122cwots
1n3jnwai
tzj2gku8
32fae4ox
wchtvpa7
3urifh7g
2334pdxu
3s8dhz9q
1xuwjeg6
10b2n2bs
2vrx5990
d1qel0dk
kv5k2u4j
3331yqvk
281xcqhe
feky4vpf
1g79fgzn
2tacuzst
1xpl029r
qvhfmrtx
5ebfmgc2
36pg7rkq
12xbhnrb
2ps1jrzc
334y9ccj
28csovh8
12zkqu9d
1ken3vbb
1efpzzfz
2sfnztoh
1ldvrbda
wndlsij5
8ise8zax
38g8sf3u
2v68tuu8
1es67mlc
1cu1ppa0
1e5cbzkn
8uj1w1ry
2lnfydxa
32zllcvk
3sbcchf2
36b5nm91
2slvvkpa
1yeq8r34
gl9e3a5f
3apoyyr6
1pl7u7nl
3cxzdj6p
1w4ahryw
5c3z1eov
vm5gcm46
2ifqsu6x
2im7fgey
ms1oq9is
11hpiux1
vqp1k0mb
3a95cfzl
91e8rtj3
1u26pgzo
2rbleln4
23uvtykj
3emp7nco
138oofpi
2r683pg4
3cg99jwm
3gjriwq3
ngqddh2i
16g6bnd2
1qj6d7sw
34vietx1
1ylsquz2
hrgeuhlk
3jr390e8
3ne9yy50
p0zjcacr
3txz5eij
39dklwko
2nhjoc78
v093o3pv
1rv8ot8e
5l065inl
m7s92omm
qax30303
1syojsoc
27mdbr4l
1wtqpggt
2rmxmtq8
36befity
1s9zx30c
bod2bj1o
9eqkaajz
q9bt7j05
3eehrw2p
3k2s59cm
3d1mxfep
3sq9ni3f
2wflew0o
3gn1o5zn
3at1zb86
2msjt6lb
37az2spr
32gqi9ub
3bffj4fc
1zh7pty8
2ksoz696
1fl0yvd1
2ezz4c2u
31v317h2
2lplvvaw
1kbd4x45
1vyyusvo
1plhnvqc
7l66iqpq
r41v6jm8
2vjqgpf4
1jbpvn8w
30tha2ic
3pz33oqz
2kwgz2j7
3jr7g29d
1kayn91m
sofgkeya
2q7p1oo8
qt0piskm
vlzi665e
20u6cq2m
3j1cq2eu
24eq84iy
3p4wrmht
6z433rag
q9i8ru30
1vpjtby7
24ka5tti
3or6o50a
1ysgnkfm
1h3g7rw0
1foaugxv
buqjzb2g
2xkfbrem
3stl6u90
15ab3o4f
34l1ix3v
3rnnakx1
2hpl4y6h
u31e3544
36z141mg
izhd5p3l
3lk2gxmc
31xj57ly
2ne43er6
1h5kp7h7
11dpoldv
3l6v5s2r
10szw0o1
3btfecfq
3ft67sd3
1f91z5l9
339fb3jn
31p5qtcl
18pghpgk
1kpihz46
3233kyr1
16ky6sdo
1r4ws50j
3go4tksz
1otr0j24
8lg6zh5h
3d31yjwr
anu259qi
29bhvdc0
337fw9hi
26jni69t
2os05rkh
35limav5
xe2v0ljg
30476qdj
1vbh4wr8
2iroha48
3unnt1f7
3s933dgq
pgiil725
8r6mg85g
v9mo7i6g
1bv0h1w1
25loirwe
15yuxh1x
1cvq7b6e
1irt1rbg
3tb04hur
11a6goe1
3js4dsog
2xjxmjsp
vtjnxkb2
jrfjnis0
14wzlimw
22hqcrv5
qdc3mapd
o69ze7i2
7dhq0jeu
6cvi62xg
30qdiu3y
255wzxzi
1swzevjn
226awi76
14e2ict9
l53o4kzv
jf2di3hw
6jhg39hx
1hfuezkh
xk8jf6ux
21p7vi1u
3m0tht0d
rb6slzzp
f8z3byxe
2a3t0nsu
gom7vbc1
3u8sxfif
1hve9zq9
f97t8j0z
16eet9dk
2dj6wb4a
2ff3n2tt
x3gasn0x
3kyr06qk
1wc18irn
2ll4oudh
fjdx3kv6
69apctbm
ilfnswcs
1bp34fjp
1kmee55e
3fnqsblb
2eo2ray1
vxfd5vk6
3dshzwhx
33pnm17f
2xbvzr9d
1dkqh5f1
cedz4p0g
1dycjczg
97abrbmv
1swzp0es
1ai1zndv
5cvjja3i
ujx0qnt0
1bfd5x4o
1jegcv8d
1xkvrpht
5l1gk99j,318155lh
2nwfpgy0
22xyoky9
igvdjjlb,rq0okqrx
191ltnl8



3r1f63xt
1g34n0jg
rgk3xsl2
1156ddzn
2uqgyzj3
27oorkxt

16ga9nlb
1lrnu9r3
3acdca95
3rc47llo
2ftzn9nc

6w3evxo0















"""


if __name__ == '__main__':
    main()
