import argparse

import matplotlib.pyplot as plt
import numpy as np

from . import eval

plt.style.use("tableau-colorblind10")

colors = ["#3a86ff", "#8338ec", "#ff006e", "#fb5607", "#ffbe0b"]

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 14

groups = dict(
    symsol1=dict(
        names=["cone", "cyl", "tet", "cube", "ico"],
        all_runs=[
            [
                "Ours w/o KP & IS",
                ["2uohhcy6", "2r7vd1vo", "ky0zords", "1658dav6", "3gu9wr9m"],
            ],
            [
                "Ours w/o KP",
                ["3cy3h59t", "5ld8nt89", "3ivh553o", "ep4fj3im", "lg67xe4y"],
            ],
            # ["Ours w/o KP", ['l6po3dyf', '2h30kgib', '1zqzj450', '19pyyju7', '36hvqfg2']],
            [
                "Ours w/o IS",
                ["3bn9wlhb", "ikadsujo", "3dcd0a5m", "36pzi6yo", "xph1lvtt"],
            ],
            [
                "Ours w/ cube KP",
                ["21emos69", "3jm12frs", "6rcgzofa", "sxbyi1w5", "1ffryeck"],
            ],
            ["Ours", ["q8jn5x0f", "15u0j7nf", "27j1jnu7", "2u5xkqc0", "2kkp80ye"]],
        ][::-1],
    ),
    symsol2=dict(
        names=["sphX", "cylO", "tetX"],
        all_runs=[
            ["Ours w/o KP & IS", ["2gd5wfhf", "20nusazu", "i507mkmm"]],
            ["Ours w/o KP", ["30h9jtv9", "qttv6zxi", "2gvx075u"]],
        ],
    ),
    symsol2_10k=dict(
        names=["sphX", "cylO", "tetX"],
        all_runs=[
            ["Ours w/o KP & IS", ["jnifms8h", "2z5tu4i1", "266str8g"]],
            ["Ours w/o KP", ["mmos5kss", "239s5635", "2q9v3rzz"]],
        ],
    ),
    tless=dict(
        names=["1", "14", "25", "27"],
        decimals=1,
        all_runs=[
            ["Ours w/o IS", ["j5tlp499", "2u2vf9ju", "1c99w969", "s2tdidlu"]],
            ["Ours", ["1dmzvzlb", "cfp61fkh", "11xk87bs", "2903di6r"]],
        ],
    ),
    hb=dict(
        names=["2", "7", "9", "21"],
        decimals=1,
        all_runs=[
            ["Ours w/o IS", ["1b9tga9t", "heuppl59", "bvb06l6o", "9gzg2c4n"]],
            ["Ours", ["116mhkmv", "lexo4tx1", "qffoc93y", "1prbniu5"]],
        ],
    ),
)

parser = argparse.ArgumentParser()
parser.add_argument("group", choices=groups.keys())
parser.add_argument("topk", type=int)
parser.add_argument("device")
parser.add_argument("--split", default="test")
parser.add_argument("--regular-grid", action="store_true")
parser.add_argument("--r", type=int, default=-1)
args = parser.parse_args()

group_name = args.group
group = groups[group_name]
plot = group_name == "symsol1"


if plot:
    plt.figure(figsize=(5, 3))
for i, (method, runs) in enumerate(group["all_runs"]):
    lls = []
    for name, run in zip(group["names"], runs):
        ll = eval.evaluate(
            run_id=run,
            device=args.device,
            topk=args.topk,
            overwrite=False,
            split=args.split,
            regular_grid=args.regular_grid,
        )
        # (n_images, n_syms, n_rec)
        # print(ll.shape)
        lls.append(ll.mean(axis=(0, 1)))
    lls = np.stack(lls)  # (n_obj, n_rec)

    # print table
    lls_r = lls[:, args.r]

    lls_r = [
        f'{ll:.{group.get("decimals", 2)}f}' for ll in [lls_r.mean()] + list(lls_r)
    ]
    print(" & ".join([method] + lls_r) + " \\\\")

    # plot figure
    lls = lls.mean(axis=0)  # (n_rec,)
    if plot:
        plt.plot(lls, label=method, c=colors[i])
if plot:
    plt.ylim(bottom=3)
    plt.xlim(left=2)
    plt.xticks([2, 3, 4, 5, 6])
    plt.legend()
    plt.xlabel("Pyramid level")
    plt.ylabel("Avg. log likelihood")
    plt.tight_layout(pad=0)
    plt.savefig("symsol1_plot.pdf")
    plt.show()
