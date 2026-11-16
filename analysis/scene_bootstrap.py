"""Recompute scene-bootstrap CIs from a pair of evaluator status files."""

import argparse
import json
import random
from collections import defaultdict

from .match_status import groups_of, gt_table, load_status

THRESHOLD = '0.25'
N_BOOT, SEED = 2000, 0


def e1vt_ids(table):
    cells = defaultdict(int)
    for row in table.values():
        if row['size'] == 'vt' and row['stratum'] == 'inside':
            cells[row['wh']] += 1
    return {gid for gid, row in table.items()
            if row['size'] == 'vt' and row['stratum'] != 'inside' and cells[row['wh']] >= 5}


def outside_ids(table):
    return {gid for gid, row in table.items() if 'E1_outside' in groups_of(row)}


def boot(status_a, status_b, members, table):
    by_parent = defaultdict(lambda: [0, 0, 0])
    for gid in members:
        a, b = status_a.get(str(gid)), status_b.get(str(gid))
        if a is None or b is None:
            continue
        p = by_parent[table[gid]['parent']]
        p[0] += 1
        p[1] += b[0] - a[0]
        p[2] += b[1] - a[1]
    parents = list(by_parent.values())
    total = [sum(p[i] for p in parents) for i in range(3)]
    rng = random.Random(SEED)
    d50, d75 = [], []
    for _ in range(N_BOOT):
        s0 = s1 = s2 = 0
        for _ in range(len(parents)):
            p = parents[rng.randrange(len(parents))]
            s0 += p[0]
            s1 += p[1]
            s2 += p[2]
        d50.append(100.0 * s1 / s0)
        d75.append(100.0 * s2 / s0)

    def ci(v):
        v = sorted(v)
        return [v[int(0.025 * N_BOOT)], v[int(0.975 * N_BOOT) - 1]]
    return {'n_gt': total[0], 'n_parents': len(parents), 'dR50': 100.0 * total[1] / total[0], 'dR50_CI95': ci(d50),
            'dR75': 100.0 * total[2] / total[0], 'dR75_CI95': ci(d75)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', help='AI-TOD-v2 COCO annotation JSON')
    parser.add_argument('baseline', help='baseline per-GT status JSON')
    parser.add_argument('variant', help='variant per-GT status JSON, same seed and split')
    args = parser.parse_args()
    table = gt_table(args.annotations)
    status_a = load_status(args.baseline, THRESHOLD)
    status_b = load_status(args.variant, THRESHOLD)
    groups = {'E1vt': e1vt_ids(table), 'E1_outside': outside_ids(table)}
    data = {name: boot(status_a, status_b, members, table) for name, members in groups.items()}
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
