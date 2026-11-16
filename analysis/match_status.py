"""Build coverage strata and load per-object matching results."""

import json
from functools import lru_cache
from pathlib import Path

from . import coverage


@lru_cache(maxsize=2)
def gt_table(gt_json):
    """{gt_id: dict(size, stratum, depth, parent, source, cat, wh)} for positive-area GTs."""
    d = json.loads(Path(gt_json).read_text(encoding='utf8'))
    ims = {im['id']: im for im in d['images']}
    table = {}
    for a in d['annotations']:
        x, y, w, h = a['bbox']
        if w <= 0 or h <= 0:
            continue
        im = ims[a['image_id']]
        r = coverage.INPUT / max(im['width'], im['height'])
        box = (x * r, y * r, (x + w) * r, (y + h) * r)
        m = coverage.best_depth(box)
        src, parent = coverage.source_and_parent(im['file_name'])
        size = 'vt' if a['area'] <= 64 else 't' if a['area'] <= 256 else 's' if a['area'] <= 1024 else 'm'
        table[int(a['id'])] = {'size': size, 'stratum': coverage.stratum(m), 'depth': m, 'parent': parent,
                               'source': src, 'cat': a['category_id'], 'wh': (int(round(w)), int(round(h)))}
    return table


def groups_of(row):
    """Endpoint groups a GT belongs to."""
    g = ['all', f"{row['size']}_all", f"{row['size']}_{row['stratum']}"]
    if row['size'] in ('vt', 't'):
        if row['stratum'] == 'inside':
            g.append('inside')
        else:
            g.append('E1')
            g.append(f"E1_{row['stratum']}")
    return g


def load_status(path, threshold='all'):
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob('*_per_gt_status.json'))
        assert len(files) == 1, f'expected one *_per_gt_status.json in {path}, found {len(files)}'
        path = files[0]
    data = json.loads(path.read_text(encoding='utf8'))
    if 'all' in data and isinstance(data['all'], dict):  # multi-threshold format
        return data[threshold]
    assert threshold == 'all', f'{path} only holds the all-detections status'
    return data
