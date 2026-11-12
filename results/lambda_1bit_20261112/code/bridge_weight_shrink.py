"""Weight-space projection of each row's error on the FP weight, per changed 2-D tensor: p = <Wq - W, W> / <W, W> (H8 of baselines/RESULTS.md)."""
import json, sys, glob, torch
from safetensors import safe_open
FP = '/projects/yanlongvla/models/groot_n17_simpler_bridge/fp'
ROWS = dict(tcq_w2_rht='/projects/yanlongvla/models/groot_ptq/tcq_w2_rht/simpler_bridge', tcq_w2_rht_d1='/projects/yanlongvla/models/groot_ptq/tcq_w2_rht_d1/simpler_bridge',
            tcq_raqft_d1_s='/projects/yanlongvla/models/groot_ptq/tcq_raqft_d1_s/simpler_bridge', qtip_w2_chamadol='/projects/yanlongvla/runs/bridge_baselines/ckpt/qtip_w2/bridge')
fp = {}
for f in sorted(glob.glob(FP + '/model-*.safetensors')):
    h = safe_open(f, 'pt')
    for k in h.keys(): fp[k] = (h, k)
out = {}
for row, d in ROWS.items():
    h = safe_open(d + '/model.safetensors', 'pt'); keys = set(h.keys())
    assert keys == set(fp), (row, len(keys ^ set(fp)))
    ps, es, n_other = [], [], 0
    num = den = e2 = 0.0
    for k in sorted(keys):
        a = fp[k][0].get_tensor(k).double(); b = h.get_tensor(k).double()
        if torch.equal(a, b): continue
        if a.ndim != 2: n_other += 1; continue
        d_ = b - a; aa = (a * a).sum().item(); da = (d_ * a).sum().item(); dd = (d_ * d_).sum().item()
        ps.append(da / aa); es.append(dd / aa); num += da; den += aa; e2 += dd
    t = torch.tensor(ps); e = torch.tensor(es)
    out[row] = dict(changed_2d=len(ps), changed_other=n_other, projection_min=t.min().item(), projection_median=t.median().item(), projection_max=t.max().item(),
                    negative=int((t < 0).sum()), pooled_projection=num / den, rel_err_energy_median=e.median().item(), pooled_rel_err_energy=e2 / den,
                    shrink_share_of_error_energy=(num * num / den) / e2)
    print(row, json.dumps(out[row]), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
