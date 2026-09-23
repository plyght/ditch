"""check_abliteration.py CUT OUT: which matrices a one-trial export changed, and each edit
against heretic's norm-preserving orthogonalisation recomputed here, independently of ditch.

    ditch CUT --n-trials 1 --n-startup-trials 1 --expert-selection broad \
        --dump-directions OUT.dirs.safetensors --trial-index 1 --model-action save \
        [--export-dtype f32] -o OUT ... > OUT.log
    python3 tools/check_abliteration.py CUT OUT [--json FILE]   # SAMPLE=4: four experts per stacked tensor

The trial's parameters are read from OUT.log (the printed min_weight is a fraction of
max_weight, heretic's parameterisation), the directions from OUT.dirs.safetensors
(`--dump-directions`). The original weights are dequantised (and rounded to bf16) as ditch
does. For every changed matrix the exact edit W' - W is recomputed (rows = the residual
dimension; W' = n * rownorm(W/n - lambda v (v^T W/n))) and compared with the exported
difference, next to the best any rank-3 delta can do (ditch stores a rank-3 delta). A bf16
export is also compared bit for bit with bf16(W + D3), D3 the optimal rank-3 delta.
"""
import json, re, sys, os, math, torch
sys.path.insert(0, '/home/user/ditch/tools')
from lazy_checkpoint import LazyCheckpoint
from ref_lazy_moe import dequant_expert
from safetensors import safe_open
cut, out = sys.argv[1], sys.argv[2]
json_out = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
log = open(out + '.log', errors='replace').read()
# --- trial parameters (the "Restoring model from trial" block) ---
blk = log[log.rindex('* Parameters:'):]
P = {}
for m in re.finditer(r'\* ([a-z_.]+) = ([^\n]+)', blk):
    P[m.group(1)] = m.group(2).strip()
scope = P.get('direction_scope', 'per layer')
def comp_params(c):
    g = lambda k: float(P[f'{c}.{k}'])
    return dict(maxw=g('max_weight'), pos=g('max_weight_position'), minw=g('min_weight') * g('max_weight'), dist=g('min_weight_distance'))  # printed min_weight is a fraction of max_weight (heretic's parameterisation)
def kernel(p, layer):
    d = abs(layer - p['pos'])
    if d > p['dist']: return None
    w = p['maxw'] + (d / p['dist']) * (p['minw'] - p['maxw'])
    return None if w == 0 else w
comps = {c: comp_params(c) for c in ('attn.o_proj', 'mlp.down_proj') if f'{c}.max_weight' in P}
# --- directions ---
with safe_open(out + '.dirs.safetensors', 'pt') as f: dirs = f.get_tensor('directions').double()
entries, hidden = dirs.shape
def direction(li):
    if scope.startswith('per layer'): return dirs[li + 1]
    di = float(P['direction_index']) + 1.0; i = int(math.floor(di)); fr = di - i
    a = dirs[min(i, entries - 1)]; b = dirs[min(i + 1, entries - 1)]
    v = a + fr * (b - a); return v / v.norm()
# --- weights ---
src = LazyCheckpoint(cut)
cfg = json.load(open(os.path.join(cut, 'config.json')))
IN_OUT_EXPERTS = (cfg.get('text_config', cfg).get('model_type') == 'gpt_oss')  # bf16 gpt-oss experts are x @ W
q = cfg.get('quantization_config') or (cfg.get('text_config') or {}).get('quantization_config') or {}
wq = next(iter((q.get('config_groups') or {}).values()), {}).get('weights', {}) if q.get('config_groups') else {}
def orig(name):  # float32 view of a source tensor, dequantised (and rounded to bf16) as ditch does
    base = name[:-len('.weight')] if name.endswith('.weight') else name
    keys = src.keys()
    if base + '_blocks' in keys:  # gpt-oss MXFP4 experts, transformers' own decoding ([E, in, out])
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors
        return convert_moe_packed_tensors(src.tensor(base + '_blocks'), src.tensor(base + '_scales')).float()
    if name in keys and src.header[name]['dtype'] not in ('F8_E4M3', 'U8', 'I8', 'I32'):
        return src.tensor(name).float()
    if name in keys and src.header[name]['dtype'] == 'F8_E4M3':
        # FP8 with a per-tensor, per-expert ([E, 1, 1]) or per-block scale
        sname = next((c for c in (base + '.weight_scale_inv', name + '_scale_inv', base + '.weight_scale', name + '_scale') if c in keys), None)
        w = src.tensor(name).float(); s = src.tensor(sname).float()
        if s.dim() == 0 or s.numel() == 1 or (s.dim() == w.dim() and all(a in (1, b) for a, b in zip(s.shape, w.shape))):
            return (w * s).to(torch.bfloat16).float()
        bs = q.get('weight_block_size') or [-(-w.shape[-2] // s.shape[-2]), -(-w.shape[-1] // s.shape[-1])]
        s = s.repeat_interleave(bs[0], -2)[..., : w.shape[-2], :].repeat_interleave(bs[1], -1)[..., : w.shape[-1]]
        return (w * s).to(torch.bfloat16).float()
    return dequant_expert(src, base, wq).to(torch.bfloat16).float()
exp_files = [os.path.join(out, f) for f in sorted(os.listdir(out)) if f.endswith('.safetensors')]
changed, unchanged, checks, bits = [], 0, [], []
EXP_BF16 = any('BF16' in open(p,'rb').read(200000).decode('latin1') for p in exp_files)
def exact_edit(W, v, lam):  # rows of W are the residual (hidden) dimension
    W = W.double(); n = W.norm(dim=1, keepdim=True); Wn = torch.where(n > 0, W / n, torch.zeros_like(W))
    a = v @ Wn; Wp = Wn - lam * torch.outer(v, a); Wp = Wp / Wp.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (n * Wp) - W
def rank_err(D, r=3):
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    return float(torch.sqrt((S[r:] ** 2).sum()) / torch.sqrt((S ** 2).sum()))
def classify(name):
    if re.search(r'(down_proj|\.w2\b|dense_4h_to_h|shared_expert|experts|latent_up|mlp\.up_proj_latent|output_linear|shared_mlp|block_sparse_moe|feed_forward\.w2|c_proj)', name) and 'attn' not in name: return 'mlp.down_proj'
    return 'attn.o_proj'
for fpath in exp_files:
    with safe_open(fpath, 'pt') as f:
        for name in f.keys():
            if name not in src.keys() and name + '_blocks' not in src.keys() and name.replace('.weight', '') + '.weight_scale_inv' not in src.keys() and not any(k.startswith(name[:-7]) for k in src.keys() if name.endswith('.weight')):
                continue
            shp = f.get_slice(name).get_shape()
            big = len(shp) >= 2 and shp[0] * shp[1] * (shp[2] if len(shp) > 2 else 1) > 300_000_000 and name in src.keys() and src.header[name]['dtype'] in ('BF16', 'F16', 'F32')
            if big:  # large plain tensors (embeddings, heads): compared in row chunks
                sl = f.get_slice(name); diff = False
                with safe_open(os.path.join(cut, 'model.safetensors'), 'pt') if os.path.exists(os.path.join(cut, 'model.safetensors')) else None as g:
                    gs = g.get_slice(name)
                    for r0 in range(0, shp[0], 8192):
                        if not torch.equal(sl[r0:r0 + 8192].float(), gs[r0:r0 + 8192].float()): diff = True; break
                if not diff: unchanged += 1; continue
                print('  large tensor changed:', name); changed.append(name); continue
            E = f.get_tensor(name).float()
            try: O = orig(name)
            except Exception as e: print('  (cannot read original', name, e, ')'); continue
            if O.shape != E.shape:
                print('  shape differs', name, tuple(O.shape), tuple(E.shape)); continue
            d = (E - O).abs().max().item()
            if d == 0 or d < 1e-7 * max(O.abs().max().item(), 1e-30): unchanged += 1; continue
            changed.append(name)
            m = re.search(r'layers\.(\d+)\.', name); li = int(m.group(1)) if m else None
            comp = classify(name); lam = kernel(comps[comp], li) if (li is not None and comp in comps) else None
            ne = E.shape[0] if E.dim() == 3 else 0
            sample = range(ne) if not os.environ.get('SAMPLE') or ne <= int(os.environ['SAMPLE']) else sorted(set([0, 1, ne // 2, ne - 1]))
            mats = [(E[e], O[e], f'{name}[{e}]') for e in sample] if E.dim() == 3 else [(E, O, name)]
            for Em, Om, label in mats:
                if (Em.shape[0] != hidden and Em.shape[1] == hidden) or (IN_OUT_EXPERTS and re.search(r'experts\.(down_proj|gate_up_proj)$', name)):
                    Em, Om = Em.T, Om.T  # stored [in, out]
                D = (Em - Om).double()
                if lam is None or Em.shape[0] != hidden:
                    checks.append((label, comp, li, None, None, None)); continue
                De = exact_edit(Om, direction(li).double(), lam)
                if EXP_BF16:
                    # bf16 export: the exported bits against bf16(W + D3), D3 the optimal rank-3 delta
                    U, S, Vh = torch.linalg.svd(De, full_matrices=False)
                    D3 = (U[:, :3] * S[:3]) @ Vh[:3]
                    pred = (Om.double() + D3).float().to(torch.bfloat16).float()
                    same = float((pred == Em).float().mean())
                    # bf16 steps between the two values, on weights that are not zero (a zero weight
                    # holds only its tiny delta, whose rounding says nothing); and the zeros' worst
                    # absolute difference relative to the mean |W|
                    # elements more than one bf16 step apart (at the larger of the two magnitudes) and
                    # by more than 1e-5 of the mean |W| (cancellation leaves values near zero whose steps are tiny)
                    step = torch.exp2(torch.floor(torch.log2(torch.maximum(pred.abs(), Em.abs()).clamp_min(1e-30))) - 7)
                    dif = (pred - Em).abs()
                    far = int(((dif > step * 1.0001) & (dif > 1e-5 * Om.abs().mean())).sum())
                    ulp = (far, float(dif.max() / Om.abs().mean()))
                    err = float((D - De).norm() / De.norm())
                    opt = float(((pred.double() - Om.double()) - De).norm() / De.norm())  # the floor: bf16(W + D3) itself
                    checks.append((label, comp, li, lam, err, opt)); bits.append((label, same, ulp))
                    continue
                err = float((D - De).norm() / De.norm()); opt = rank_err(De)
                checks.append((label, comp, li, lam, err, opt))
print(f'changed {len(changed)} tensors, unchanged {unchanged}')
for c in changed: print('  edited:', c)
worst = 0
for label, comp, li, lam, err, opt in checks:
    if err is None: print(f'  ?? {label}: no kernel weight / not [hidden, *] (comp {comp}, layer {li})'); continue
    worst = max(worst, err - opt)
    if len(checks) <= 40 or err - opt > 1e-3: print(f'  {label}: {comp} layer {li} λ={lam:.4f}  |D-Dexact|/|Dexact| = {err:.2e}  (best rank-3{", rounded" if EXP_BF16 else ""}: {opt:.2e})')
for b in bits: print('   bits', b[0], 'equal %.6f' % b[1], 'far', b[2][0], 'max %.2e' % b[2][1])
if EXP_BF16:
    print(f'bf16 export: worst share of elements equal to bf16(W + D3): {min(b[1] for b in bits) if bits else 1:.6f}; elements more than one bf16 step (and 1e-5 of mean |W|) apart: {sum(b[2][0] for b in bits) if bits else 0}; largest difference {max(b[2][1] for b in bits) if bits else 0:.1e} of mean |W|; over {len(bits)} matrices; (E-W) against the exact edit: worst excess over the error of bf16(W + D3) itself {worst:.2e}; scope {scope}')
else:
    print(f'worst excess over the rank-3 optimum: {worst:.2e} over {len(checks)} matrices; scope {scope}')
