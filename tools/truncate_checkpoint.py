"""truncate_checkpoint.py REPO N [OUT_DIR]: build a checkpoint of the first N decoder layers by
range-fetching only the needed tensors from the remote safetensors shards."""
import sys, json, struct, os, re, requests
from huggingface_hub import hf_hub_url, hf_hub_download, list_repo_files
repo, N = sys.argv[1], int(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else f"models/{repo.replace('/','__')}-L{N}"
os.makedirs(out, exist_ok=True)
files = list_repo_files(repo)
for f in files:
    if f.endswith(('.json', '.jinja', '.model', '.txt', '.tiktoken', '.py')) and not f.startswith('.') and '/' not in f:
        if f == 'model.safetensors.index.json': continue
        hf_hub_download(repo, f, local_dir=out)
cfg = json.load(open(f"{out}/config.json"))
tc = cfg.get('text_config', cfg)
tc['num_hidden_layers'] = N
for key in ('layer_types', 'mlp_layer_types', 'num_attention_heads_per_layer'):
    if isinstance(tc.get(key), list): tc[key] = tc[key][:N]
json.dump(cfg, open(f"{out}/config.json", 'w'), indent=1)
if 'model.safetensors.index.json' in files:
    wm = json.load(open(hf_hub_download(repo, 'model.safetensors.index.json', local_dir='/tmp/truncate_idx_' + repo.replace('/', '_'))))['weight_map']
else:
    wm = None
shards = sorted(set(wm.values())) if wm else ['model.safetensors']
s = requests.Session()
def keep(name):
    m = re.search(r'\.layers\.(\d+)\.', name)
    return m is None or int(m.group(1)) < N
tensors = {}
for sh in shards:
    url = hf_hub_url(repo, sh)
    r = s.get(url, headers={'Range': 'bytes=0-7'}); n = struct.unpack('<Q', r.content)[0]
    hdr = json.loads(s.get(url, headers={'Range': f'bytes=8-{8+n-1}'}).content)
    hdr.pop('__metadata__', None)
    for k, v in hdr.items():
        if not keep(k): continue
        a, b = v['data_offsets']
        data = s.get(url, headers={'Range': f'bytes={8+n+a}-{8+n+b-1}'}).content if b > a else b''
        assert len(data) == b - a, (k, len(data), b - a)
        tensors[k] = (v['dtype'], v['shape'], data)
# write one safetensors file
header, off = {}, 0
for k, (dt, shp, data) in tensors.items():
    header[k] = {'dtype': dt, 'shape': shp, 'data_offsets': [off, off + len(data)]}; off += len(data)
hb = json.dumps(header).encode(); hb += b' ' * ((8 - len(hb) % 8) % 8)
with open(f"{out}/model.safetensors", 'wb') as f:
    f.write(struct.pack('<Q', len(hb))); f.write(hb)
    for k, (dt, shp, data) in tensors.items(): f.write(data)
print(out, len(tensors), 'tensors', off / 1e9, 'GB')
