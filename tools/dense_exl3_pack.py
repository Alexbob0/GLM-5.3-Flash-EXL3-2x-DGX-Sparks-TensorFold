#!/usr/bin/env python3
"""Build GLM-5.3-Flash's dense EXL3 pack for TF_GLM_DENSE_EXL3 from turboderp/GLM-5.3-Flash-exl3, by HTTP range reads
(only the non-expert groups are fetched: ~3.5 GB for the default mix, not the whole quant).

Default mix ("k4mix", measured on 2x GB10): attention (KDA q/k/v/o, DSA q_a/q_b/kv_a/o) and the shared experts at 4 bpw
from the 2.05bpw branch, the three dense MLPs at 5 bpw and the head at 6 bpw from the 4.05bpw branch. The Hub quant
stores KDA's q/k/v fused as qkv_proj; they are cut back into q_proj / k_proj / v_proj (equal thirds of the output tiles
and svh, one shared suh), the names the engine reads.

    python3 tools/dense_exl3_pack.py --out ~/.cache/huggingface/dense-exl3/glm53-k4mix.safetensors
    python3 tools/dense_exl3_pack.py --attn 4.05bpw --mlp 4.05bpw --head 4.05bpw --out .../glm53-k6.safetensors

Range reads and the fused split follow vcruz305/vllm-exl3's tools/dense_overlay.py (Apache-2.0)."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request

REPO = "https://huggingface.co/{repo}/resolve/{branch}/"
ROOT = "model.language_model."
SIZE = {"I16": 2, "F16": 2, "I32": 4}
ATTN = re.compile(r"^model\.language_model\.layers\.(\d+)\.self_attn\.(o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|qkv_proj)\.")
SHARED = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.shared_experts\.(gate|up|down)_proj\.")
MLP = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.")
PARTS = ("trellis", "suh", "svh", "mul1", "mcg")


def get(url: str, rng: tuple[int, int] | None = None, tries: int = 6) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dense_exl3_pack/1"})
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=120) as r:
                if rng and r.status != 206:
                    raise RuntimeError(f"expected 206, got {r.status}")
                data = r.read()
            if rng and len(data) != rng[1] - rng[0] + 1:
                raise RuntimeError(f"short read {len(data)} of {rng[1] - rng[0] + 1}")
            return data
        except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as e:
            if attempt == tries - 1:
                raise
            print(f"  retry {attempt + 1}: {e}", flush=True)
            time.sleep(2 ** attempt)


class Branch:
    """One branch of the Hub quant: its index and the headers of the files we read, fetched once."""

    def __init__(self, repo: str, branch: str) -> None:
        self.base = REPO.format(repo=repo, branch=branch)
        self.map = json.loads(get(self.base + "model.safetensors.index.json"))["weight_map"]
        self.headers: dict[str, tuple[int, dict]] = {}

    def meta(self, name: str):
        fn = self.map.get(name)
        if fn is None:
            return None
        if fn not in self.headers:
            hlen = struct.unpack("<Q", get(self.base + fn, (0, 7)))[0]
            self.headers[fn] = (hlen, json.loads(get(self.base + fn, (8, 8 + hlen - 1))))
        hlen, h = self.headers[fn]
        return fn, hlen, h[name]

    def read(self, name: str) -> tuple[str, list[int], bytes]:
        fn, hlen, m = self.meta(name)
        a, b = m["data_offsets"]
        return m["dtype"], m["shape"], get(self.base + fn, (8 + hlen + a, 8 + hlen + b - 1))


def groups(branch: Branch, pick) -> list[str]:
    """The module names (``<module>`` of ``<module>.trellis``) of ``branch`` that ``pick`` keeps, in order."""
    return sorted({k[:-len(".trellis")] for k in branch.map if k.endswith(".trellis") and pick(k)},
                  key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])


def tensors_of(branch: Branch, module: str):
    """(name, dtype, shape, bytes) of a module's group; a fused qkv_proj comes out as q/k/v_proj thirds."""
    parts = {p: branch.read(f"{module}.{p}") for p in PARTS if branch.meta(f"{module}.{p}") is not None}
    if not module.endswith(".qkv_proj"):
        return [(f"{module}.{p}", *v) for p, v in parts.items()]
    base = module[:-len("qkv_proj")]
    dt, (kt, nt, w), raw = parts["trellis"]
    if nt % 3:
        raise ValueError(f"{module}: {nt} output tiles do not split in thirds")
    row, third = nt * w * 2, nt // 3 * w * 2
    sdt, (n,), svh = parts["svh"]
    out = []
    for j, name in enumerate(("q_proj", "k_proj", "v_proj")):
        tr = b"".join(raw[r * row + j * third:r * row + (j + 1) * third] for r in range(kt))
        out += [(base + name + ".trellis", dt, [kt, nt // 3, w], tr), (base + name + ".suh", *parts["suh"]),
                (base + name + ".svh", sdt, [n // 3], svh[j * n // 3 * 2:(j + 1) * n // 3 * 2])]
        out += [(base + name + "." + p, *parts[p]) for p in ("mul1", "mcg") if p in parts]
    return out


def write(path: str, entries: list, metadata: dict) -> None:
    header, off = {"__metadata__": metadata}, 0
    for name, dtype, shape, data in entries:
        n = SIZE[dtype]
        for d in shape:
            n *= d
        if n != len(data):
            raise ValueError(f"{name}: {len(data)} bytes for {dtype} {shape}")
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for e in entries:
            f.write(e[3])
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="turboderp/GLM-5.3-Flash-exl3")
    ap.add_argument("--attn", default="2.05bpw", help="branch for attention and the shared experts (default K4)")
    ap.add_argument("--mlp", default="4.05bpw", help="branch for the dense MLPs (default K5)")
    ap.add_argument("--head", default="4.05bpw", help="branch for lm_head (default K6); 'none' keeps the BF16 head")
    ap.add_argument("--layers", type=int, default=45, help="decoder layers to take (the MTP layer, 45, is left out)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    cache: dict[str, Branch] = {}

    def br(name: str) -> Branch:
        if name not in cache:
            cache[name] = Branch(a.repo, name)
        return cache[name]

    def layer_ok(k: str, rx) -> bool:
        m = rx.match(k)
        return bool(m) and int(m.group(1)) < a.layers

    plan = [(a.attn, m) for m in groups(br(a.attn), lambda k: layer_ok(k, ATTN) or layer_ok(k, SHARED))]
    plan += [(a.mlp, m) for m in groups(br(a.mlp), lambda k: layer_ok(k, MLP) and not SHARED.match(k))]
    if a.head != "none":
        plan.append((a.head, "lm_head"))
    entries, t0, done = [], time.time(), 0
    for i, (b, module) in enumerate(plan):
        for e in tensors_of(br(b), module):
            entries.append(e)
            done += len(e[3])
        if i % 20 == 0 or i == len(plan) - 1:
            print(f"  {i + 1}/{len(plan)} {module} ({b}) {done / 1e9:.2f} GB {time.time() - t0:.0f}s", flush=True)
    write(a.out, entries, {"format": "pt", "source": f"{a.repo} attn+shared@{a.attn} mlp@{a.mlp} head@{a.head}"})
    print(f"wrote {a.out}: {len(entries)} tensors, {done / 1e9:.2f} GB in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
