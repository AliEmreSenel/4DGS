#!/usr/bin/env python3
from pathlib import Path
from html import escape
from urllib.parse import quote
import argparse, csv, os

STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
EXTS = VIDEO_EXTS | IMAGE_EXTS

LABEL = {
    "bouncingballs": "bouncing balls", "trex": "trex",
    "use_usplat": "usplat", "no_usplat": "no_usplat",
    "anisotropic": "anisotropic", "isotropic": "isotropic",
    "rgb": "RGB", "sh3": "SH3",
    "sort": "sort", "sort_free": "sort-free",
    "interleaved_prune_densify": "prune+densify", "no_pruning": "no pruning",
    "dropout": "dropout", "yes_dropout": "dropout", "no_dropout": "no dropout",
    "ess": "ESS", "no_ess": "no ESS",
}


def lab(x): return LABEL.get(x, x)


def parse(p):
    xs = p.stem.split("__")
    if len(xs) < 9: return None
    try: steps = int(xs[8])
    except ValueError: return None
    keys = ["scene", "gaussian", "usplat", "color", "sorting", "pruning", "dropout", "ess"]
    return dict(zip(keys, xs[:8]), steps=steps, path=p, filename=p.name,
                is_image=p.suffix.lower() in IMAGE_EXTS)


def load_metrics(root):
    """Return dict keyed by (scene, gaussian, usplat, color, sorting, pruning, dropout, ess, steps)."""
    metrics = {}
    for f in root.rglob("checkpoint_eval_metrics.csv"):
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("status") != "ok":
                        continue
                    try:
                        steps = int(float(row["eval_checkpoint_iteration"]))
                        key = (
                            row["scene_name"],
                            row["isotropy"],           # gaussian
                            row.get("usplat"),         # None if column absent
                            row["appearance"],         # color
                            row["sorting"],
                            row["pruning"],
                            row.get("dropout"),        # None if column absent
                            row.get("ess"),            # None if column absent
                            steps,
                        )
                        metrics[key] = {
                            "fps":     round(float(row["render_fps"])),
                            "psnr":    round(float(row["psnr"]), 2),
                            "size_mb": round(int(row["checkpoint_size_bytes"]) / 1_000_000, 1),
                        }
                    except (ValueError, KeyError):
                        pass
        except Exception:
            pass
    return metrics


def get_metrics(v, metrics):
    """Return metrics for the last (highest iteration) checkpoint of this config.

    Slots that are None in the metrics key (column absent from that CSV) are
    treated as wildcards so trex and bouncing-balls CSVs can both match.
    """
    config = (v["scene"], v["gaussian"], v["usplat"], v["color"],
              v["sorting"], v["pruning"], v["dropout"], v["ess"])
    candidates = {
        k[8]: metrics[k] for k in metrics
        if all(mk is None or mk == ck for mk, ck in zip(k[:8], config))
    }
    if not candidates:
        scenes = sorted({k[0] for k in metrics})
        same_scene = sorted({k for k in metrics if k[0] == config[0]})
        detail = (
            f"No metrics found for: {v['filename']}\n"
            f"  Looked up config: {config}\n"
            f"  Scenes in metrics: {scenes}\n"
            + (f"  Configs for scene '{config[0]}':\n" +
               "\n".join(f"    {k}" for k in same_scene)
               if same_scene else f"  No entries at all for scene '{config[0]}'")
        )
        raise ValueError(detail)
    return candidates[max(candidates)]


def src(p, outdir):
    return quote(os.path.relpath(p, outdir).replace(os.sep, "/"))


def vid(v, outdir, metrics):
    if not v:
        return '<div class="missing">missing</div>'
    m = get_metrics(v, metrics)
    overlay = ""
    if m:
        overlay = (
            f'<div class="overlay">'
            f'<span>{m["psnr"]} dB</span>'
            f'<span>{m["size_mb"]} MB</span>'
            f'<span>{m["fps"]} fps</span>'
            f'</div>'
        )
    url = src(v["path"], outdir)
    if v["is_image"]:
        media = f'<img src="{url}" loading="lazy">'
    else:
        media = f'<video muted loop playsinline preload="metadata"><source src="{url}"></video>'
    return f'<figure class="vid">{media}{overlay}</figure>'


def trex_section(vs, outdir, metrics):
    vs = [v for v in vs if v["scene"] == "trex" and v["usplat"] == "no_usplat" and v["steps"] == 20000]
    rows = sorted({(v["gaussian"], v["color"], v["ess"]) for v in vs})
    cols = sorted({(v["sorting"], v["pruning"], v["dropout"]) for v in vs})
    # prefer video over image when both exist for the same config
    lookup = {}
    for v in vs:
        k = (v["gaussian"], v["color"], v["ess"], v["sorting"], v["pruning"], v["dropout"])
        if k not in lookup or lookup[k]["is_image"]:
            lookup[k] = v

    head = "".join(f"<th>{'<br>'.join(map(lab, c))}</th>" for c in cols)
    body = ""
    for r in rows:
        cells = "".join(f"<td>{vid(lookup.get((*r, *c)), outdir, metrics)}</td>" for c in cols)
        body += f"<tr><th class='row'>{' / '.join(map(lab, r))}</th>{cells}</tr>"

    return f"""
<section>
  <h2>trex / no_usplat / 10,000 steps</h2>
  <p>Rows: gaussian x color x ESS. Columns: sorting x pruning x dropout.</p>
  <div class="wrap">
    <table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>
  </div>
</section>"""


def sweep_section(vs, outdir, metrics):
    vs = [v for v in vs if v["scene"] == "bouncingballs" and v["steps"] in STEPS and v["usplat"] in {"use_usplat", "no_usplat"}]

    def base(v):
        return tuple(v[k] for k in ["gaussian", "color", "sorting", "pruning", "dropout", "ess"])

    bases = sorted({base(v) for v in vs})
    # prefer video over image when both exist
    lookup = {}
    for v in vs:
        k = (base(v), v["usplat"], v["steps"])
        if k not in lookup or lookup[k]["is_image"]:
            lookup[k] = v

    head = "".join(f"<th>{s // 1000}k</th>" for s in STEPS)
    body = ""

    for i, b in enumerate(bases):
        group = f"g{i % 2}"
        for u in ["use_usplat", "no_usplat"]:
            label = f"{lab(u)} / {' / '.join(map(lab, b))}"
            cells = "".join(f"<td>{vid(lookup.get((b, u, s)), outdir, metrics)}</td>" for s in STEPS)
            body += f"<tr class='{group}'><th class='row'>{escape(label)}</th>{cells}</tr>"

    return f"""
<section>
  <h2>bouncing balls / usplat vs no_usplat / 1,000-7,000 steps</h2>
  <p>Rows: ablation variant. Columns: training iteration. usplat/no_usplat are stacked.</p>
  <div class="wrap">
    <table class="sweep"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>
  </div>
</section>"""


def page(vs, out, metrics):
    outdir = out.parent
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Ablation Videos</title>
<style>
body {{ margin:0; padding:18px; background:white; color:#111; font-family:system-ui,sans-serif; }}
h1 {{ margin:0 0 18px; }}
section {{ margin:0 0 36px; padding:14px; border:2px solid #999; border-radius:10px; background:#f7f7f7; }}
h2 {{ margin:0 0 4px; }}
p {{ margin:0 0 12px; color:#555; }}
button {{ padding:6px 10px; margin:0 6px 16px 0; border:1px solid #888; border-radius:7px; background:white; cursor:pointer; }}
.wrap {{ overflow:auto; background:white; border:1px solid #ccc; border-radius:8px; }}
table {{ border-collapse:collapse; width:max-content; min-width:1050px; }}
.sweep {{ min-width:max-content; }}
th, td {{ border:1px solid #ddd; padding:4px; vertical-align:top; }}
th {{ background:#eee; font-size:12px; }}
.row {{ position:sticky; left:0; z-index:1; min-width:135px; max-width:160px; text-align:left; }}
.vid {{ position:relative; width:205px; margin:0; border:1px solid #ddd; border-radius:7px; overflow:hidden; background:white; }}
video {{ width:205px; height:130px; display:block; background:#000; object-fit:contain; }}
img {{ width:205px; height:130px; display:block; object-fit:contain; background:#000; }}
.missing {{ width:205px; height:130px; display:grid; place-items:center; border:1px dashed #999; border-radius:7px; color:#777; }}
.overlay {{ position:absolute; bottom:4px; right:4px; display:flex; flex-direction:column; align-items:flex-end; gap:2px; pointer-events:none; }}
.overlay span {{ background:rgba(0,0,0,0.55); color:#fff; font-size:10px; padding:1px 4px; border-radius:3px; white-space:nowrap; }}
.sweep .row {{ min-width:190px; max-width:220px; }}
.sweep .vid, .sweep video, .sweep img, .sweep .missing {{ width:190px; }}
.sweep video, .sweep img, .sweep .missing {{ height:120px; }}
.sweep tr.g0 td, .sweep tr.g0 .row {{ background:#fff7e8; }}
.sweep tr.g1 td, .sweep tr.g1 .row {{ background:#eaf7ff; }}
</style>
</head>
<body>
<h1>Ablation Videos</h1>

<button onclick="pauseAll=false; updateVideos()">Resume visible</button>
<button onclick="pauseAll=true; document.querySelectorAll('video').forEach(v=>v.pause())">Pause all</button>

{trex_section(vs, outdir, metrics)}
{sweep_section(vs, outdir, metrics)}

<script>
let pauseAll = false;
const visible = new WeakMap();

function playFromStart(v) {{
  v.currentTime = 0;
  v.play().catch(() => {{}});
}}

function updateVideos() {{
  document.querySelectorAll("video").forEach(v => {{
    if (pauseAll || !visible.get(v)) v.pause();
    else if (v.paused) playFromStart(v);
  }});
}}

const io = new IntersectionObserver(entries => {{
  entries.forEach(e => {{
    const v = e.target;
    const isVisible = e.isIntersecting && e.intersectionRatio > 0.25;
    const wasVisible = visible.get(v);
    visible.set(v, isVisible);

    if (!isVisible) v.pause();
    else if (!pauseAll && !wasVisible) playFromStart(v);
  }});
}}, {{ threshold: [0, 0.25, 0.5, 1] }});

document.querySelectorAll("video").forEach(v => {{
  v.loop = true;
  v.muted = true;
  v.addEventListener("ended", () => playFromStart(v));
  io.observe(v);
}});
</script>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=".")
    ap.add_argument("-o", "--output", default="ablations.html")
    ap.add_argument("--no-recursive", action="store_true")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    out = Path(args.output).resolve()
    files = root.glob("*") if args.no_recursive else root.rglob("*")
    vs = [v for p in files if p.is_file() and p.suffix.lower() in EXTS for v in [parse(p)] if v]

    metrics = load_metrics(root)
    print(f"Loaded metrics for {len(metrics)} checkpoints")

    out.write_text(page(vs, out, metrics), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Included {len(vs)} files ({sum(1 for v in vs if not v['is_image'])} videos, {sum(1 for v in vs if v['is_image'])} images)")


if __name__ == "__main__":
    main()