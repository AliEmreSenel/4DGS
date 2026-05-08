#!/usr/bin/env python3
from pathlib import Path
from html import escape
from urllib.parse import quote
import argparse, os

STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
EXTS = {".mp4", ".webm", ".mov", ".m4v"}

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
    return dict(zip(keys, xs[:8]), steps=steps, path=p, filename=p.name)


def src(p, outdir):
    return quote(os.path.relpath(p, outdir).replace(os.sep, "/"))


def vid(v, outdir):
    if not v:
        return '<div class="missing">missing</div>'
    return f"""
<figure class="vid">
  <video muted loop playsinline preload="metadata">
    <source src="{src(v["path"], outdir)}">
  </video>
</figure>"""


def trex_section(vs, outdir):
    vs = [v for v in vs if v["scene"] == "trex" and v["usplat"] == "no_usplat" and v["steps"] == 10000]
    rows = sorted({(v["gaussian"], v["color"], v["ess"]) for v in vs})
    cols = sorted({(v["sorting"], v["pruning"], v["dropout"]) for v in vs})
    lookup = {(v["gaussian"], v["color"], v["ess"], v["sorting"], v["pruning"], v["dropout"]): v for v in vs}

    head = "".join(f"<th>{'<br>'.join(map(lab, c))}</th>" for c in cols)
    body = ""
    for r in rows:
        cells = "".join(f"<td>{vid(lookup.get((*r, *c)), outdir)}</td>" for c in cols)
        body += f"<tr><th class='row'>{' / '.join(map(lab, r))}</th>{cells}</tr>"

    return f"""
<section>
  <h2>trex / no_usplat / 10,000 steps</h2>
  <p>Rows: gaussian x color x ESS. Columns: sorting x pruning x dropout.</p>
  <div class="wrap">
    <table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>
  </div>
</section>"""


def sweep_section(vs, outdir):
    vs = [v for v in vs if v["scene"] == "bouncingballs" and v["steps"] in STEPS and v["usplat"] in {"use_usplat", "no_usplat"}]

    def base(v):
        return tuple(v[k] for k in ["gaussian", "color", "sorting", "pruning", "dropout", "ess"])

    bases = sorted({base(v) for v in vs})
    lookup = {(base(v), v["usplat"], v["steps"]): v for v in vs}

    head = "".join(f"<th>{s // 1000}k</th>" for s in STEPS)
    body = ""

    for i, b in enumerate(bases):
        group = f"g{i % 2}"
        for u in ["use_usplat", "no_usplat"]:
            label = f"{lab(u)} / {' / '.join(map(lab, b))}"
            cells = "".join(f"<td>{vid(lookup.get((b, u, s)), outdir)}</td>" for s in STEPS)
            body += f"<tr class='{group}'><th class='row'>{escape(label)}</th>{cells}</tr>"

    return f"""
<section>
  <h2>bouncing balls / usplat vs no_usplat / 1,000-7,000 steps</h2>
  <p>Rows: ablation variant. Columns: training iteration. usplat/no_usplat are stacked.</p>
  <div class="wrap">
    <table class="sweep"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>
  </div>
</section>"""


def page(vs, out):
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
.vid {{ width:205px; margin:0; border:1px solid #ddd; border-radius:7px; overflow:hidden; background:white; }}
video {{ width:205px; height:130px; display:block; background:#000; object-fit:contain; }}
.missing {{ width:205px; height:130px; display:grid; place-items:center; border:1px dashed #999; border-radius:7px; color:#777; }}
.sweep .row {{ min-width:190px; max-width:220px; }}
.sweep .vid, .sweep video, .sweep .missing {{ width:190px; }}
.sweep video, .sweep .missing {{ height:120px; }}
.sweep tr.g0 td, .sweep tr.g0 .row {{ background:#fff7e8; }}
.sweep tr.g1 td, .sweep tr.g1 .row {{ background:#eaf7ff; }}
</style>
</head>
<body>
<h1>Ablation Videos</h1>

<button onclick="pauseAll=false; updateVideos()">Resume visible</button>
<button onclick="pauseAll=true; document.querySelectorAll('video').forEach(v=>v.pause())">Pause all</button>

{trex_section(vs, outdir)}
{sweep_section(vs, outdir)}

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

    out.write_text(page(vs, out), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Included {len(vs)} videos")


if __name__ == "__main__":
    main()