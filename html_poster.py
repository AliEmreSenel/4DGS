#!/usr/bin/env python3
from pathlib import Path
from urllib.parse import quote
from html import escape
import argparse, csv, os, re


def vid(scene, shape, usplat, color, sort, steps):
    return f"writeup/img/{scene}__{shape}__{usplat}__{color}__{sort}__no_pruning__no_dropout__no_ess__{steps}.mp4"


VIDEOS = [
    ("Bouncing Balls ⚾", "Fixed: No USplat/Prune/ESS/Dropout, Sort, 10k.", ["Sort", "Sort-Free"], [
        ("Ellipsoid", [vid("bouncingballs", "anisotropic", "no_usplat", "sh3", "sort", 10000), vid("bouncingballs", "anisotropic", "use_usplat", "sh3", "sort_free", 10000)]),
        ("Spherical", [vid("bouncingballs", "isotropic", "no_usplat", "sh3", "sort", 10000), vid("bouncingballs", "isotropic", "use_usplat", "sh3", "sort_free", 10000)]),
    ]),
    ("TRex 🐉", "Fixed: No USplat/Prune/ESS/Dropout, Sort, 20k.", ["SH(3)", "RGB"], [
        ("Ellipsoid", [vid("trex", "anisotropic", "no_usplat", "sh3", "sort", 20000), vid("trex", "anisotropic", "no_usplat", "rgb", "sort", 20000)]),
        ("Spherical", [vid("trex", "isotropic", "no_usplat", "sh3", "sort", 20000), vid("trex", "isotropic", "no_usplat", "rgb", "sort", 20000)]),
    ]),
]

MOG = [
    ("bounded_novel_dropout", "output/MOG_Dyna_full_drop"),
    ("bounded_novel_sortfree", "output/MOG_Dyna_full_sort-free"),
    ("bounded_novel_sort", "output/MOG_Dyna_full"),
]

CSS = r"""
:root{color-scheme:light dark;--bg:#fff;--fg:#111;--muted:#555;--panel:#f7f7f7;--card:#fff;--border:#999;--thin-border:#ddd;--th:#eee;--button:#fff;--button-fg:#111}:root.dark{--bg:#101114;--fg:#eee;--muted:#aaa;--panel:#181a20;--card:#111318;--border:#444854;--thin-border:#30343d;--th:#222631;--button:#1b1e26;--button-fg:#eee}@media(prefers-color-scheme:dark){:root:not(.light){--bg:#101114;--fg:#eee;--muted:#aaa;--panel:#181a20;--card:#111318;--border:#444854;--thin-border:#30343d;--th:#222631;--button:#1b1e26;--button-fg:#eee}}body{margin:0;padding:18px;min-height:100vh;box-sizing:border-box;background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif}.layout-both{height:100vh;overflow:hidden;display:grid;grid-template-rows:auto minmax(0,1fr) minmax(0,1fr);gap:18px}h1{margin:0 0 18px;padding-right:460px}.layout-both h1{margin:0}.controls{position:fixed;top:12px;right:12px;z-index:10;display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}button{padding:6px 10px;border:1px solid var(--border);border-radius:7px;background:var(--button);color:var(--button-fg);cursor:pointer}.grid,.mog-grid{display:grid;gap:18px}.grid{grid-template-columns:repeat(2,minmax(0,1fr));align-items:stretch}.mog-grid{grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;align-items:center}section{padding:14px;border:2px solid var(--border);border-radius:10px;background:var(--panel)}section.wide{margin-top:18px}.layout-both section.wide{margin-top:0}.layout-both #ablationsPanel,.layout-both #mogPanel,.layout-both section,.layout-both table{min-height:0}.layout-both #ablationsPanel,.layout-both #mogPanel{overflow:hidden}.layout-both #ablationsPanel>section,.layout-both #mogPanel,.layout-ablations #ablationsPanel>section,.layout-mog #mogPanel{display:flex;flex-direction:column}.layout-both table,.layout-ablations table,.layout-both .mog-grid,.layout-mog .mog-grid{flex:1}.layout-ablations #mogPanel,.layout-mog #ablationsPanel{display:none}.layout-ablations #ablationsPanel,.layout-mog #mogPanel{min-height:calc(100vh - 78px)}h2{margin:0 0 4px}h3{margin:0 0 8px;font-size:15px}p{margin:0 0 12px;color:var(--muted)}table{width:100%;border-collapse:collapse;background:var(--card)}th,td{border:1px solid var(--thin-border);padding:6px;vertical-align:top}th{background:var(--th);font-size:13px}.row{width:95px;text-align:left}.vid{position:relative;margin:0;border:1px solid var(--thin-border);border-radius:7px;overflow:hidden;background:#000}video{display:block;width:100%;height:auto;background:#000}.grid video{aspect-ratio:16/10;object-fit:contain}.layout-both .grid video{height:clamp(70px,15vh,155px);aspect-ratio:auto}.layout-ablations .grid video{height:clamp(150px,28vh,360px);aspect-ratio:auto}.overlay{position:absolute;right:5px;bottom:5px;display:flex;flex-direction:column;align-items:flex-end;gap:2px;pointer-events:none}.overlay span{background:#0009;color:white;font-size:10px;padding:1px 4px;border-radius:3px;white-space:nowrap}.missing-note{position:absolute;inset:0;display:grid;place-items:center;background:#000c;color:#fff;font-size:13px}.mog-card{display:flex;flex-direction:column;min-height:0;padding:10px;border:1px solid var(--thin-border);border-radius:9px;background:var(--card)}.mog-card video{aspect-ratio:auto;height:auto;max-height:72vh;object-fit:contain}.layout-both .mog-grid{min-height:0;align-items:center}.layout-both .mog-card video{max-height:calc((100vh - 245px)/2);width:100%;height:auto}.layout-mog .mog-grid{align-items:center}.layout-mog .mog-card{justify-content:center}.layout-mog .mog-card video{max-height:calc(100vh - 190px)}@media(max-width:1200px){.layout-both{height:auto;overflow:auto;display:block}h1{padding-right:0;margin-top:44px}.controls{left:12px;right:12px;justify-content:flex-start}.grid,.mog-grid{grid-template-columns:1fr}.layout-both .grid video,.layout-ablations .grid video{height:auto}.layout-both .mog-card video,.layout-mog .mog-card video{max-height:none}}
"""

JS = r"""
let pauseAll=false,visible=new WeakMap();const $=q=>document.querySelector(q),$$=q=>document.querySelectorAll(q);function play(v){v.currentTime=0;v.play().catch(()=>{})}function updateVideos(){$$("video").forEach(v=>pauseAll||!visible.get(v)?v.pause():v.paused&&play(v))}const io=new IntersectionObserver(es=>es.forEach(e=>{let v=e.target,on=e.isIntersecting&&e.intersectionRatio>.25,was=visible.get(v);visible.set(v,on);if(!on)v.pause();else if(!pauseAll&&!was)play(v)}),{threshold:[0,.25,.5,1]});$$("video").forEach(v=>{v.loop=v.muted=true;v.addEventListener("ended",()=>play(v));io.observe(v)});const themeButton=$("#themeToggle"),layoutButton=$("#layoutToggle"),media=matchMedia("(prefers-color-scheme: dark)");function currentTheme(){return localStorage.theme||"system"}function applyTheme(m){document.documentElement.classList.remove("light","dark");if(m!="system")document.documentElement.classList.add(m);themeButton.textContent="Theme: "+(m[0].toUpperCase()+m.slice(1))}function cycleTheme(){let m=currentTheme(),n=m=="system"?"light":m=="light"?"dark":"system";localStorage.theme=n;applyTheme(n)}media.addEventListener("change",()=>currentTheme()=="system"&&applyTheme("system"));function currentLayout(){return localStorage.layout||"both"}function applyLayout(m){document.body.className="layout-"+m;layoutButton.textContent="View: "+(m=="mog"?"MOG":m[0].toUpperCase()+m.slice(1));localStorage.layout=m;requestAnimationFrame(updateVideos)}function cycleLayout(){let m=currentLayout();applyLayout(m=="both"?"ablations":m=="ablations"?"mog":"both")}applyTheme(currentTheme());applyLayout(currentLayout());
"""


def parse_filename(path):
    xs = Path(path).stem.split("__")
    try:
        return dict(zip("scene gaussian usplat color sorting pruning dropout ess".split(), xs[:8]), steps=int(xs[8])) if len(xs) >= 9 else None
    except ValueError:
        return None


def load_metrics(root):
    out = {}
    for f in root.rglob("checkpoint_eval_metrics.csv"):
        try:
            with f.open(newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("status") != "ok":
                        continue
                    try:
                        key = tuple(r.get(k) for k in ("scene_name", "isotropy", "usplat", "appearance", "sorting", "pruning", "dropout", "ess")) + (int(float(r["eval_checkpoint_iteration"])),)
                        out[key] = {"fps": round(float(r["render_fps"])), "psnr": round(float(r["psnr"]), 2), "size_mb": round(int(r["checkpoint_size_bytes"]) / 1_000_000, 1)}
                    except (KeyError, ValueError):
                        pass
        except Exception:
            pass
    return out


def get_metrics(path, metrics):
    v = parse_filename(path)
    if not v:
        return None
    cfg = tuple(v[k] for k in "scene gaussian usplat color sorting pruning dropout ess".split()) + (v["steps"],)
    return next((m for k, m in metrics.items() if all(a is None or a == b for a, b in zip(k, cfg))), None)


def first_metric(text, names):
    num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    for n in names:
        m = re.search(rf"\b{re.escape(n)}\b\s*[:=]\s*({num})", text, re.I)
        if m:
            return float(m.group(1))
    for line in text.splitlines():
        if any(n.lower() in line.lower() for n in names):
            m = re.search(num, line)
            if m:
                return float(m.group())


def load_mog_metrics(root, run_dir):
    run, out = root / run_dir, {}
    info = run / "renders/bounded_novel/render_info.txt"
    if info.exists():
        text = info.read_text(encoding="utf-8", errors="replace")
        for key, names, rnd in (
            ("psnr", ["psnr", "render_psnr", "mean_psnr"], 2),
            ("fps", ["fps", "render_fps", "mean_fps", "avg_fps"], 0),
            ("size_mb", ["size_mb", "checkpoint_size_mb", "checkpoint_mb"], 1),
        ):
            val = first_metric(text, names)
            if val is not None:
                out[key] = round(val) if key == "fps" else round(val, rnd)
    if "size_mb" not in out:
        for name in "chkpnt_best.pth chkpnt30000.pth chkpnt25000.pth chkpnt15000.pth chkpnt12500.pth chkpnt10000.pth chkpnt5000.pth".split():
            ckpt = run / name
            if ckpt.exists():
                out["size_mb"] = round(ckpt.stat().st_size / 1_000_000, 1)
                break
    return out or None


def src(path, outdir):
    return quote(os.path.relpath(path, outdir).replace(os.sep, "/"))


def overlay(m):
    return "" if not m else '<div class="overlay">' + "".join(f"<span>{m[k]} {u}</span>" for k, u in (("psnr", "dB"), ("size_mb", "MB"), ("fps", "fps")) if k in m) + "</div>"


def video_cell(path, root, outdir, metrics=None, mog_metrics=None, cls=""):
    full = root / path
    m = mog_metrics if mog_metrics is not None else get_metrics(path, metrics or {})
    return f'<figure class="vid{(" " + escape(cls)) if cls else ""}"><video muted loop playsinline preload="metadata"><source src="{src(full, outdir)}" type="video/mp4"></video>{overlay(m)}{("" if full.exists() else "<div class=\"missing-note\">missing file</div>")}</figure>'


def ablation_block(data, root, outdir, metrics):
    title, subtitle, cols, rows = data
    head = "".join(f"<th>{escape(c)}</th>" for c in cols)
    body = "".join(f"<tr><th class='row'>{escape(name)}</th>{''.join(f'<td>{video_cell(p, root, outdir, metrics)}</td>' for p in paths)}</tr>" for name, paths in rows)
    return f'<section class="ablation-card"><h2>{escape(title)}</h2><p>{escape(subtitle)}</p><table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></section>'


def mog_section(root, outdir):
    cards = "".join(
        f'<div class="mog-card"><h3>{escape(label)}</h3>{video_cell(f"{run}/renders/bounded_novel/bounded_novel.mp4", root, outdir, mog_metrics=load_mog_metrics(root, run), cls="mog-video")}</div>'
        for label, run in MOG
    )
    return f'<section id="mogPanel" class="wide"><h2>MOG Dyna / bounded novel</h2><p>Metrics are read from each run’s renders/bounded_novel/render_info.txt. Checkpoint size comes from chkpnt_best.pth when not listed in the log.</p><div class="mog-grid">{cards}</div></section>'


def page(root, out, metrics):
    outdir = out.parent
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Ablation Videos</title><style>{CSS}</style></head><body class="layout-both"><h1>Ablation Videos</h1><div class="controls"><button onclick="pauseAll=false;updateVideos()">Resume visible</button><button onclick="pauseAll=true;document.querySelectorAll('video').forEach(v=>v.pause())">Pause all</button><button id="layoutToggle" onclick="cycleLayout()">View: Both</button><button id="themeToggle" onclick="cycleTheme()">Theme: System</button></div><div id="ablationsPanel" class="grid">{''.join(ablation_block(v, root, outdir, metrics) for v in VIDEOS)}</div>{mog_section(root, outdir)}<script>{JS}</script></body></html>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=".")
    ap.add_argument("-o", "--output", default="ablations.html")
    args = ap.parse_args()
    root, out = Path(args.dir).resolve(), Path(args.output).resolve()
    metrics = load_metrics(root)
    out.write_text(page(root, out, metrics), encoding="utf-8")
    print(f"Loaded metrics for {len(metrics)} checkpoints")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
