"""Build the humanoid-packoff-cell project page: index.template.html + copy.json -> index.html.

    python3 website/build.py

Every {{clip:NAME}} marker becomes a <figure> with the video, its poster, and the caption and notes from
website/copy.json (the public text, ASD-STE100). The clips are copied from output/ into website/static/videos/,
and a poster frame is cut from each with ffmpeg (FFMPEG env var, else ffmpeg on PATH). The still images listed
in IMAGES are copied into website/static/images/. Missing sources are reported and left out.
"""

import html
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VIDEOS = os.path.join(HERE, "static", "videos")
IMAGES_DIR = os.path.join(HERE, "static", "images")
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
FFPROBE = os.path.join(os.path.dirname(FFMPEG), "ffprobe") if FFMPEG else None

# published name -> source, relative to the project root
IMAGES = {
    "cell_table.png": "output/readiness/relay/03_table_view_free_camera_rgb.png",
    "relay_dropped.png": "output/readiness/relay/04_bag03_dropped_to_floor_free_camera_rgb.png",
    "pilot_events.png": "output/readiness/pilot/events_ep_s00007.png",
    "pilot_loss.png": "output/readiness/pilot/loss_pilot_v1_cpu.png",
}


def video_size(src):
    """(width, height) of the first video stream, so each clip keeps its own aspect ratio on the page."""
    if not FFPROBE or not os.path.exists(FFPROBE):
        return None
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", src], capture_output=True, text=True, check=True).stdout
    w, h = out.strip().split(",")
    return int(w), int(h)


def figure(name, m, size=None):
    def e(text):
        return html.escape(text).replace(" -&gt; ", " &rarr; ")
    notes = "".join(f"<li>{e(n)}</li>" for n in m.get("notes", []))
    details = f"<details><summary>Details</summary><ul>{notes}</ul></details>" if notes else ""
    desc = f" {e(m['description'])}" if m.get("description") else ""
    dims = f' width="{size[0]}" height="{size[1]}"' if size else ""
    return (f'<figure class="clip">'
            f'<video{dims} data-autoplay muted loop playsinline preload="none" poster="static/videos/{name}.jpg">'
            f'<source src="static/videos/{name}.mp4" type="video/mp4"></video>'
            f'<figcaption><span class="clip-title">{e(m["title"])}.</span> '
            f'<span class="clip-sub">{e(m.get("subtitle") or "")}.</span>{desc}{details}'
            f'</figcaption></figure>')


def poster(src, t, dst):
    if not FFMPEG:
        print("warning: no ffmpeg; set FFMPEG=/path/to/ffmpeg to make posters")
        return
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-ss", str(t), "-i", src, "-frames:v", "1", "-q:v", "3", dst],
                   check=True)


def main():
    copy = json.load(open(os.path.join(HERE, "copy.json")))
    os.makedirs(VIDEOS, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    missing = []
    for out, rel in IMAGES.items():
        src = os.path.join(ROOT, rel)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(IMAGES_DIR, out))
        else:
            missing.append(rel)
    page = open(os.path.join(HERE, "index.template.html")).read()

    def expand(mt):
        name = mt.group(1)
        m = copy.get(name)
        src = os.path.join(ROOT, m["src"]) if m else ""
        if not m or not os.path.exists(src):
            missing.append(name)
            return f"<!-- clip {name} not built yet -->"
        shutil.copy2(src, os.path.join(VIDEOS, f"{name}.mp4"))
        poster(src, m.get("poster_t", 1.0), os.path.join(VIDEOS, f"{name}.jpg"))
        return figure(name, m, video_size(src))

    page = re.sub(r"\{\{clip:([\w-]+)\}\}", expand, page)
    open(os.path.join(HERE, "index.html"), "w").write(page)
    size = sum(os.path.getsize(os.path.join(VIDEOS, f)) for f in os.listdir(VIDEOS)) / 1e6
    print(f"website/index.html written; static/videos = {size:.1f} MB")
    if missing:
        print("missing:", ", ".join(missing))
        sys.exit(1)


if __name__ == "__main__":
    main()
