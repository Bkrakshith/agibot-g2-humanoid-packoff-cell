# agibot-g2-humanoid-packoff-cell project page

This folder is the static project page of agibot-g2-humanoid-packoff-cell, in the style of academic project pages (Nerfies).
It is not published yet.

## How it is made

All words live on the page. The file `index.template.html` holds the section text. The script `build.py` replaces
every `{{clip:NAME}}` marker with the video, its poster and its caption from `copy.json`. Each entry in `copy.json`
names its source clip in `output/` and the time of its poster frame. The script also copies the still images
listed in `IMAGES` into `static/images/`. The figure `static/images/architecture.svg` is drawn by hand.

## Build and preview

```bash
FFMPEG=/path/to/ffmpeg python3 website/build.py   # writes website/index.html, reports missing sources
cd website && python3 -m http.server 8000          # then open http://localhost:8000
```

ffmpeg makes the posters and ffprobe (next to it) reads the clip sizes. The videos are about 13 MB.

## Publish

The page is served by GitHub Pages from the `gh-pages` branch of this repository, at
https://bkrakshith.github.io/agibot-g2-humanoid-packoff-cell/. To update it, run `build.py`, then copy `index.html`
and `static/` into a checkout of `gh-pages`, commit and push. On `main`, the built page and the media are ignored.
The page loads Bulma, Font Awesome and Noto Sans from public CDNs.

## Licences shown on the page

Genie Sim is MPL 2.0. GenieSimAssets (the scene assets and the G2 model files) are under the AgiBot World community
licence, CC BY-NC-SA 4.0 (non-commercial use). Keep client names out of the page.
