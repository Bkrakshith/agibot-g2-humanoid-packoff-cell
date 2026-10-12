# docs/media: images for the top-level README

The top-level README shows these files. The sources are the published clips and stills of the project page
(`gh-pages` branch, `static/`), which `website/build.py` makes from `output/`.

| File | Source |
|---|---|
| `g2_packing_cycle.gif` | `01_packing_overview.mp4` (cropped) and `02_packing_head.mp4`, side by side, 3.5× speed, 8 fps |
| `packing_storyboard.jpg` | `02_packing_head.mp4` at 12 s, 27 s, 44 s and 56 s, with step labels |
| `cell_plan.svg` | Top view drawn from the numbers in `assets/cell/build.py` (`LAYOUT`). Update it when the layout changes. |
| `object_relay_before_after.jpg` | `cell_table.png` and `relay_dropped.png`, with labels |
| `teleop_replay.jpg` | poster frame of `03_teleop_replay.mp4` |
| `pilot_episode_events.jpg`, `pilot_training_loss.png` | `pilot_events.png`, `pilot_loss.png` |
| `slam_nav.jpg` | poster frame of `04_slam_nav.mp4` |

Make the GIF again (Inter font from the `fonts-inter` package; keep the file below approximately 8 MB):

```bash
F=/usr/share/fonts/opentype/inter/Inter-SemiBold.otf
L="fontfile=$F:fontsize=14:fontcolor=white:box=1:boxcolor=0x1f2733@0.72:boxborderw=6"
ffmpeg -i 01_packing_overview.mp4 -i 02_packing_head.mp4 -filter_complex \
 "[0:v]crop=1120:504:20:150,hqdn3d=3:3:6:6,scale=600:270:flags=lanczos,setpts=PTS/3.5,drawtext=$L:text='CELL VIEW':x=10:y=10[a];\
  [1:v]hqdn3d=3:3:6:6,scale=432:270:flags=lanczos,setpts=PTS/3.5,drawtext=$L:text='G2 HEAD CAMERA':x=10:y=10,\
drawtext=$L:text='Isaac Sim · 3.5× speed':x=w-tw-16:y=h-th-16[b];[a][b]hstack,fps=8,split[s0][s1];\
  [s0]palettegen=max_colors=128:stats_mode=full[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
 docs/media/g2_packing_cycle.gif
```

All files come from simulation. They show the AgiBot G2 model and GenieSimAssets scenes (CC BY-NC-SA 4.0,
non-commercial use). Refer to the licence section of the top-level README.
