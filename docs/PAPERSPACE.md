# Running the G2 cell on Paperspace

## Machine

| | minimum | notes |
|---|---|---|
| GPU | 24 GB VRAM, RTX-class: A5000 or A6000 | A100 has 40/80 GB but no RT cores; NVIDIA does not list it for Isaac Sim RTX rendering, so cameras may fail. Test before relying on it |
| Disk | ~200 GB | the cell image unpacks to ~56 GB with Docker's classic overlay2 store (set by `scripts/server_setup.sh`); the containerd store would need ~2x. After resizing the disk in the console, `server_setup.sh` grows the LVM root |
| OS | Ubuntu 22.04, no driver needed | `scripts/server_setup.sh` installs nvidia-driver-580 (workstation: 580.178.04), Docker CE, the NVIDIA runtime |

Everything runs headless. You need an SSH shell (teleop) and one forwarded port (Foxglove).

## First-time setup (recommended: build off the GPU box)

Full flow, timings and cost plan: [OPTIMISATION.md, "Build & deploy"](OPTIMISATION.md#build--deploy).
In short:

```bash
# CPU builder (cheap Paperspace CPU machine or any Linux box with >= 150-200 GB disk)
HOST=builder scripts/sync_to_server.sh                         # on the workstation
scripts/server_setup.sh --builder                              # on the builder
echo "$GHCR_PAT" | docker login ghcr.io -u <user> --password-stdin
NAMESPACE=<user> scripts/build_and_push.sh                     # private ghcr.io/<user>/g2cell, ~1.5-2.5 h

# GPU box (ssh alias "paperspace")
scripts/sync_to_server.sh                                      # on the workstation: project + geniesim_assets -> ~/g2/
ssh paperspace
cd ~/g2/g2_packoff_cell
scripts/server_setup.sh                                        # disk grow, Docker, toolkit, genie_sim, driver -> asks for a reboot
sudo reboot                                                    # then:
scripts/server_setup.sh                                        # GPU-in-docker self-test
newgrp docker                                                  # only if setup added you to the group
docker login ghcr.io                                           # read-only token
NAMESPACE=<user> scripts/pull_on_server.sh                     # ~15-25 min
sudo scripts/idle_shutdown.sh install                          # power off after 30 idle min
scripts/session.sh collect                                     # or: up | showcase [--mobile] | status | down [--poweroff]
```

Result, side by side under `~/g2`: `g2_packoff_cell/`, `genie_sim/` (cloned at `6ca11c7` by
`server_setup.sh`), `geniesim_assets/` (synced from the workstation, incl. `scenes/g2cell`).
`scripts/sync_to_server.sh --dry-run` lists what would be sent; `--pull-output` fetches `output/` back.
Every script is safe to rerun.

Fallback without a builder/registry: `scripts/server_setup.sh`, then `BAKED=1 docker/build_all.sh` on
the GPU box (~1.5 h of GPU time), then `docker/cell.sh up`.

Set Paperspace's own **Auto-Shutdown** (machine page, Settings) as a backstop, e.g. 8 h.

## Connect

```bash
ssh -L 8765:localhost:8765 paperspace@<machine-ip>
```

Then open Foxglove (app or browser) at `ws://localhost:8765`. Use `tmux` on the server so the sim
keeps running when SSH drops.

## Moving images without the registry scripts

`scripts/build_and_push.sh` / `scripts/pull_on_server.sh` are the supported path (content-hash tags,
private registry, zstd layers). Without a registry, copy the image as a file instead (no credentials;
needs disk for the archive on both ends, plus the loaded image on Paperspace):

```bash
# on a machine that has the image
docker save g2cell/cell:dev | zstd -T0 -3 > g2cell-cell.tar.zst
rsync -avP g2cell-cell.tar.zst paperspace:g2/
# on Paperspace
zstd -dc ~/g2/g2cell-cell.tar.zst | docker load
```

Only `g2cell/cell:dev` is needed to run (`docker/cell.sh up`); the base is needed only to rebuild the
cell image there. `genie_sim/` and `geniesim_assets/` are bind-mounted, not baked in, so they must be on
the server either way (`scripts/server_setup.sh` + `scripts/sync_to_server.sh`). Keep any registry
copy PRIVATE: the image contains the NVIDIA Isaac Sim base, whose licence limits redistribution.
