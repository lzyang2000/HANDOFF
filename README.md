# HANDOFF — Project Page

Project website for **HANDOFF: Humanoid Agentic Task-Space Whole-Body Control via Distilled Complementary Teachers**.

- 📄 Paper: https://arxiv.org/abs/2606.06493
- 💻 Code: https://github.com/lzyang2000/HANDOFF
- 🌐 Live page: https://lzyang2000.github.io/handoff_wbc/

Built from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
(adopted from the [Nerfies](https://nerfies.github.io) project page).

## Structure

- `index.html` — the entire page (hero, abstract, 10-D interface, method, agentic planner, task-rollout video carousel, results, BibTeX).
- `static/images/` — figures copied from the paper (`head`, `system`, `agent`, `cbf-filter`, `experiment_snapshots`, velocity/workspace plots).
- `static/videos/` — task rollout clips, compressed to 720p VBR H.264 (`task1`–`task6`, `teleop`) plus poster frames.
- `static/css`, `static/js` — Bulma + carousel/slider assets from the template.

## Local preview

```bash
python -m http.server 8000
# open http://localhost:8000
```

## Deploy (GitHub Pages)

Push to the `handoff_wbc` repo and enable Pages on the `main` branch (root):

```bash
git remote add origin git@github.com:lzyang2000/handoff_wbc.git
git push -u origin main
# GitHub → Settings → Pages → Source: main / root
```

## Re-compressing videos

Source clips live in the paper repo's `website/vids/`. To regenerate the web versions:

```bash
for f in task1 task2 task3 task4 task5 task6 teleop; do
  ffmpeg -i "$SRC/$f.mp4" -vf "scale=-2:720" -c:v libx264 -crf 26 -preset slow \
    -movflags +faststart -an "static/videos/$f.mp4"
  ffmpeg -i "static/videos/$f.mp4" -frames:v 1 -q:v 3 "static/videos/${f}_poster.jpg"
done
```
