# QNLP Poster

A picture-first conference poster explaining the toy Quantum NLP sentiment
classifier in this repo — written so a complete beginner (no quantum / ML
background) can follow it.

Built on the [Stanford LaTeX Poster Template](https://github.com/RylanSchaeffer/Stanford-LaTeX-Poster-Template)
(gemini theme + Stanford colors). Landscape, 120 cm × 72 cm, three columns.

## Build

Compile with **XeLaTeX** (the gemini theme uses `fontspec`):

```bash
cd poster
xelatex main.tex
```

Output: `main.pdf` (one page).

The theme normally wants the **Raleway** and **Lato** fonts. They are not
required here — `beamerthemegemini.sty` has been patched to fall back to the
default sans font when they are missing, so the poster compiles anywhere. To
get the intended look, install Raleway and Lato and recompile.

## Contents

- `main.tex` — the poster source. Word roles are color-coded consistently
  throughout (adjective = orange, subject = blue, verb = green, object = purple).
- `assets/loss_curve.png` — training curve, generated from the **current**
  circuit-simulation model. Regenerate with:
  ```bash
  cd ..
  PYTHONPATH=src python3 -c "from qnlp_tutorial.data import load_dataset, train_test_split; \
from qnlp_tutorial.quantum import ToyQNLPClassifier; \
from qnlp_tutorial.visualize import save_loss_curve; \
ds=load_dataset(); tr,_=train_test_split(ds); m=ToyQNLPClassifier.from_dataset(ds); \
save_loss_curve(m.fit(tr,epochs=50), assets_dir='poster/assets')"
  ```
- `beamerthemegemini.sty`, `beamercolorthemestanford.sty`, `stanford_logos/` —
  template support files.

## Editing notes

- Authors/affiliation are in the `\title`, `\author`, `\institute`, and
  `\footercontent` commands near the top of `main.tex` — edit those for your
  names.
- Header banner color: change the `headline` color in
  `beamercolorthemestanford.sty` (a few commented options are listed there).
- All diagrams are drawn inline with TikZ (no external image dependencies
  except the loss curve), so colors and labels can be tweaked directly.
