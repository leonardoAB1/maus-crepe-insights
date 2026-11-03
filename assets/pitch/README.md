# Final presentation

The CrêpeFlow Team's BaselHack 2026 pitch for the Maus Robotics challenge,
as a [reveal.js](https://revealjs.com) deck. The story runs data first
(run history, control theory and a designed experiment), then vision
(segmentation, fold state and pose, running on a Raspberry Pi 5).

| File | What it is |
| --- | --- |
| `index.html` | the deck; needs internet for reveal.js and the fonts |
| `crepeflow-pitch.pdf` | the same 15 slides as a PDF, works offline |
| `video/` | the two annotated demo runs on slide 10, with poster frames |
| `source/` | the template and the scripts that build the deck and the PDF |

## Presenting

Open `index.html` in Chrome or Edge. Keep it next to the `video/` folder.

- Arrow keys or space: next and previous slide
- `F`: fullscreen
- `S`: speaker notes with the talk track, in a second window
- `Esc`: overview of all slides

The Raspberry Pi addresses on slide 11 were on the venue network.

## Rebuilding

Edit `source/template.html`, then from the repository root:

```bash
python assets/pitch/source/build.py     # writes assets/pitch/index.html
python assets/pitch/source/make_pdf.py  # writes crepeflow-pitch.pdf
```

`build.py` inlines the images from `assets/` and downloads reveal.js's
stylesheet once into `source/vendor/` (gitignored). `make_pdf.py` needs
Chrome or Edge (set `CHROME` if it is not found) and Pillow; it renders each
slide as an image, so the PDF text is not selectable.

The numbers on the slides come from [project status](../../documentation/status.md)
and the pages it links. The run videos are Maus footage, cleared for this
repository only.
