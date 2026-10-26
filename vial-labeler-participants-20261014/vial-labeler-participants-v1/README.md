# Vial labeling tool for hackathon participants

This standalone tool creates an offline browser page for **manual** labeling of 60 positions in a vial fixture image. It makes no class predictions, requires no API key or GPU, and does not send images over the network. Two public images are included as geometry references and examples. No organizer review history or held-out annotations are included.

## Try the review page

Open `demo/index.html` in Chrome, Edge, or Firefox. No installation is required to use this example. All positions start unresolved. The circles mark the target insertion positions; a crop can contain neighboring vials. Inspect the full scene as well as the selected crop before choosing a class.

1. Select a scene and good or scrap tray. Click a numbered circle or crop tile.
2. Choose a class, add an observation if useful, and check **I reviewed this position**. Changing its class clears confirmation.
3. Use **Previous** and **Next** to move between positions. Green checks mean confirmed; orange marks mean pending.
4. Click **Export review JSON** to download your work. Browser storage is a convenience, not a backup: export regularly. Closing the page without exporting can lose edits if storage is cleared or unavailable.
5. **Load review JSON** restores compatible exports into the same page. To update CSV files and fixture summaries, import the download with Python as shown below. Browser edits do not rewrite files on disk.

## Install

Use Python 3.10 or newer. Unzip the package, open a terminal in its folder, and run:

```sh
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Only NumPy and Pillow are required. Internet access is needed to install dependencies unless you already have them; labeling itself is offline.

## Label your public images

The public dataset is distributed separately. Supply its `images` folder and explicit image basenames:

```sh
python labeler.py run --images /path/to/public_release/images \
  --files tray_000030.jpg tray_000070.jpg --output review-01
```

To try the included example:

```sh
python labeler.py run --output my-first-review
```

Open the new folder's `index.html`. Each output includes source image copies, unmarked crops, annotated scenes, contact sheets, `labels.json`, `slots.csv`, and `manifest.json`. The HTML embeds the scene and marked crops, so it can open without a server. Output folders use relative image paths and can be moved together. Existing output directories are rejected to protect previous reviews.

## Geometry and alignment

The supplied geometry supports **1920 × 1080** images from the public fixed-camera view. Small fixture translations are estimated using the two public geometry references. Check every ring against its target hole before labeling; a low alignment residual is not proof that all crops are correct. Unfamiliar alignment is rejected. A changed camera, image size, rotation, or lens requires recalibration rather than guessed labels.

For an inspected geometry profile with a verified translation, you may specify it explicitly:

```sh
python labeler.py run --images /path/to/images --files your-image.jpg \
  --profile view_001 --offset 0 0 --output review-manual-alignment
```

This bypasses automatic alignment checking, so use it only after visually verifying all target positions. For a custom view of the same resolution, copy `calibration.json`, adjust the four extreme hole centers (top-left, top-right, bottom-left, bottom-right) and crop offsets for both trays, and use `--calibration your-calibration.json --profile your-profile --offset 0 0`. Automatic alignment with a custom calibration also requires its matching reference JPEGs under a neighboring `examples/` folder and their correct hashes.

These rectangles are inherited review context crops, **not** tight object boxes or the separate AOI fixed-body 256 × 256 export. A crop can clip a projecting damaged body; use the full scene for that judgment. Do not treat its rectangle as detector ground truth.

## Import your reviewed download

Keep the original output folder as the immutable template. Import the browser download into a new folder:

```sh
python labeler.py import-review /path/to/labels-DATE.json \
  --template review-01/labels.json --output review-01-confirmed
```

The importer checks source bytes, image identities, slot geometry and review states, then recomputes presence, fixture summaries and CSV rows. Only class, note and confirmation fields are editable. Subset exports update matching images while retaining other template images. Changing source images or geometry requires a new review page. Unresolved classes cannot be confirmed.

## Class and sequence guide

| Class | Visible condition |
| --- | --- |
| `pass` | The visible cap and vial body appear acceptable. |
| `missing_cap` | A vial is present without a cap. |
| `crooked_cap` | The cap is tilted or improperly seated relative to the mouth; a leaning vial alone is insufficient. |
| `damaged_vial` | The visible body is crushed, torn, broken or visibly deformed. |
| `no_vial` | The target position is empty. |

Leave a position unresolved when glare, overlap or visibility prevents a supported decision. Unresolved means `class: null`; it is not a sixth class. When several defects coexist, use damaged vial first, then crooked cap, then missing cap, and record other observations in the note. Transparent plastic bodies, yellow caps, metal supports and barrel distortion can make angles or apparent occupancy misleading. An open vial mouth is not an empty hole.

Each tray has six columns of five positions. Start at the top right, count downward, then move one column left. G01–G30 are good-tray positions; S01–S30 are scrap. An empty position before a later occupied position is a possible missed placement. Trailing empty positions may be not yet filled. A final missed placement needs a placement-count or completion signal to establish. Visible pass in the scrap tray does not establish a routing mistake because upstream inspection can see hidden defects.

Use only confirmed labels for training or evaluation. Keep each image and its crops together when making splits, and keep related public images 1 and 6 together. This tool does not establish classifier accuracy or certify fill level, seal integrity or hidden defects.

## Tests and troubleshooting

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
```

Optional browser tests need Node.js and Playwright: install Playwright in your development environment, then run `node tests/test_review_ui.cjs demo/index.html`. Browser dependencies are not needed for normal use.

If alignment fails, inspect the scene and calibration; do not force confirmation. If import reports a metadata mismatch, use the original matching `labels.json` template and its unchanged source-image folder. If browser storage is unavailable, JSON export/import still provides a portable save workflow. If output already exists, choose a new directory.

The ZIP includes `SHA256SUMS` for integrity checking and `PACKAGE.json` for its version and file inventory. The source is included for participants to inspect and adapt for the hackathon. This package does not add a new software or dataset license; any redistribution remains subject to the event's supplied terms.
