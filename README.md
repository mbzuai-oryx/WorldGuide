# WorldGuide — academic project page

This is the independent academic-style version of the WorldGuide research page. It is self-contained and does not import files from, or modify, `Project_Page`.

## Local preview

From this folder, run:

```sh
python3 -m http.server 8772 --bind 127.0.0.1
```

Visit http://127.0.0.1:8772/. Use an HTTP server rather than opening the HTML as a file because the page loads JSON data.

## GitHub Pages

Copy this folder's contents into the root of a repository, including `.nojekyll`. In the repository's Pages settings, choose **Deploy from a branch**, select your publishing branch, and select **/(root)**. No build command, npm dependencies, API keys, or backend are required.

All internal paths are relative, so the page supports both a `username.github.io` site and a repository subpath such as `username.github.io/worldguide/`. Do not replace relative asset URLs with absolute paths beginning with `/`.

## Content and assets

- `index.html`: paper title, five authors, abstract, method figures, both benchmark summaries, ablations, video comparisons, dataset description, and resources.
- `styles.css`: white academic layout, moderate typography, responsive tables and figures, and reduced-motion support. The design uses system fonts and needs no font service.
- `app.js`: method tabs, comparison controls, manifest loading with fallback, figure dialogs, benchmark selection, and configurable resource links.
- `manifest.json`: the full existing collection of 18 samples and nine models; no samples or model outputs have been removed.
- `results.json`: all ten reported metrics for 21 models on WorldGuide Bench and 21 models on Video-CraftBench.
- `resources.json`: editable Paper, GitHub, and Dataset destinations. Paper and GitHub are blank; Dataset is labeled “Coming soon” until a URL is supplied, following the current publication-link preference.
- `assets/`: independent copies of the latest figure renders, posters, and favicon. The existing local font assets retain their license but are not needed by this design.
- `.nojekyll`: enables direct static-file hosting on GitHub Pages.

Videos remain hosted remotely and play inside the page. The page does not provide an outgoing link to the Hugging Face results repository. The Dataset resource is reserved for the eventual training/benchmark data release.

The hosted manifest is fetched from `https://huggingface.co/datasets/ankanmbz/worldguide-results/resolve/main/manifest.json`. The bundled manifest is used if the remote request fails. Video paths are encoded and resolved against the same remote base. Hero autoplay respects reduced-motion and data-saving preferences. The comparison retains original clip durations and provides native playback controls, paired play/pause, restart, and retry.

## Paper and results provenance

The page was checked against the supplied 45-page manuscript at `../WorldGuide_ICLR.pdf`. Every numeric value in the two 21-model result tables was verified against Tables 2 and 3 of that PDF. The displayed ablations come from Table 4; the planner and human-evaluation text follows Table 5 and Appendix F. The PDF itself is not bundled or linked because publication links are currently left blank.

Both benchmark protocols are stated explicitly. On WorldGuide Bench, unstarred baselines receive reference actions, while WorldGuide and starred planner–executor baselines receive the initial image and task goal. On Video-CraftBench, every model receives the initial image and goal. The paper's p=0.10 caveat for the WorldGuide Bench comparison with MiniMax-H3 is retained. Numerical differences are labeled in percentage points.

The figure assets match the latest requested sources:

| Asset | PDF source in `../WorldGuide-ICLR/Figures/` |
| --- | --- |
| `architecture.webp` | `WorldGuide_main_v10.pdf` |
| `paradigms.webp` | `WorldGuide_vs_all_v6.pdf` |
| `memory.webp` | `mem_hirrchy.pdf` |
| `qualitative.webp` | `result_3.pdf` |
| `dataset.webp` | `worldGuide_Dataset.pdf` |

The expanded fried-rice figure remains collapsible. Other method figures are shown directly and can be enlarged. No conference-acceptance claim, invented affiliations, or unverified citation metadata is included.

## Validation

Checked in Chrome at desktop, tablet (768 px), and mobile (390 and 320 px) widths. Both 21-model benchmark tables, all 18 video samples, eight selectable baselines, remote playback, keyboard navigation, figure enlargement, manifest fallback, and video error recovery passed. A GitHub Pages-style repository subpath was also checked. The automated WCAG A/AA scan reported no violations. All 37 original `Project_Page` files were verified unchanged by SHA-256 comparison.
