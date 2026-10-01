# Dataset Schema

## Overview

Each of the 200 samples consists of a pair of hand-drawn sketches (desktop and
mobile), a reference HTML implementation, and a metadata file. Samples share a
zero-padded four-digit identifier (`0001` … `0200`).

## Directory Structure

```
dataset/
  sketches/
    {id}-d-v2.png                    desktop sketch (1280 px layout; from Sketch2Code)
    {id}-m-v2.png                    mobile sketch (375 px layout; drawn for Responsive-Bench)
  references/
    {id}.html                        benchmark reference HTML (accessibility-augmented)
    analysis/{id}.json               stage-1 sketch analysis used to augment the reference
  metadata/
    {id}.json                        per-sample metadata (see below)

  source_html/
    {id}.html                        original Sketch2Code / Design2Code HTML
  layout_similarity_study/           Appendix D.3 only
    html_and_screenshots/{id}.html   reference rebuilt from the source HTML + its screenshots
    screenshots_only/{id}.html       reference rebuilt from screenshots alone
```

`sketches/`, `references/`, and `metadata/` are all the benchmark needs.
`source_html/` holds the pages the references were augmented from and is the
input to the augmentation modes. `layout_similarity_study/` is used only for the
original-vs-modified Layout Similarity comparison in Appendix D.3.

Every HTML directory contains a copy of `rick.jpg`, the placeholder image the
Design2Code pages load by relative path.

## Metadata Schema

```json
{
  "id": "0001",
  "sketches": {
    "desktop": "sketches/0001-d-v2.png",
    "mobile": "sketches/0001-m-v2.png"
  },
  "reference_html": "references/0001.html",
  "source_html": "source_html/0001.html",
  "created_at": "2026-03-02T18:31:31.658845+00:00",
  "page_type": "unknown"
}
```

| Field | Meaning |
|---|---|
| `id` | Sample identifier; matches the metadata filename. |
| `sketches` | Paths (relative to `dataset/`) to the desktop and mobile sketches. |
| `reference_html` | Path (relative to `dataset/`) to the benchmark reference HTML. |
| `source_html` | Optional. Path to the original page the reference was built from. |
| `created_at` | ISO 8601 timestamp of metadata creation. |
| `page_type` | High-level page category, or `"unknown"`. |

## Validation

```bash
PYTHONPATH=src python -m responsive.dataset.loader validate --root dataset
```

This checks that every metadata file conforms to the schema, that all referenced
files exist and are non-empty, and that sample IDs are unique.
