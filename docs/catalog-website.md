# Catalog website

**Browse the catalog online: [ersilia-os.github.io/eosbench](https://ersilia-os.github.io/eosbench/)**

The catalog is also browsable as a static website, published with GitHub Pages: a single page
with client-side search, sorting, and filters over every dataset.

`scripts/export_catalog.py` reads the bundled metadata (the same source as `eosbench catalog`)
across all sources and both tasks, writes `site/catalog.json`, and HEAD-probes each dataset's
`data.csv` on the public S3 bucket so each row shows an available/pending badge — reflecting
what's actually fetchable, not just what has metadata.

The `.github/workflows/pages.yml` workflow rebuilds and redeploys the site on every push to
`main` (and can be triggered manually from the Actions tab). Because the availability badges
are checked at build time, re-run the workflow after uploading new datasets to S3 to refresh
them. To preview locally:

```bash
python scripts/export_catalog.py        # writes site/catalog.json (with live S3 checks)
python -m http.server -d site           # open http://localhost:8000
```

**One-time setup:** in the GitHub repo, Settings → Pages → Build and deployment → Source =
GitHub Actions. After that the workflow publishes automatically.
