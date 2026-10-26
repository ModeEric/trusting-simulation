# Release preparation, October 26, 2026

The recovered research archive is retained locally and is not the public release.
Changes made for publication:

- Added an MIT license for original work, retained upstream licenses/notices, and
  excluded third-party papers, website snapshots, per-trial derived tables and
  large prediction outputs from Git.
- Added a manifest of the 55 required inputs and a bounded, checksum-enforcing
  acquisition/import command. The original acquisition audit remains unchanged.
- Limited raw-input verification to required computational inputs. The original
  full log also includes unused research PDFs, repository listings and README
  snapshots; they are no longer required to execute the released pipeline.
- Read SIMPLER's immutable source revision from its existing acquisition record,
  rather than requiring an unused GitHub commit-response snapshot.
- Created missing output directories before starting a fresh reproduction.
- Added a separately labeled public-data reproduction for SIMPLER and LIBERO-Plus
  and six regression tests for checksum preservation and safe acquisition.
- Replaced the landing page and added an overview covering all five studies.
  Preserved the original two-page note as the physical-evidence report.

No predictor, split, seed, metric, tuning grid, analysis plan or frozen statistical
source was changed. Existing freeze hashes are retained. Publication packaging
does not turn the archived plans into an external preregistration.
