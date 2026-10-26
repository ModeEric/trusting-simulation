# Third-party materials and release boundaries

The root MIT license covers Eric Modesitt's original analysis code and writing.
It does not relicense upstream material or grant rights to benchmark assets,
models, videos, task instructions, or third-party websites.

| Source | Evidence retained | Public release decision |
| --- | --- | --- |
| SIMPLER, commit `eddb5690749a1e34267807c0f7b5eede323918d7` | [Pinned MIT license](https://github.com/simpler-env/SimplerEnv/blob/eddb5690749a1e34267807c0f7b5eede323918d7/LICENSE) | Include `simpler_metrics.py`, `simpler_calc_metrics.py` and their original `simpler_LICENSE`. No paper PDF or simulator assets. |
| Targeted Data LIBERO-Plus, commit `554a0d06924f64f15551119e393373a9bc008771` | Archived Apache-2.0 LICENSE and NOTICE; NOTICE explicitly describes results as Apache-2.0 | Include the archived outcome release with its original license, notices, README, claims, provenance and checksums. It contains no weights or benchmark simulator assets. |
| RoboDojo website and rollout metadata | Repository code license is separate from website/data rights | Exclude website HTML/JavaScript and rollout metadata. Acquire directly from upstream or import from the researcher's original private archive. |
| RoboDojo paper, arXiv:2607.04434v1/v3 | [Article license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) grants distribution to arXiv | Exclude full article HTML/PDF. Retain citations and acquisition hashes. |
| Robocurve September 23 report and trial CSVs | Public source URLs; no explicit redistribution permission established in this review | Exclude source HTML and trial/cell CSVs. Retain aggregate analysis and original figures. |
| RoboTwin website leaderboard, commit `0abb5a03455073fa74047a1c4015b0f88cedc254` | Pinned source URL; no explicit leaderboard redistribution permission established in this review | Exclude raw leaderboard and derived per-policy task matrices; acquire from the pinned upstream URL. |

This is a record of packaging decisions and the permissions found, rather than
a claim that absent license evidence proves copying is prohibited. No general
permission is inferred from public accessibility or from an unrelated code license.

The original full acquisition history is in `data/download_audit.json`.
`data/release_inputs.json` identifies only the inputs required by the released
pipeline and preserves their original hashes. The larger recovery ZIP, local
application materials, raw website snapshots, processed per-episode/per-policy
data and large prediction exports are excluded from Git and release attachments.

If any excluded snapshot is unavailable upstream, the public repository cannot
by itself supply that snapshot. Do not replace its hash or substitute a new
leaderboard silently. See [reproduction](docs/reproduction.md) for the exact
archive-import path and the public download audit.

Original retained notices:

- [SIMPLER MIT license](data/raw/simpler_LICENSE)
- [Targeted Data Apache-2.0 license](data/raw/libero_plus/LICENSE)
- [Targeted Data NOTICE](data/raw/libero_plus/NOTICE)
- [Targeted Data source README](data/raw/libero_plus/README.md)

Source authors retain ownership of their work. No affiliation with or endorsement
by SIMPLER, RoboDojo, RoboTwin, Robocurve, or the Targeted Data authors is implied.
