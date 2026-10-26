# Research inputs

This folder includes only the licensed SIMPLER files and Targeted Data LIBERO-Plus
release listed in [third-party notices](../../THIRD_PARTY_NOTICES.md).

Other required inputs are intentionally absent from Git. Acquire them with:

```bash
python scripts/acquire_inputs.py
```

Every input must match its original SHA-256 in `data/release_inputs.json`.
Acquisition never executes downloaded Python, HTML or JavaScript. Analysis reads
literal constants and tables. The offline reproduction command never downloads.

Some upstream URLs may no longer serve the archived bytes. The researcher can
restore excluded inputs from the recovered archive with:

```bash
python scripts/acquire_inputs.py --archive /path/to/Trusting_Simulation_Analysis.zip
```

That archive is not a public release asset. See `docs/reproduction.md` for the
verified public-data availability limitation.
