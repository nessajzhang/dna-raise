# Frozen core simulation source data | DNA-RAISE

Version: p4_v8810_20261024. These are existing frozen author-generated simulation records and source tables; publication does not rerun or alter the studies. Files are byte-identical to their preserved sources, identified in SOURCE_DATA_MANIFEST.csv. The initial public scientific-code commit is9bb2934264865ef143dc6ba7d63d0fedb001e591. The frozen research-package identity is SHA256 fb78d06af184facf63f47d8ad3b5b6758a75c9f107c55d4be678a5f4a1b609ba.

## Contents and evidence boundary

- figure_source/: current author-generated simulation plotting inputs, including the Figure3 coordinates and all independent V8.8 spending-profile results. Current schematic Figure1 contains no measured data and is not redistributed here.
- simulation_summaries/: frozen synthetic source/design/diagnostic tables for the earlier and supplementary comparisons. Secondary individual-episode caches remain in the full research archive.
- v8_7/: preregistration/design identities, tuning, held-out and negative-control mission records, summaries, gates, seed ledgers and the locked selection. The held-out file contains240000 policy-mission rows for60000 root missions, not240000 independent missions.
- v8_8/: the independent design, tuning and finalized held-out spending-profile records plus the offline control. The failed maintenance-burden gate is retained, alongside all seven reported test profiles. Failed/partial intermediate postprocessing is not substituted for the finalized frozen tables.

The data support review of the reported simulation values and the principal locked mission-level comparisons. They are not a complete copy of every historical release or environment. The reference implementation returns mission summaries, not a persistent per-look certificate log. Code for the studies and original plotting/analysis programs remains in the repository's existing Research_Layers/ and Code/ directories. Original drivers can launch simulation/bootstrap jobs; this data publication does not execute them.

## Read-only check

From repository root, run python verify_public_data.py. The checker verifies exact data hashes and tallies stored V8.7 outcomes plus the locked V8.8 gate statuses. It generates no tasks, fits, bootstrap samples or figures. The original scientific programs are unchanged.

## Units and interpretation

Read root_seed as a string or exact unsigned64-bit integer, never floating point. Different policies/profiles sharing a root are repeated views of one mission; tuning/test/control cohorts retain their recorded seed namespaces. Internal +screen means the DNA-RAISE authorization layer. Latent true hazard and chain diagnostics are offline simulator audit quantities, not deployed controller inputs. Mission-level authorization error, risk conditional on approval and whole-archive service reliability remain distinct. Do not infer physical qualification from model simulation records.

## External molecular data and rights

Third-party qPCR and sequencing inputs and their derived endpoint tables are not newly redistributed in this update. Obtain those data from the original sources cited in the article/SI, under their own access and licence terms; existing public acquisition/reanalysis code identifies the provenance. This update publishes the author's own synthetic records only. No new project reuse licence is assigned; the author's prior licence decision remains unresolved. This availability record does not establish image/AI creation history, material permissions or final journal approval.
