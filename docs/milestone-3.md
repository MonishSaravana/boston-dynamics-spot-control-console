# Milestone 3: persistent world model and memory

Status: implementation in progress on `codex/milestone-3-memory`, based on the completed Milestone 2 head `d418400`.

## Representation

Saved RGB-D and semantic episode assets remain on disk. SQLite stores hashes and references to those assets, detached episode-local entity hypotheses, explicit map alignment, identity audits, append-only belief revisions, and change events. SQL triggers reject updates and deletion of historical rows. Asset checks reject replay after evidence changes. Raw observations, local entities, global identity links, and current beliefs remain separate.

Map alignment is supplied: either a named known shared frame or an external rigid transform with translation and rotation uncertainty. No automatic map registration is assumed. Synthetic repeat visits use exact shared coordinates and simulated timestamps. Identity scores and displayed confidence are evidence heuristics, not calibrated correctness probabilities.

## Checkpoints

The first checkpoint adds immutable episode imports and saved-mask replay. Further identity, visibility, viewer, and benchmark checkpoints will be recorded here as they pass their tests.
