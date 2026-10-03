# Milestone 3: persistent world model and memory

Status: implementation in progress on `codex/milestone-3-memory`, based on the completed Milestone 2 head `d418400`.

## Representation

Saved RGB-D and semantic episode assets remain on disk. SQLite stores hashes and references to those assets, detached episode-local entity hypotheses, explicit map alignment, identity audits, append-only belief revisions, and change events. SQL triggers reject updates and deletion of historical rows. Asset checks reject replay after evidence changes. Raw observations, local entities, global identity links, and current beliefs remain separate.

Map alignment is supplied: either a named known shared frame or an external rigid transform with translation and rotation uncertainty. No automatic map registration is assumed. Synthetic repeat visits use exact shared coordinates and simulated timestamps. Identity scores and displayed confidence are evidence heuristics, not calibrated correctness probabilities.

## Checkpoints

- `6e546f0 Persist immutable semantic episodes`: episode imports, asset integrity, frame IDs, and saved-mask replay; pushed.

## Identity and change evidence

The matcher combines class-distribution compatibility, observed dimensions, centered visible-surface shape, location compatibility scaled by a small mobility prior and elapsed time, nearby-class distances, and supplied alignment uncertainty. A table-location contradiction is an alignment diagnostic, not an estimated registration. Candidates retain all component scores and gate reasons. One-to-one links require score and competing-candidate margins; otherwise they remain unresolved. Unresolved observations do not create a confident replacement ID. No appearance embeddings are used.

Positive links append a new global belief revision. A supported displacement records `MOVED`; a weaker accepted displacement records `POSSIBLY_MOVED`. Previous locations stay in earlier revisions and event records. The lifecycle uses `VISIBLE`, `NOT_CURRENTLY_OBSERVED`, `POSSIBLY_MOVED`, `POSSIBLY_MISSING`, and `IDENTITY_UNCERTAIN`; confidence is separate from state.

An unmatched global hypothesis is checked against each later camera view. The old visible-surface samples must project into the image, have sufficient valid depth and pixel area, avoid nearer occluding surfaces, and have measured rays passing beyond the old object region. M1 free-space classification is retained as an additional diagnostic. Two distinct adequate viewpoints are required for `MISSING_HYPOTHESIS`; history is never deleted. Out-of-view, occluded, invalid-depth, and detector-miss cases retain the old positive sighting. Real detector miss probabilities are unmeasured, so negative detection reliability defaults to zero for real models.
