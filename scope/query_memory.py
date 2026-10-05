"""Verified query keyframes use the existing M3 immutable evidence import."""

import json

from .entities import EntityStore
from .mapping import VoxelMap
from .memory import process_episode
from .memory_store import MapAlignment,MemoryStore,episode_snapshot
from .semantic_pipeline import SemanticRun
from .semantic_storage import save_semantic_run
from .storage import save_episode,save_map


def save_query_memory(pipeline,output,world_frame):
    if not pipeline.runtime.state("query_memory").config.enabled or not pipeline.memory_evidence:
        return {"state":"UNAVAILABLE","reason":"No verified query with sufficient calibrated depth"}
    records = sorted(pipeline.memory_evidence.values(),key=lambda r:r[0].timestamp_s)
    frames = [r[0] for r in records]
    if any(f.intrinsics != frames[0].intrinsics for f in frames):
        raise ValueError("Memory episode requires stable calibration")
    detections = [list(r[1].values()) for r in records]
    projections = [list(r[2].values()) for r in records]
    store,mapping = EntityStore(),VoxelMap(pipeline.core.mapping.config)
    for f,objects in zip(frames,projections):
        store.add_frame(objects);mapping.integrate(f)
    run = SemanticRun(frames,detections,projections,store,[],"open-vocabulary-query-evidence","entities")
    directory = output/"memory-episode"
    save_episode(directory,frames,"verified-open-vocabulary")
    save_map(directory,mapping)
    save_semantic_run(directory,run)
    memory = MemoryStore(output/"memory.sqlite")
    try:
        identity = "query-"+output.name
        memory.import_snapshot(episode_snapshot(identity,directory,run,MapAlignment(world_frame,"known_shared")))
        process_episode(memory,identity)
        result = {"state":"PERSISTED","global_entities":len(memory.beliefs()),"world_frame":world_frame}
    finally:
        memory.close()
    (output/"query-events.json").write_text(json.dumps(pipeline.events,indent=2)+"\n")
    return result
