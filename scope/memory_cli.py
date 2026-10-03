"""Offline memory commands; no robot, language, or planning dependencies."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from .memory import process_episode
from .memory_identity import match_episode
from .memory_store import MapAlignment, MemoryStore, episode_snapshot
from .semantic_storage import load_semantic_run


def _alignment(args, suffix=""):
    path = getattr(args, "alignment"+suffix, None)
    if path:
        return MapAlignment.from_dict(json.loads(path.read_text()))
    if not args.shared_frame:
        raise ValueError("Declare --shared-frame or supply alignment JSON for every episode")
    return MapAlignment(args.shared_frame, "known_shared")


def _output(name):
    return Path("runs") / f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _print(data):
    print(json.dumps(data, indent=2))


def register_memory_commands(commands):
    memory = commands.add_parser("memory", help="Import, build, or inspect persistent offline memory")
    actions = memory.add_subparsers(dest="memory_action", required=True)
    add = actions.add_parser("add", help="Import immutable episode evidence without linking it")
    add.add_argument("directory", type=Path)
    add.add_argument("--db", type=Path, required=True)
    add.add_argument("--episode-id", required=True)
    group = add.add_mutually_exclusive_group(required=True)
    group.add_argument("--shared-frame", help="Explicit claim of an exact common coordinate frame")
    group.add_argument("--alignment", type=Path, help="Supplied rigid transform and uncertainty JSON")
    add.add_argument("--time-offset-s", type=float, default=0.)
    add.add_argument("--clock", choices=("source_seconds","unix_utc"), default="source_seconds")
    add.add_argument("--negative-reliability", type=float,
                     help="Externally justified detector absence reliability; real default is zero")
    for name in ("build","list"):
        action = actions.add_parser(name)
        action.add_argument("--db", type=Path, required=True)
    episode = actions.add_parser("episode", help="Inspect the stored episode snapshot and local entities")
    episode.add_argument("episode_id")
    episode.add_argument("--db", type=Path, required=True)
    for name in ("history","last-seen","link-entities"):
        command = commands.add_parser(name)
        command.add_argument("entity_or_episode")
        command.add_argument("--db", type=Path, required=True)
    compare = commands.add_parser("compare-episodes", help="Run only identity scoring on two saved semantic runs")
    compare.add_argument("episode_a", type=Path)
    compare.add_argument("episode_b", type=Path)
    compare.add_argument("--shared-frame")
    compare.add_argument("--alignment-a", type=Path)
    compare.add_argument("--alignment-b", type=Path)
    replay = commands.add_parser("replay-memory", help="Record and open episode-end memory snapshots in Rerun")
    replay.add_argument("--db", type=Path, required=True)
    replay.add_argument("--output", type=Path)
    replay.add_argument("--no-viewer", action="store_true")
    from .repeat_visits import SCENARIOS
    demo = commands.add_parser("memory-demo", help="Build a deterministic two-visit synthetic memory demo")
    demo.add_argument("scenario", choices=SCENARIOS, nargs="?", default="moved_backpack")
    demo.add_argument("--output", type=Path)
    demo.add_argument("--frames", type=int, default=8)
    demo.add_argument("--width", type=int, default=192)
    demo.add_argument("--no-viewer", action="store_true")
    benchmark = commands.add_parser("benchmark-memory", help="Measure repeat-visit identity and memory failures")
    benchmark.add_argument("--output", type=Path)


def run_memory_command(args):
    if args.command == "compare-episodes":
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp)/"memory.sqlite")
            a = episode_snapshot("compare-a", args.episode_a, load_semantic_run(args.episode_a), _alignment(args,"_a"))
            b = episode_snapshot("compare-b", args.episode_b, load_semantic_run(args.episode_b), _alignment(args,"_b"))
            store.import_snapshot(a)
            process_episode(store, "compare-a")
            _print(match_episode(b, store.beliefs()))
            store.close()
        return 0
    if args.command == "memory-demo":
        from .repeat_visits import scenario_visits, save_visit
        from .memory_viewer import record_memory
        from .viewer import open_recording
        if args.frames<3 or args.width<64:
            raise ValueError("Memory demos require at least 3 frames and image width 64")
        output = args.output or _output("memory-"+args.scenario)
        metadata = {"scenario": args.scenario, "frames": args.frames, "width": args.width}
        existing = output/"scenario.json"
        if output.exists() and any(output.iterdir()) and not existing.exists():
            raise ValueError("Use an empty output directory; existing assets will not be overwritten")
        if existing.exists() and json.loads(existing.read_text()) != metadata:
            raise ValueError("Output already contains a different immutable scenario")
        output.mkdir(parents=True, exist_ok=True)
        (output/"scenario.json").write_text(json.dumps(metadata, indent=2)+"\n")
        store = MemoryStore(output/"memory.sqlite")
        if not store.episodes():
            for visit in scenario_visits(args.scenario, args.frames, args.width):
                store.import_snapshot(save_visit(visit,output/visit.episode_id))
                process_episode(store,visit.episode_id)
        if len(store.episodes())!=2:
            raise ValueError("Incomplete scenario; use a new output directory")
        recording = record_memory(store,output)
        _print({"database":str(store.path), "recording":str(recording.resolve()),
                "entities": [{"global_id":gid,"status":b["status"],
                              "position_m":b["geometry"]["center_m"]} for gid,b in store.beliefs().items()]})
        store.close()
        if not args.no_viewer:
            open_recording(recording)
        return 0
    if args.command == "benchmark-memory":
        from .memory_eval import benchmark_memory
        _print(benchmark_memory(args.output or _output("memory-benchmark")))
        return 0
    if not args.db.exists() and not (args.command=="memory" and args.memory_action=="add"):
        raise ValueError(f"Memory database does not exist: {args.db}")
    store = MemoryStore(args.db)
    try:
        if args.command == "memory":
            if args.memory_action == "add":
                run = load_semantic_run(args.directory)
                snapshot = episode_snapshot(args.episode_id,args.directory,run,_alignment(args),
                    time_offset_s=args.time_offset_s,time_domain=args.clock)
                if args.negative_reliability is not None:
                    if not 0<=args.negative_reliability<=1:
                        raise ValueError("Detector reliability must be in [0,1]")
                    snapshot["detector"]["negative_detection_reliability"] = args.negative_reliability
                store.import_snapshot(snapshot)
                _print({"imported":args.episode_id,"local_entities":len(snapshot["local_entities"]),
                        "alignment":snapshot["alignment"],"time_domain":snapshot["time_domain"]})
            elif args.memory_action == "build":
                for episode in store.episodes():
                    process_episode(store,episode["episode_id"])
                _print({"episodes":len(store.episodes()),"global_entities":len(store.beliefs()),
                        "events":len(store.history())})
            elif args.memory_action == "episode":
                store.verify_assets(args.episode_id)
                _print(store.episode(args.episode_id))
            else:
                _print({"episodes":[{"episode_id":e["episode_id"],"start_s":e["start_s"],"end_s":e["end_s"],
                                     "alignment":e["alignment"]} for e in store.episodes()],
                        "global_entities":[{"global_id":gid,"status":b["status"],
                                            "last_positive":b["last_positive"]} for gid,b in store.beliefs().items()]})
        elif args.command == "link-entities":
            stored = store.links(args.entity_or_episode)
            _print(stored if stored else match_episode(store.episode(args.entity_or_episode),store.beliefs()))
        elif args.command == "last-seen":
            result = store.last_seen(args.entity_or_episode)
            if result["time_domain"]=="unix_utc":
                result["last_positive_utc"] = datetime.fromtimestamp(
                    result["last_positive"]["time_s"],timezone.utc).isoformat()
            _print(result)
        elif args.command == "history":
            if args.entity_or_episode not in store.beliefs():
                raise ValueError("Unknown global entity")
            _print(store.history(args.entity_or_episode))
        elif args.command == "replay-memory":
            from .memory_viewer import record_memory
            from .viewer import open_recording
            recording = record_memory(store,args.output or args.db.parent)
            print(f"Saved memory recording: {recording.resolve()}")
            if not args.no_viewer:
                open_recording(recording)
        return 0
    finally:
        store.close()
