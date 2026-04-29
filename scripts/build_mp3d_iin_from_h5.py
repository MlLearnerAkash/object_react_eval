"""
Build an `hm3d_iin_val`-style episode dataset from an E3D action H5 file.

Each H5 episode (key = RXR ``instruction_id``) is converted into one folder
named ``<scan>_<EPID7d>_<class>_<inst>_`` under ``<out_root>/mp3d_iin_<split>/``,
mirroring the layout of ``/data/ws/object-rel-nav/data/hm3d_iin_val/`` so the
existing eval pipeline (``main.py`` with ``goal_source: lang_e3d``) can consume
it after pointing ``path_scenes_root`` at the MP3D scenes directory.

Per-episode files written:
  - agent_states.npy           list[habitat_sim.AgentState] (one per H5 frame)
  - images/<idx:05d>.png       RGB from H5
  - images_sem/<idx:05d>.npy   semantic instance map (int32, 0 = background)
  - instruction.txt            language instruction
  - nodes_gt_topometric.pickle raw graph bytes copied from H5
  - goalImg.png                last-frame RGB
  - instructions.lmdb/         per-frame instruction LMDB (see below)

The ``instructions.lmdb`` directory is written inside each episode folder and
stores per-frame records keyed by ``{seq_frame_idx:03d}`` where
``seq_frame_idx`` is the 0-indexed position of the frame within the episode's
sorted frame list — matching the ``agent_states`` array index used by the eval
pipeline for GT-localization step lookups.  Each record is a pickle dict:
    {"episodic_instruction": str, "next_action_instruction": str}
``next_action_instruction`` is an empty string when absent from the H5 frame.
The eval pipeline reads this LMDB to feed per-frame next-action instructions to
the cost-predictor (set ``instruction_type: next_action`` in the eval config).

Run:
  /home/opervu-user/miniconda3/envs/nav/bin/python \\
      eval/scripts/build_mp3d_iin_from_h5.py \\
      --h5 /media/opervu-user/Data2/ws/data_langgeonet_e3d_action/e3d_test.h5 \\
      --rxr-dir /data/dataset/RXR/rxr-data \\
      --out-root /data/ws/VLN-CE/controller/object_react/eval/data \\
      --split test
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import types
from pathlib import Path

import cv2
import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Mock _magnum so MP3D-pickled graphs unpickle without the native extension.
# Mirrors train/lange3dnet_train/joint_dataset.py::_install_mock_magnum.
# Only used when habitat_sim/magnum aren't available (real magnum supersedes it).
# ---------------------------------------------------------------------------
def _install_mock_magnum() -> None:
    if "_magnum" in sys.modules:
        return

    class _MagnumBase:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            self._state = state

    class _Vector3(_MagnumBase):
        def as_array(self) -> np.ndarray:
            return np.frombuffer(self._state, dtype=np.float32).astype(np.float64)

    mod = types.ModuleType("_magnum")
    mod.Vector3 = _Vector3
    for _name in ("Vector4", "Matrix4", "Matrix3", "Quaternion",
                  "Rad", "Deg", "Range3D", "Range2D"):
        setattr(mod, _name, type(_name, (_MagnumBase,), {}))
    sys.modules["_magnum"] = mod


# Try real habitat_sim/magnum first; fall back to mock for unpickling only.
try:
    import habitat_sim                              # noqa: E402
except ImportError:
    habitat_sim = None  # type: ignore
import quaternion as npq                            # noqa: E402  (numpy-quaternion)

if "_magnum" not in sys.modules:
    _install_mock_magnum()


# ---------------------------------------------------------------------------
# RXR instruction_id -> scan mapping
# ---------------------------------------------------------------------------
_RXR_FILES = (
    "rxr_train_guide.jsonl",
    "rxr_train_guide.jsonl.gz",
    "rxr_val_seen_guide.jsonl",
    "rxr_val_seen_guide.jsonl.gz",
    "rxr_val_unseen_guide.jsonl",
    "rxr_val_unseen_guide.jsonl.gz",
    "rxr_test_standard_public_guide.jsonl.gz",
    "rxr_test_challenge_public_guide.jsonl.gz",
)


def build_rxr_scene_map(rxr_dir: Path, wanted_ids: set[str]) -> dict[str, str]:
    """Return {str(instruction_id): scan} for ids found across RXR jsonl files."""
    out: dict[str, str] = {}
    remaining = set(wanted_ids)
    for fname in _RXR_FILES:
        if not remaining:
            break
        path = rxr_dir / fname
        if not path.exists():
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f:
            for line in f:
                if not remaining:
                    break
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iid = d.get("instruction_id")
                scan = d.get("scan")
                if iid is None or scan is None:
                    continue
                key = str(iid)
                if key in remaining:
                    out[key] = scan
                    remaining.discard(key)
    return out


# ---------------------------------------------------------------------------
# Per-episode conversion
# ---------------------------------------------------------------------------
def _agent_state_from_pose(position: np.ndarray, rot_mat: np.ndarray) -> habitat_sim.AgentState:
    s = habitat_sim.AgentState()
    s.position = np.asarray(position, dtype=np.float32)
    # 3x3 rotation matrix -> np.quaternion
    s.rotation = npq.from_rotation_matrix(np.asarray(rot_mat, dtype=np.float64))
    return s


def _frame_keys_sorted(frames_grp: h5py.Group) -> list[str]:
    return sorted(frames_grp.keys(), key=lambda k: int(k))


def _build_per_frame_poses(graph) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """frame_step -> (agent_position[3], agent_rotation[3,3]).

    Falls back to the first node found at that step; all nodes in the same
    frame share the same agent pose in the E3D graphs.
    """
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n, attr in graph.nodes(data=True):
        m = attr.get("map")
        if m is None:
            continue
        step = int(m[0])
        if step in out:
            continue
        pos = np.asarray(attr["agent_position"], dtype=np.float32)
        rot = np.asarray(attr["agent_rotation"], dtype=np.float32)
        out[step] = (pos, rot)
    return out


def _goal_class_and_instance(ep: h5py.Group, frame_keys: list[str]) -> tuple[str, int]:
    """Return (class_name, instance_id) for the goal object.

    Goal = the object whose ``e3d_distances`` reaches 0 in the last frame
    (closest object at the goal pose). Falls back to the first instance.
    """
    last = ep["frames"][frame_keys[-1]]
    inst_ids = last["instance_ids"][()]
    cls_ids = last["class_ids"][()]
    dists = last["e3d_distances"][()]
    if len(inst_ids) == 0:
        return "unknown", 0
    # pick instance with minimum (finite) distance
    finite = np.isfinite(dists)
    if finite.any():
        idx = int(np.argmin(np.where(finite, dists, np.inf)))
    else:
        idx = 0

    # MP3D mpcat40 names matching LangGeoNet trainer's MP3D_CATS
    from dataset import MP3D_CATS  # type: ignore  # imported lazily; see sys.path setup
    name = MP3D_CATS.get(int(cls_ids[idx]), f"cls{int(cls_ids[idx])}")
    return name, int(inst_ids[idx])


def _stamp_semantic(masks: np.ndarray, instance_ids: np.ndarray) -> np.ndarray:
    """Fold [K,H,W] uint8 masks into a single [H,W] int32 instance map.

    Earlier masks are overwritten by later ones on overlap.
    """
    K, H, W = masks.shape
    sem = np.zeros((H, W), dtype=np.int32)
    for k in range(K):
        sem[masks[k].astype(bool)] = int(instance_ids[k])
    return sem


def convert_episode(
    ep_id: str,
    ep: h5py.Group,
    scan: str,
    out_root: Path,
    write_images: bool = True,
    write_sem: bool = True,
    overwrite: bool = False,
) -> Path | None:
    frames_grp = ep["frames"]
    frame_keys = _frame_keys_sorted(frames_grp)
    if not frame_keys:
        return None

    # Goal class/instance
    try:
        cls_name, inst_id = _goal_class_and_instance(ep, frame_keys)
    except Exception:
        cls_name, inst_id = "unknown", 0
    cls_name = cls_name.replace(" ", "-").replace("/", "-") or "unknown"

    ep_dir_name = f"{scan}_{int(ep_id):07d}_{cls_name}_{inst_id}_"
    ep_dir = out_root / ep_dir_name

    sentinel = ep_dir / "agent_states.npy"
    if sentinel.exists() and not overwrite:
        return ep_dir

    (ep_dir / "images").mkdir(parents=True, exist_ok=True)
    if write_sem:
        (ep_dir / "images_sem").mkdir(parents=True, exist_ok=True)

    # Instruction
    instr = ep["instruction"][()]
    if isinstance(instr, (bytes, np.bytes_)):
        instr = instr.decode("utf-8", errors="replace")
    (ep_dir / "instruction.txt").write_text(instr or "")

    # Topometric graph (raw bytes from H5)
    raw = ep["graph"][()]
    if isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    (ep_dir / "nodes_gt_topometric.pickle").write_bytes(raw)

    # Per-frame poses from the unpickled graph
    graph = pickle.loads(raw)
    pose_by_step = _build_per_frame_poses(graph)

    agent_states: list[habitat_sim.AgentState] = []
    last_rgb = None

    for fkey in frame_keys:
        f = frames_grp[fkey]
        step = int(f["frame_idx"][()])

        # Pose lookup: prefer graph (per node), else carry forward last known
        if step in pose_by_step:
            pos, rot = pose_by_step[step]
        elif agent_states:
            prev = agent_states[-1]
            pos = np.asarray(prev.position, dtype=np.float32)
            rot = npq.as_rotation_matrix(prev.rotation).astype(np.float32)
        else:
            # No pose anywhere → identity at origin (rare)
            pos = np.zeros(3, dtype=np.float32)
            rot = np.eye(3, dtype=np.float32)

        agent_states.append(_agent_state_from_pose(pos, rot))

        if write_images:
            rgb = f["rgb"][()]                          # [H, W, 3] uint8
            last_rgb = rgb
            cv2.imwrite(str(ep_dir / "images" / f"{step:05d}.png"),
                        rgb[:, :, ::-1])                # RGB -> BGR

        if write_sem:
            masks = f["masks"][()]                      # [K, H, W] uint8
            inst_ids = f["instance_ids"][()]
            sem = _stamp_semantic(masks, inst_ids)
            np.save(ep_dir / "images_sem" / f"{step:05d}.npy", sem)

    np.save(ep_dir / "agent_states.npy",
            np.array(agent_states, dtype=object), allow_pickle=True)

    if write_images and last_rgb is not None:
        cv2.imwrite(str(ep_dir / "goalImg.png"), last_rgb[:, :, ::-1])

    _write_next_action_instructions(ep, ep_dir, frame_keys)
    return ep_dir


def _write_next_action_instructions(
    ep: h5py.Group,
    ep_dir: Path,
    frame_keys: list[str],
) -> None:
    """Write ``<ep_dir>/next_action_instructions.json``.

    Maps ``"seq_idx:03d"`` (0-indexed position in sorted frame list, matching
    the agent_states index) to the per-frame next_action_instruction string.
    Frames without a next_action_instruction are stored as empty strings.
    The LMDB consumed by the eval pipeline is built at main.py startup from
    this file together with instruction.txt.
    """
    import json as _json
    nai_map: dict[str, str] = {}
    for seq_idx, fkey in enumerate(frame_keys):
        frame_grp = ep["frames"][fkey]
        nai = ""
        if "next_action_instruction" in frame_grp:
            raw_nai = frame_grp["next_action_instruction"][()]
            nai = (raw_nai.decode("utf-8", errors="replace")
                   if isinstance(raw_nai, (bytes, np.bytes_)) else str(raw_nai))
        nai_map[f"{seq_idx:03d}"] = nai
    (ep_dir / "next_action_instructions.json").write_text(
        _json.dumps(nai_map, ensure_ascii=False, indent=None)
    )


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True, help="Path to e3d_*.h5")
    p.add_argument("--rxr-dir", required=True,
                   help="Directory with rxr_*_guide.jsonl[.gz] files")
    p.add_argument("--out-root", required=True,
                   help="Output root; episodes go under <out-root>/mp3d_iin_<split>/")
    p.add_argument("--split", default="test",
                   help="Subdir tag (default: test → mp3d_iin_test/)")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only convert this many episodes")
    p.add_argument("--no-images", action="store_true",
                   help="Skip writing per-frame RGB pngs")
    p.add_argument("--no-sem", action="store_true",
                   help="Skip writing per-frame semantic maps")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-convert episodes even if agent_states.npy exists")
    args = p.parse_args()

    # Allow `from dataset import MP3D_CATS` inside _goal_class_and_instance.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "train" / "lange3dnet_train"))

    h5_path = Path(args.h5)
    rxr_dir = Path(args.rxr_dir)
    out_root = Path(args.out_root) / f"mp3d_iin_{args.split}"
    out_root.mkdir(parents=True, exist_ok=True)

    h5 = h5py.File(str(h5_path), "r")
    ep_ids = list(h5.keys())
    if args.limit > 0:
        ep_ids = ep_ids[: args.limit]
    print(f"[build] {len(ep_ids)} episodes in {h5_path.name}")

    print("[build] looking up RXR scene mapping…")
    scene_map = build_rxr_scene_map(rxr_dir, set(ep_ids))
    missing = [e for e in ep_ids if e not in scene_map]
    print(f"[build] mapped {len(scene_map)}/{len(ep_ids)} episodes "
          f"({len(missing)} missing scan)")
    if missing[:5]:
        print(f"[build]   first missing: {missing[:5]}")

    n_ok, n_skip, n_fail = 0, 0, 0
    for i, ep_id in enumerate(ep_ids):
        scan = scene_map.get(ep_id)
        if scan is None:
            n_skip += 1
            continue
        try:
            ep_dir = convert_episode(
                ep_id=ep_id,
                ep=h5[ep_id],
                scan=scan,
                out_root=out_root,
                write_images=not args.no_images,
                write_sem=not args.no_sem,
                overwrite=args.overwrite,
            )
            if ep_dir is None:
                n_skip += 1
            else:
                n_ok += 1
                if i % 5 == 0 or i == len(ep_ids) - 1:
                    print(f"[build]  ({i+1}/{len(ep_ids)}) {ep_dir.name}")
        except Exception as e:
            n_fail += 1
            print(f"[build]  ! {ep_id} ({scan}) failed: {e!r}")

    h5.close()
    print(f"[build] done: ok={n_ok}, skipped={n_skip}, failed={n_fail}")
    print(f"[build] output: {out_root}")


if __name__ == "__main__":
    main()
