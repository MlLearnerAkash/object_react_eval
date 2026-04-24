import sys
import re
import pickle
import argparse
import yaml
import numpy as np
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm

episode_names_all = natsorted(Path("./data/mp3d_iin_test/").iterdir())

parser = argparse.ArgumentParser()
parser.add_argument("base_dir", help="Path to directory containing results.")
parser.add_argument("--test-one-third", action="store_true")
parser.add_argument(
    "--h5_path", default=None,
    help="Optional path to the H5 dataset used for lang_e3d evaluation. "
         "When provided, NDTW is computed using GT trajectories from the H5 graph.")
args = parser.parse_args()

args.base_dir = Path(args.base_dir)

# ── NDTW helpers ──────────────────────────────────────────────────────────────

def _dtw_distance(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """Compute DTW distance between two 2-D paths ([N,2] and [M,2])."""
    n, m = len(path_a), len(path_b)
    if n == 0 or m == 0:
        return float("inf")
    # pairwise Euclidean distances
    diff = path_a[:, None, :] - path_b[None, :, :]      # [N, M, 2]
    dist = np.linalg.norm(diff, axis=-1)                 # [N, M]
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = dist[i - 1, j - 1] + min(dp[i - 1, j],
                                                  dp[i, j - 1],
                                                  dp[i - 1, j - 1])
    return float(dp[n, m])


def _load_gt_xz_path(h5_file, ep_id: str):
    """Return GT (x, z) positions in frame order from the H5 graph.

    Installs the ``_magnum`` mock if needed, then unpickles the graph stored
    under ``h5_file[ep_id]['graph']``.
    """
    import sys, types

    # Install mock _magnum once so Habitat C++ types unpickle silently.
    if "_magnum" not in sys.modules:
        class _MB:
            def __init__(self, *a, **k): pass
            def __setstate__(self, s): self._s = s
        mod = types.ModuleType("_magnum")
        for _n in ("Vector3", "Vector4", "Matrix4", "Matrix3", "Quaternion",
                   "Rad", "Deg", "Range3D", "Range2D"):
            setattr(mod, _n, type(_n, (_MB,), {}))
        sys.modules["_magnum"] = mod

    if ep_id not in h5_file:
        return None
    graph_bytes = bytes(h5_file[ep_id]["graph"][()])
    G = pickle.loads(graph_bytes)
    nodes_data = dict(G.nodes(data=True))
    # Build an ordered list of agent positions, one per unique frame step.
    step_to_pos = {}
    for nid, nd in nodes_data.items():
        step = nd["map"][0]
        if step not in step_to_pos:
            pos = np.asarray(nd["agent_position"], dtype=np.float64)
            step_to_pos[step] = pos
    if not step_to_pos:
        return None
    ordered = [step_to_pos[s] for s in sorted(step_to_pos)]
    return np.array([[p[0], p[2]] for p in ordered])   # (x, z) only


# ── H5 file handle (opened once) ──────────────────────────────────────────────
_h5_file = None
if args.h5_path is not None:
    import h5py
    _h5_file = h5py.File(args.h5_path, "r")

with open("configs/defaults.yaml", "r") as f:
    blacklists = yaml.safe_load(f)["episode_blacklists"]

# find subdirs recursively which have timestamp in their name in format 20241212-13-40-50_*
pattern = re.compile(r"^\d{8}-\d{2}-\d{2}-\d{2}_.+")

if pattern.match(args.base_dir.name):
    results_dirs = [args.base_dir]
else:
    results_dirs = [
        subdir
        for subdir in args.base_dir.rglob(
            "*"
        )  # Recursively iterate through all entries
        if subdir.is_dir()
        and pattern.match(
            subdir.name
        )  # Check if it's a directory and matches the pattern
    ]
    print(f"Found {len(results_dirs)=}")
    results_dirs = natsorted(results_dirs)
report_collisions = False
compute_spl = True
compute_soft_spl = True
verbose = True

paper_results = {}

loopCount = 0
for results_dir in tqdm(results_dirs):
    episode_dirs = natsorted(
        [d for d in results_dir.iterdir() if d.is_dir() and "summary" not in d.name]
    )
    if args.test_one_third:
        episode_dirs = episode_dirs[::3]
    if len(episode_dirs) == 0:
        continue
    method_type = results_dir.parents[2].stem
    if "" not in str(results_dir):
        continue

    if verbose:
        print(f"\nProcessing {str(results_dir)} with {len(episode_dirs)} episodes")
    task_type = results_dir.parents[3].stem

    episode_blacklist = [*blacklists["all_tasks"], *blacklists.get(task_type)]

    if verbose:
        print(f"{task_type=}")
        print(f"{episode_blacklist=}")

    episode_identifiers = [ed.stem for ed in episode_names_all]
    num_success, num_exceeded, num_errors, num_no_status, num_ignored = 0, 0, 0, 0, 0
    avg_collisions_list, spl_list, soft_spl_list, ndtw_list = [], [], [], []
    for ei, ed in enumerate(episode_dirs):
        episode_identifier = ed.name.split("__")[0] + "_"
        if episode_identifier in episode_identifiers:
            episode_identifiers.remove(episode_identifier)
        if episode_identifier in episode_blacklist:
            num_ignored += 1
            continue
        metadata_filename = ed / "metadata.txt"
        metadata = metadata_filename.read_text().splitlines()
        metadata_vals = [m.split(":")[-1] for m in metadata]
        metadata_dict = {
            m.split("=")[0]: m.split("=")[1] for m in metadata_vals if "=" in m
        }
        if "success_status" not in metadata_dict:
            num_no_status += 1
            continue

        shortest_path_length = float(metadata_dict["distance_to_final_goal_from_start"])
        remain_distance = float(metadata_dict["final_distance"])

        if compute_soft_spl:
            soft_spl = max(0, (1 - remain_distance / (shortest_path_length+1e-8)))
            soft_spl_list.append(soft_spl)

        # ── NDTW (requires H5 GT path) ────────────────────────────────────────
        if _h5_file is not None:
            # Episode dir name: "ep_10014_learnt_lang_e3d" → id "10014"
            ep_parts = ed.name.split("_")
            ep_id = ep_parts[1] if len(ep_parts) > 1 else ed.name
            gt_xz = _load_gt_xz_path(_h5_file, ep_id)
            results_csv_filename = ed / "results.csv"
            if gt_xz is not None and results_csv_filename.exists():
                results_csv = results_csv_filename.read_text().splitlines()
                pred_x = [float(r.split(",")[1]) for r in results_csv[1:]]
                pred_z = [float(r.split(",")[3]) for r in results_csv[1:]]
                pred_xz = np.array(list(zip(pred_x, pred_z)))
                if len(pred_xz) > 0 and len(gt_xz) > 0:
                    gt_path_len = float(
                        np.linalg.norm(gt_xz[1:] - gt_xz[:-1], axis=1).sum())
                    dtw_dist = _dtw_distance(pred_xz, gt_xz)
                    ndtw = float(np.exp(-dtw_dist / (gt_path_len + 1e-6)))
                    ndtw_list.append(ndtw)

        success_status = metadata_dict[
            "success_status"
        ]  # [m for m in metadata if 'success_status' in m]
        if len(success_status) == "":
            # print("Unknown status", success_status)
            pass
        elif success_status == "success":
            # print(f"Episode {ei} [{ed.name}]:", success_status)
            num_success += 1
            if report_collisions or compute_spl:

                results_csv_filename = ed / "results.csv"
                results_csv = results_csv_filename.read_text().splitlines()

                if report_collisions:
                    collisions = [int(r.split(",")[-1]) for r in results_csv[1:]]
                    collisions_num = np.mean(collisions)

                if compute_spl:
                    x_pos = [float(r.split(",")[1]) for r in results_csv[1:]]
                    z_pos = [float(r.split(",")[3]) for r in results_csv[1:]]
                    xz = np.array(list(zip(x_pos, z_pos)))
                    if len(xz)> 0:
                        path_length = (
                            np.linalg.norm(xz[1:] - xz[:-1], axis=1).sum() + remain_distance
                        )
                    else:
                        path_length = remain_distance

            if report_collisions:
                avg_collisions_list.append(collisions_num)
            if compute_spl:
                spl = path_length / max((shortest_path_length, path_length, 1))
                spl_list.append(spl)
            if compute_soft_spl:
                soft_spl_list.pop()
                soft_spl_list.append(1.0)
        elif success_status == "exceeded_steps":
            num_exceeded += 1
        elif success_status != "":
            num_errors += 1
            if verbose:
                print(f"Episode {ei} [{ed.name}]:", success_status)
        else:
            raise ValueError(f"Unknown success_status: {success_status}")

    if len(episode_dirs) == num_ignored:
        print("WARNING: Run only contains ignored episodes")
        continue

    denom = len(episode_dirs) - num_ignored  # len(episode_names_ignore)

    if verbose:
        print(f"[{num_success/denom*100:.2f}%] {num_success=} of {denom} episodes")
        print(f"[{num_exceeded/denom*100:.2f}%] {num_exceeded=} of {denom} episodes")
        print(f"[{num_errors/denom*100:.2f}%] {num_errors=} of {denom} episodes")
        print(f"[{num_no_status/denom*100:.2f}%] {num_no_status=} of {denom} episodes")
        print(f"Num Missing episodes: {len(episode_identifiers)}")
        print(f"Num Ignored episodes: {num_ignored}")
        if report_collisions:
            print(
                f"MeanAvg Collisions of Success Runs: {np.mean(avg_collisions_list):.2f}"
            )
        if compute_spl:
            print(f"Mean SPL of Success Runs: {np.sum(spl_list)/denom*100:.2f}")
        if compute_soft_spl:
            print(f"Mean Soft SPL: {np.sum(soft_spl_list)/denom*100:.2f}")
        if ndtw_list:
            print(f"Mean NDTW ({len(ndtw_list)} eps): {np.mean(ndtw_list)*100:.2f}")

    if task_type not in paper_results:
        paper_results[task_type] = {}
    result_entry = {
        "success_rate": num_success / denom * 100,
        "soft_spl": np.sum(soft_spl_list) / denom * 100,
        "spl": np.sum(spl_list) / denom * 100,
    }
    if ndtw_list:
        result_entry["ndtw"] = float(np.mean(ndtw_list) * 100)
    paper_results[task_type].update({method_type: result_entry})
    loopCount += 1
