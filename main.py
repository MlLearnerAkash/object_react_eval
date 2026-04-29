def seed_everything(seed: int = 42):
    import random
    import numpy as np
    import torch
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()

import os
import sys
import torch
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import logging
import yaml
import traceback

from libs.logger import default_logger

if "LOG_LEVEL" not in os.environ:
    os.environ["LOG_LEVEL"] = "DEBUG"
default_logger.setup_logging(level=os.environ["LOG_LEVEL"])

from libs.common import utils
from libs.experiments import task_setup
from libs.experiments.task_setup import UnreachableGoalError


def _filter_episodes_by_h5(episodes, h5_file):
    """Keep only episodes whose numeric ID appears as a key in *h5_file*.

    Episode dir names follow the pattern ``<scene>_<zero-padded-id>_<cat>_<inst>_``
    (e.g. ``17DRP5sb8fy_0011458_sink_216_``).  The numeric id field is
    ``split('_')[1]`` which may have leading zeros; H5 keys are stripped
    integers (e.g. ``"11458"``).  Episodes absent from the H5 are dropped
    because they would run with an empty instruction string.
    """
    if h5_file is None:
        return episodes
    h5_keys = set(h5_file.keys())
    kept, dropped = [], []
    for p in episodes:
        parts = p.name.split("_")
        # Robust: strip leading zeros via int() conversion
        ep_id = str(int(parts[1])) if len(parts) > 1 and parts[1].isdigit() else p.name
        if ep_id in h5_keys:
            kept.append(p)
        else:
            dropped.append(p.name)
    if dropped:
        print(
            f"[lang_e3d] H5 filter: dropped {len(dropped)} episodes with no instruction "
            f"(not in H5): {dropped[:5]}{'...' if len(dropped) > 5 else ''}"
        )
    print(f"[lang_e3d] H5 filter: {len(kept)}/{len(kept) + len(dropped)} episodes have H5 data")
    return kept


def run(args):
    # # set up all the paths
    path_dataset = Path(args.path_dataset)

    # Scene dataset selection: hm3d (default) or mp3d. For mp3d the scenes live
    # in <scene_root>/<scan>/<scan>.glb instead of HM3D's *<id>_dir/ basis.glb.
    scene_dataset = getattr(args, "scene_dataset", "hm3d").lower()
    explicit_scenes_root = getattr(args, "path_scenes_root", "") or ""
    if explicit_scenes_root:
        path_scenes_root_hm3d = Path(explicit_scenes_root)
    elif scene_dataset == "mp3d":
        # default MP3D location used in this workspace
        path_scenes_root_hm3d = Path(
            "/data/dataset/RXR/dataset/mp3d/data/scene_datasets/mp3d/v1/tasks/mp3d_habitat/mp3d"
        )
    else:
        path_scenes_root_hm3d = path_dataset / "hm3d_v0.2" / args.split

    sh_map = args.sim["sensor_height_map"]
    if args.task_type == "via_alt_goal":
        map_dir = f"hm3d_generated/stretch_maps/hm3d_iin_{args.split}/maps_via_alt_goal"
        if sh_map != 1.31:
            map_dir += f"-sh_{sh_map}/"
    else:
        if sh_map != 1.31:
            map_dir = (
                f"hm3d_generated/stretch_maps/hm3d_iin_{args.split}/height-sh_{sh_map}"
            )
        else:
            map_dir = f"hm3d_iin_{args.split}"

    explicit_episodes_root = getattr(args, "path_episodes_root", "") or ""
    if explicit_episodes_root:
        path_episode_root = Path(explicit_episodes_root)
    else:
        path_episode_root = path_dataset / map_dir
    print(f"Root path for episodes: {path_episode_root}")
    print(f"Scene dataset: {scene_dataset}, scenes root: {path_scenes_root_hm3d}")

    # Results tracking
    results_summary = {
        "total_episodes": 0,
        "successful_episodes": 0,
        "failed_episodes": 0,
        "skipped_unreachable": 0,   # NavMesh-disconnected episodes
        "success_rate": 0.0,
        "failure_reasons": {},
    }
    if args.log_wandb:
        task_setup.setup_wandb_logging(args)

    path_results_folder = task_setup.init_results_dir_and_save_cfg(args, default_logger)
    print("\nConfig file saved in the results folder!\n")

    preload_data = task_setup.preload_models(args)

    episodes = task_setup.load_run_list(args, path_episode_root)[
        args.start_idx : args.end_idx : args.step_idx
    ]

    # For lang_e3d: build a per-run instruction LMDB from episode folders.
    # Reads instruction.txt (episodic) and next_action_instructions.json
    # (per-frame NAI, written by build_mp3d_iin_from_h5.py) for every episode,
    # then stores both in a single LMDB under path_results_folder.
    if args.goal_source == "lang_e3d":
        task_setup.build_run_instructions_lmdb(episodes, path_results_folder, preload_data)

    if len(episodes) == 0:
        raise ValueError(
            f"No episodes found at {path_episode_root=}. Please check the dataset path and indices."
        )
    print(f"Total episodes to process: {len(episodes)}")

    for ei, path_episode in tqdm(
        enumerate(episodes),
        total=len(episodes),
        desc=f"Processing Episodes (Total: {len(episodes)})",
    ):
        results_summary["total_episodes"] += 1
        episode_name = path_episode.parts[-1].split("_")[0]
        if scene_dataset == "mp3d":
            # MP3D layout: <scenes_root>/<scan>/<scan>.glb
            scene_name_hm3d = str(path_scenes_root_hm3d / episode_name / f"{episode_name}.glb")
        else:
            path_scene_hm3d = sorted(path_scenes_root_hm3d.glob(f"*{episode_name}"))[0]
            scene_name_hm3d = str(sorted(path_scene_hm3d.glob("*basis.glb"))[0])

        episode_runner = None
        success_status = None
        try:
            episode_runner = task_setup.Episode(
                args, path_episode, scene_name_hm3d, path_results_folder, preload_data
            )

            if args.plot:
                ax, plt = episode_runner.init_plotting()

            for step in range(args.max_steps):
                if episode_runner.is_done():
                    break

                logger.info(f"\tAt step {step} (ep {ei}): Getting sensor observations")
                observations = episode_runner.sim.get_sensor_observations()
                display_img, depth, semantic_instance_sim = utils.split_observations(
                    observations
                )

                if args.infer_depth:
                    depth = (
                        preload_data["depth_model"].infer(display_img) * 0.44
                    )  # is a scaling factor

                logger.info(f"\tAt step {step} (ep {ei}): Getting goal")
                episode_runner.get_goal(display_img, depth, semantic_instance_sim)

                if not args.infer_traversable:  # override the FastSAM traversable mask
                    episode_runner.traversable_mask = utils.get_traversibility(
                        torch.from_numpy(semantic_instance_sim),
                        episode_runner.traversable_class_indices,
                    ).numpy()

                logger.info(f"\tAt step {step} (ep {ei}): Getting control signal")
                episode_runner.get_control_signal(step, display_img, depth)

                logger.info(f"\tAt step {step} (ep {ei}): Executing Action")
                episode_runner.execute_action()

                if args.plot:
                    episode_runner.plot(
                        ax, plt, step, display_img, depth, semantic_instance_sim
                    )

                if args.log_robot:
                    episode_runner.log_results(step)

                print(f"...Steps completed: {step + 1}/{args.max_steps}", end="\r")

        except UnreachableGoalError as e:
            # Clean skip — goal is on a disconnected NavMesh island.
            # episode_runner.close() was already called inside ready_agent().
            success_status = "unreachable_goal"
            logger.warning(str(e))
            if args.except_exit:
                exit(-1)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            e_filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            success_status = (
                f"{type(e).__name__}: {e} in {e_filename} #{exc_tb.tb_lineno}"
            )

            if episode_runner is not None:
                episode_runner.success_status = success_status

            if args.except_exit:
                traceback.print_exc()
                if episode_runner:
                    episode_runner.close(step)
                exit(-1)

        if episode_runner is not None and success_status != "unreachable_goal":
            episode_runner.close(step)  # need to close vis first to save video to wandb
            if args.log_wandb:
                task_setup.wandb_log_episode(
                    path_episode.name,
                    episode_runner.results_dict,
                    episode_runner.video_cfg["savepath"],
                )
            success_status = episode_runner.success_status

        # Track success and failure statistics
        if success_status == "unreachable_goal":
            results_summary["skipped_unreachable"] += 1
        elif success_status is None or "success" in success_status.lower():
            results_summary["successful_episodes"] += 1
        else:
            results_summary["failed_episodes"] += 1
            if success_status not in results_summary["failure_reasons"]:
                results_summary["failure_reasons"][success_status] = 0
            results_summary["failure_reasons"][success_status] += 1

        print(f"Completed with success status: {success_status}")

    results_summary = utils.create_results_summary(
        args, results_summary, path_results_folder
    )

    return results_summary


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        "-c",
        help="Path to the config file",
        default="configs/defaults.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger("[Goal Control]")  # Logger for this script

    args = parse_args()

    config_file = args.config_file
    if not os.path.exists(config_file):
        logger.warning(
            f"Using default config file, create {config_file} to customise the parameters"
        )
        config_file = "defaults.yaml"

    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            logger.info(f"Config File {config_file} params: {config}")
            # pass the config to the args
            for k, v in config.items():
                setattr(args, k, v)

    # unsupported combinations
    if args.reverse and args.task_type != "original":
        raise ValueError("Reverse is only supported for original task type")

    # setup traversable classes for TANGO
    setattr(
        args,
        "traversable_class_names",
        [
            "floor",
            "flooring",
            "floor mat",
            "floor vent",
            "carpet",
            "mat",
            "rug",
            "doormat",
            "shower floor",
            "pavement",
            "ground",
            "tiles",
            # "door",
            # "wall",
            # "table",
            # "tv_monitor",
            # "sink",
            # "cushion",
            # "chair",
            # "window",
            # "towel",
            # "plant",
            # "shelving",
            # "curtain",
            # "seating",
            # "picture",
            # "bed",
            # "couch",
            # "stair",
            # "staircase",
            # "window"
        ],
    )

    run(args)
