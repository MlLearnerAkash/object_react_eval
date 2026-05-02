import os
import json
import math
import numpy as np
from pathlib import Path


class UnreachableGoalError(RuntimeError):
    """Raised when start and goal positions are on disconnected NavMesh islands.

    This typically means the episode was built with a goal that cannot be
    reached via any navigable path from the agent's start position
    (geodesic_distance == inf).  The episode should be skipped rather than
    run for the full step budget.
    """
import pickle
import yaml
import torch
import cv2
from datetime import datetime
import shutil
from natsort import natsorted
import wandb
import networkx as nx

import habitat_sim

import logging

logger = logging.getLogger(
    "[Task Setup]"
)  # logger level is explicitly set below by LOG_LEVEL
from libs.logger.level import LOG_LEVEL

logger.setLevel(LOG_LEVEL)

from libs.goal_generator import goal_gen
from libs.experiments import model_loader
from libs.control.robohop import control_with_mask

from libs.common import utils_data
from libs.common import utils_visualize as utils_viz
from libs.common import utils_goals
from libs.common import utils
from libs.common import utils_sim_traj as ust
from libs.logger.visualizer import Visualizer


class Episode:
    def __init__(
        self, args, path_episode, scene_name_hm3d, path_results_folder, preload_data={}
    ):
        if args is None:
            args = utils.get_default_args()
        self.args = args
        self.steps = 0  # only used when running real in remote mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path_episode = path_episode
        logger.info(f"Running {self.path_episode=}...")
        self.scene_name_hm3d = scene_name_hm3d
        self.preload_data = preload_data

        self.init_controller_params()
        self.agent_states = None
        self.final_goal_position = None
        self.traversable_class_indices = None

        # Resolve episode numeric ID (strips 7-digit zero-padding: "0011458" → "11458")
        _parts = path_episode.name.split("_")
        _raw_id = _parts[1] if len(_parts) > 1 else path_episode.name
        self.ep_num = str(int(_raw_id)) if _raw_id.isdigit() else _raw_id

        self.instruction_type = getattr(self.args, "instruction_type", "episodic")

        # Episodic instruction: prefer instruction.txt in the episode folder
        # (written by build_mp3d_iin_from_h5.py), fall back to H5 file.
        self.episode_instruction = ""
        _instr_txt = path_episode / "instruction.txt"
        if _instr_txt.exists():
            self.episode_instruction = _instr_txt.read_text().strip()
            logger.info(f"[ep {self.ep_num}] episodic instruction (from txt): "
                        f"{self.episode_instruction[:80]}…")
        elif self.args.goal_source.lower() == "lang_e3d":
            h5_file = self.preload_data.get("h5_file")
            if h5_file is not None and self.ep_num in h5_file:
                self.episode_instruction = (
                    h5_file[self.ep_num]["instruction"][()].decode("utf-8")
                )
                logger.info(f"[ep {self.ep_num}] episodic instruction (from H5): "
                            f"{self.episode_instruction[:80]}…")
            else:
                logger.warning(f"[ep {self.ep_num}] no instruction.txt and not in H5; "
                               "using empty instruction")

        # Run-level instruction LMDB (built by build_run_instructions_lmdb in run())
        self.instr_lmdb = self.preload_data.get("instr_lmdb")

        if self.args.log_robot:
            episode_results_dir = f"{self.path_episode.parts[-1]}_{self.args.method.lower()}_{self.args.goal_source}"
            self.path_episode_results = path_results_folder / episode_results_dir
            self.path_episode_results.mkdir(exist_ok=True, parents=True)
            self.set_logging()

        if args.env == "sim":
            if not (self.path_episode / "agent_states.npy").exists():
                raise FileNotFoundError(
                    f'{self.path_episode / "agent_states.npy"} does not exist...'
                )

        self.num_map_images = len(os.listdir(self.path_episode / "images"))
        self.cull_categories = [
            "floor",
            "floor mat",
            "floor vent",
            "carpet",
            "rug",
            "doormat",
            "shower floor",
            "pavement",
            "ground",
            "ceiling",
            "ceiling lower",
        ]

        # map params
        self.get_map_graph_path()
        self.graph_instance_ids, self.graph_path_lengths = None, None
        self.final_goal_mask_vis_sim = None

        # experiment params
        self.success_status = "exceeded_steps"
        self.distance_to_goal = np.nan
        self.step_real_complete = True

        self.image_height, self.image_width = (
            self.args.sim["height"],
            self.args.sim["width"],
        )
        self.sim, self.agent, self.distance_to_final_goal = None, None, np.nan
        if args.env == "sim":
            self.setup_sim_agent()
            self.ready_agent()
        self.load_map_graph()
        self.set_goal_generator()

        self.set_controller()

        # setup visualizer
        self.vis_img_default = np.zeros(
            (self.image_height, self.image_width, 3)
        ).astype(np.uint8)
        self.vis_img = self.vis_img_default.copy()
        self.video_cfg = {
            "savepath": str(self.path_episode_results / "repeat.mp4"),
            "codec": "mp4v",
            "fps": 6,
        }
        self.vis = Visualizer(
            self.sim, self.agent, self.scene_name_hm3d, env=self.args.env
        )
        if self.args.env == "sim":
            self.vis.draw_teach_run(self.agent_states)

            # marker so the BEV start matches the real spawn location.
            if not self.args.reverse:
                self.vis.draw_start(self.vis.sim_to_tdv(self.start_position))

            if self.args.task_type in ["alt_goal"]:
                self.vis.draw_goal(self.vis.sim_to_tdv(self.final_goal_position))

            if self.args.reverse:
                self.vis.draw_goal(self.vis.sim_to_tdv(self.final_goal_position))
                self.vis.draw_start(self.vis.sim_to_tdv(self.start_position))

    def init_controller_params(self):
        # data params
        # TODO: multiple hfov variables
        is_robohop_tango = np.isin(["robohop", "tango"], self.args.method.lower()).any()
        self.fov_deg = self.args.sim["hfov"] if is_robohop_tango else 79
        self.hfov_radians = np.pi * self.fov_deg / 180

        # controller params
        self.time_delta = 0.1
        self.theta_control = np.nan
        self.velocity_control = 0.05 if is_robohop_tango else np.nan
        self.pid_steer_values = (
            [0.25, 0, 0] if self.args.method.lower() == "tango" else []
        )
        self.discrete_action = -1
        self.controller_logs = None

    def setup_sim_agent(self) -> tuple:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

        # get the scene
        update_nav_mesh = False
        args_sim = self.args.sim
        self.sim, self.agent, vel_control = utils.get_sim_agent(
            self.scene_name_hm3d,
            update_nav_mesh,
            width=args_sim["width"],
            height=args_sim["height"],
            hfov=args_sim["hfov"],
            sensor_height=args_sim["sensor_height"],
        )
        self.sim.agents[0].agent_config.sensor_specifications[1].normalize_depth = True

        # create and configure a new VelocityControl structure
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.controlling_ang_vel = True
        vel_control.ang_vel_is_local = True
        self.vel_control = vel_control

        (
            self.traversable_class_indices,
            self.bad_goal_classes,
            self.cull_instance_ids,
        ) = get_semantic_filters(
            self.sim, self.args.traversable_class_names, self.cull_categories
        )

    def ready_agent(self):
        # get the initial agent state for this episode (i.e. the starting pose)
        path_agent_states = self.path_episode / "agent_states.npy"
        self.agent_states = np.load(str(path_agent_states), allow_pickle=True)

        self.agent_positions_in_map = np.array([s.position for s in self.agent_states])

        # set the final goal state for this episode
        self.final_goal_state = None
        self.final_goal_position = None
        self.final_goal_image_idx = None
        if self.args.reverse:
            self.final_goal_state = utils_data.get_final_goal_state_reverse(
                self.sim, self.agent_states
            )
            self.final_goal_position = self.final_goal_state.position

            # set default, as it is not well defined for the reverse setting
            self.final_goal_image_idx = len(self.agent_states) - 1

        elif self.args.task_type in ["alt_goal"]:
            self.final_goal_image_idx, _, goal_instance_id = (
                utils_data.get_goal_info_alt_goal(
                    self.path_episode, self.args.task_type
                )
            )
            instance_position = None
            for instance in self.sim.semantic_scene.objects:
                if instance.semantic_id == goal_instance_id:
                    instance_position = instance.aabb.center
                    break
            if instance_position is None:
                raise ValueError("Could not obtain goal instance...")
            avg_floor_height = self.agent_positions_in_map[:, 1].mean()
            instance_position[1] = avg_floor_height
            self.final_goal_position = self.sim.pathfinder.snap_point(instance_position)
            self.agent_positions_in_map = self.agent_positions_in_map[
                : self.final_goal_image_idx + 1
            ]

        else:
            self.final_goal_position = self.agent_states[-1].position
            self.final_goal_image_idx = len(self.agent_states) - 1

        # set the start state and set the agent to this pose
        start_state = select_starting_state(
            self.sim, self.args, self.agent_states, self.final_goal_position
        )
        if start_state is None:
            self.success_status = (
                f"Could not find a valid start state for {self.path_episode}"
            )
            self.close()
            raise ValueError(self.success_status)
        self.agent.set_state(start_state)  # set robot to this pose
        self.start_position = start_state.position

        # define measure of success
        self.distance_to_final_goal = ust.find_shortest_path(
            self.sim, p1=self.start_position, p2=self.final_goal_position
        )[0]

        # Guardrail: skip episodes where the goal is on a disconnected NavMesh
        # island — the agent can never reach it regardless of how long it runs.
        if math.isinf(self.distance_to_final_goal):
            self.success_status = "unreachable_goal"
            self.close()
            raise UnreachableGoalError(
                f"Goal is unreachable from start (geodesic_distance=inf) "
                f"for episode {self.path_episode.name}. "
                f"Start: {self.start_position}, Goal: {self.final_goal_position}. "
                f"Skipping — start and goal are on disconnected NavMesh islands."
            )


    def get_map_graph_path(self):
        self.path_graph = None
        goal_source = self.args.goal_source.lower()
        if goal_source == "gt_metric":
            pass
        elif goal_source == "lang_e3d":
            pass  # No map graph needed; instruction drives the goal
        elif goal_source in ["gt_topological", "topological", "gt_topometric"]:
            # load robohop graph
            graph_filename = None
            if self.args.graph_filename is not None:
                graph_filename = self.args.graph_filename
            elif goal_source == "topological":
                suffix_str_depth = ""
                if self.args.goal_gen["edge_weight_str"] in [
                    "e3d_max",
                    "e3d_avg",
                    "e3d_min",
                ]:
                    suffix_str_depth = f"_depth_inferred"
                graph_filename = f'nodes_{self.args.goal_gen["map_segmentor_name"]}_{self.args.goal_gen["map_matcher_name"]}{suffix_str_depth}.pickle'
            elif goal_source == "gt_topological":
                graph_filename = "nodes_gt_topological.pickle"
            elif goal_source == "gt_topometric":
                graph_filename = "nodes_gt_topometric.pickle"

            self.path_graph = self.path_episode / graph_filename
            if not self.path_graph.exists():
                raise FileNotFoundError(f"{self.path_graph} does not exist...")

    def load_map_graph(self):
        self.map_graph = None
        if self.path_graph is not None:
            logger.info(f"Loading graph: {self.path_graph}")
            map_graph = pickle.load(open(str(self.path_graph), "rb"))
            map_graph = utils.change_edge_attr(map_graph)
            self.map_graph = map_graph

            if self.args.goal_source == "topological":
                goalNodeIdx = None
                if not self.args.goal_gen["goalNodeIdx"] or self.args.env == "sim":
                    goalNodeIdx = self.get_task_goalNodeIdx()

                if self.args.cull_map_instances:
                    assert not self.args.goal_gen["rewrite_graph_with_allPathLengths"]
                    save_path = (
                        self.path_episode
                        / f"images_cull_mask_{self.args.cull_map_method}"
                    )
                    save_path.mkdir(exist_ok=True, parents=True)
                    self.map_graph = self.cull_map_instances(goalNodeIdx, save_path)

    def get_task_goalNodeIdx(self):
        self.goal_object_id, goalNodeIdx = None, None

        if not self.args.reverse:
            nodeID_to_imgRegionIdx = np.array(
                [self.map_graph.nodes[node]["map"] for node in self.map_graph.nodes()]
            )
            goalNodeIdx, self.final_goal_mask_vis_sim = utils_data.get_goalNodeIdx(
                str(self.path_episode),
                self.map_graph,
                nodeID_to_imgRegionIdx,
                self.args.task_type,
                ret_final_goalMask_vis=True,
            )

            if self.final_goal_mask_vis_sim is not None:
                cv2.imwrite(
                    str(self.path_episode_results / "final_goal_mask_vis_sim.jpg"),
                    self.final_goal_mask_vis_sim,
                )

        # reverse mode goalNodeIdx 'inferring' only for sim
        elif "sim" in self.args.env:
            reverse_goal_path = f"{self.path_episode}/reverse_goal.npy"
            if os.path.exists(reverse_goal_path):
                self.goal_object_id = np.load(reverse_goal_path, allow_pickle=True)[()][
                    "instance_id"
                ]
                goalNodeIdx = utils_data.get_goalNodeIdx_reverse(
                    str(self.path_episode), self.map_graph, self.goal_object_id
                )

        if goalNodeIdx is None:
            self.success_status = f"Could not find goalNodeIdx for {self.path_episode}"
            self.close()
            raise ValueError(self.success_status)

        self.args.goal_gen.update({"goalNodeIdx": goalNodeIdx})

        return goalNodeIdx

    def cull_map_instances(self, goalNodeIdx, save_cull_mask_path=None):
        map_graph = self.map_graph
        cull_categories = ["floor", "ceiling"]
        method = self.args.cull_map_method

        if method == "fast_sam":
            img_dir = self.path_episode / "images"
            if self.args.segmentor == "fast_sam":
                fast_sam = self.preload_data["segmentor"]
            else:
                fast_sam = model_loader.get_segmentor(
                    "fast_sam", self.image_width, self.image_height, device="cuda"
                )
        else:
            img_dir = self.path_episode / "images_sem"

        img_paths = natsorted(img_dir.iterdir())
        assert len(img_paths) > 0, f"No images found in {img_dir}"

        cull_inds = []
        for si, img_path in enumerate(img_paths):

            cull_mask_path_i = str(save_cull_mask_path / f"{si:04d}.jpg")
            # Temporarily disable loading cull masks from disk
            if 0:  # os.path.exists(cull_mask_path_i):
                cull_mask = cv2.imread(cull_mask_path_i, cv2.IMREAD_GRAYSCALE).astype(
                    bool
                )
            else:
                if method == "fast_sam":
                    img = cv2.imread(str(img_path))[:, :, ::-1]
                    cull_mask = fast_sam.segment(
                        img,
                        retMaskAsDict=False,
                        textLabels=cull_categories,
                        textCulls=False,
                    )[0]
                    if cull_mask is None:
                        cull_mask = fast_sam.no_mask.copy()
                        logger.info(f"cull_mask is None for image idx {si}")
                        continue
                    cull_mask = cull_mask.sum(0).astype(bool)
                else:
                    sem_instance = np.load(str(img_path), allow_pickle=True)
                    cull_mask = np.sum(
                        sem_instance[None, ...]
                        == self.cull_instance_ids[:, None, None],
                        axis=0,
                    ).astype(bool)

                if save_cull_mask_path is not None:
                    cv2.imwrite(cull_mask_path_i, cull_mask.astype(np.uint8) * 255)

            areaThresh = np.ceil(0.001 * cull_mask.shape[0] * cull_mask.shape[1])

            nodeInds = np.array(
                [
                    n
                    for n in map_graph.nodes
                    if map_graph.nodes[n]["map"][0] == si and n != goalNodeIdx
                ]
            )

            local_cull_count = 0
            for n in nodeInds:
                graph_sem_instance = utils.rle_to_mask(
                    map_graph.nodes[n]["segmentation"]
                )
                if graph_sem_instance[cull_mask].sum() >= areaThresh:
                    cull_inds.append(n)
                    local_cull_count += 1
            logger.info(
                f"{local_cull_count}/{len(nodeInds)} nodes to cull for image idx {si}"
            )
        logger.info(
            f"Before culling: {map_graph.number_of_nodes()=}, {map_graph.number_of_edges()=}"
        )
        edges = np.concatenate([list(map_graph.edges(n)) for n in cull_inds]).tolist()
        logger.info(f"{len(edges)} edges to remove")
        map_graph.remove_edges_from(edges)
        logger.info(
            f"After culling: {map_graph.number_of_nodes()=}, {map_graph.number_of_edges()=}"
        )

        # remove precomputed path lengths as the graph is now different
        allPathLengths = map_graph.graph.get("allPathLengths", {})
        edge_weight_str = self.args.goal_gen["edge_weight_str"]
        if edge_weight_str in allPathLengths:
            allPathLengths.pop(edge_weight_str)
        return map_graph

    def set_goal_generator(self):
        goal_source = self.args.goal_source.lower()
        if goal_source == "topological":
            segmentor_name = self.args.segmentor.lower()

            self.segmentor = self.preload_data["segmentor"]
            cfg_goalie = self.args.goal_gen
            cfg_goalie.update({"use_gt_localization": self.args.use_gt_localization})
            if segmentor_name == "sam2":
                assert (
                    cfg_goalie["matcher_name"] == "sam2"
                ), "TODO: is other matcher implemented for this segmentor?"
                cfg_goalie.update({"sam2_tracker": self.segmentor})

            self.goalie = goal_gen.Goal_Gen(
                W=self.image_width,
                H=self.image_height,
                G=self.map_graph,
                map_path=str(self.path_episode),
                poses=self.agent_states,
                task_type=self.args.task_type,
                cfg=cfg_goalie,
            )

            goalNodeImg = self.goalie.visualize_goal_node()
            cv2.imwrite(
                str(self.path_episode_results / "final_goal_mask_vis_pred.jpg"),
                goalNodeImg[:, :, ::-1],
            )

            if self.args.use_gt_localization:
                self.goalie.localizer.localizedImgIdx, _ = self.get_GT_closest_map_img()
            if self.args.reverse:
                self.goalie.localizer.localizer_iter_ub = self.num_map_images

            # to save time over storage
            if (
                not self.goalie.planner_g.precomputed_allPathLengths_found
                and not self.goalie.planner_g.preplan_to_goals_only
                and cfg_goalie["rewrite_graph_with_allPathLengths"]
            ):
                logger.info("Rewritng graph with allPathLengths")
                allPathLengths = self.map_graph.graph.get("allPathLengths", {})
                allPathLengths.update(
                    {
                        cfg_goalie[
                            "edge_weight_str"
                        ]: self.goalie.planner_g.allPathLengths
                    }
                )
                self.map_graph.graph["allPathLengths"] = allPathLengths
                pickle.dump(self.map_graph, open(self.path_graph, "wb"))

        elif goal_source == "gt_metric":
            # map_graph is not needed
            pass
        elif goal_source == "lang_e3d":
            # No graph or goal generator needed; instruction is stored per-episode
            pass
        elif goal_source in ["gt_topological", "gt_topometric"]:
            self.get_goal_object_id()
            self.precompute_graph_paths()
        elif goal_source == "image_topological":
            if self.args.method.lower() == "learnt":
                self.goalie = type("", (), {})()
                self.goalie.config = self.preload_data["goal_controller"].config
                self.goalie.map_images = []
                self.goalie.loc_radius = self.goalie.config["loc_radius"]
                img_paths = natsorted((self.path_episode / "images/").iterdir())
                img_paths = img_paths[: self.final_goal_image_idx + 1]
                for img_path in img_paths:
                    self.goalie.map_images.append(cv2.imread(str(img_path))[:, :, ::-1])
                self.goalie.goal_idx, _ = self.get_GT_closest_map_img()
                self.goalie.num_map_images = len(self.goalie.map_images)
        elif goal_source == "lang_e3d":
            # Goal is built online from language + LangGeoNetV2; no static map needed.
            pass
        else:
            raise NotImplementedError(f"{self.args.goal_source=} is not defined...")

    def get_goal_object_id(self):
        if self.args.reverse:
            self.goal_object_id = utils_data.find_reverse_traverse_goal(
                self.agent, self.sim, self.final_goal_state, self.map_graph
            )
            if not os.path.exists(f"{self.path_episode}/reverse_goal.npy"):
                print(f"Saving reverse goal to {self.path_episode}/reverse_goal.npy")
                np.save(
                    f"{self.path_episode}/reverse_goal.npy",
                    {
                        "instance_id": self.goal_object_id,
                        "agent_state": self.final_goal_state,
                    },
                )

        elif self.args.task_type in ["alt_goal"]:
            self.goal_object_id = utils_data.get_goal_info_alt_goal(
                self.path_episode, self.args.task_type
            )[-1]
        else:
            self.goal_object_id = int(str(self.path_episode).split("_")[-2])

    def precompute_graph_paths(self):
        self.graph_instance_ids, self.graph_path_lengths = (
            utils_goals.find_graph_instance_ids_and_path_lengths(
                self.map_graph,
                self.goal_object_id,
                device=self.device,
                weight=(
                    self.args.goal_gen["edge_weight_str"]
                    if self.args.goal_source.lower() == "gt_topometric"
                    else "margin"
                ),
            )
        )

    def set_controller(self):
        control_method = self.args.method.lower()
        goal_controller = None
        self.collided = None

        # select the type of controller to use
        if control_method == "tango":
            from libs.control.tango.pid import SteerPID
            from libs.control.tango.tango import TangoControl

            pid_steer = SteerPID(
                Kp=self.pid_steer_values[0],
                Ki=self.pid_steer_values[1],
                Kd=self.pid_steer_values[2],
            )

            intrinsics = utils.build_intrinsics(
                image_width=self.image_width,
                image_height=self.image_height,
                field_of_view_radians_u=self.hfov_radians,
                device=self.device,
            )

            goal_controller = TangoControl(
                traversable_classes=self.traversable_class_indices,
                pid_steer=pid_steer,
                default_velocity_control=self.velocity_control,
                h_image=self.image_height,
                w_image=self.image_width,
                intrinsics=intrinsics,
                time_delta=self.time_delta,
                grid_size=0.125,
                device=self.device,
            )

        elif control_method == "pixnav":
            from libs.pixnav.policy_agent import Policy_Agent
            from libs.pixnav.constants import POLICY_CHECKPOINT

            goal_controller = Policy_Agent(model_path=POLICY_CHECKPOINT)
            self.collided = False

        elif control_method == "learnt":
            goal_controller = self.preload_data["goal_controller"]
            goal_controller.reset_params()
            goal_controller.dirname_vis_episode = self.dirname_vis_episode

        self.goal_controller = goal_controller

    def get_GT_closest_map_img(self):
        dists = np.linalg.norm(
            self.agent_positions_in_map - self.agent.get_state().position, axis=1
        )
        topK = 2 * self.args.goal_gen["loc_radius"]
        closest_idxs = np.argsort(dists)[:topK]
        # approximately subsample ref indices
        closest_idxs = sorted(closest_idxs)[:: self.args.goal_gen["subsample_ref"]]
        closest_idx = np.argmin(dists)
        return closest_idx, closest_idxs

    def get_goal(self, rgb, depth, semantic_instance):
        goal_source = self.args.goal_source.lower()
        control_method = self.args.method.lower()
        self.traversable_mask = None
        self.goal_mask = None
        self.semantic_instance_predicted = None

        goal_img_idx, localizedImgInds = None, None
        if self.args.use_gt_localization:
            goal_img_idx, localizedImgInds = self.get_GT_closest_map_img()

        if goal_source == "gt_metric":
            _, plsDict, self.goal_mask = ust.get_pathlength_GT(
                self.sim,
                self.agent,
                depth,
                semantic_instance,
                self.final_goal_position,
                None,
            )
            self.control_input_robohop = semantic_instance

            instaIds, pls = list(zip(*plsDict.items()))
            masks = semantic_instance[None, ...] == np.array(instaIds)[:, None, None]
            self.control_input_learnt = [masks, np.array(pls)]

        elif goal_source in ["gt_topological", "gt_topometric"]:
            self.goal_mask = utils_goals.get_goal_mask_GT(
                graph_instance_ids=self.graph_instance_ids,
                pls=self.graph_path_lengths,
                sem=semantic_instance,
                device=self.device,
            )
            self.control_input_robohop = semantic_instance

            if not control_method == "learnt":
                # remove masks
                self.goal_mask[np.isin(semantic_instance, self.bad_goal_classes)] = 99

            masks = (
                semantic_instance[None, ...]
                == np.unique(semantic_instance)[:, None, None]
            )
            pls = [self.goal_mask[m].mean() for m in masks]
            self.control_input_learnt = [masks, np.array(pls)]

        elif goal_source == "topological":
            remove_mask = None
            # if 0:
            #     instance_ids_to_remove = np.concatenate([bad_goal_classes, traversable_class_indices])
            #     remove_mask = (semantic_instance_sim[:, :, None] == instance_ids_to_remove[None, None, :]).sum(-1).astype(bool)
            if len(self.args.goal_gen["textLabels"]) > 0:
                assert self.args.segmentor == "fast_sam"
            seg_results = self.segmentor.segment(
                rgb[:, :, :3], textLabels=self.args.goal_gen["textLabels"]
            )
            if self.args.segmentor.lower() in ["fast_sam", "sam2"]:
                self.semantic_instance_predicted, _, self.traversable_mask = seg_results
            else:
                self.semantic_instance_predicted = seg_results

            if self.args.cull_qry_instances:
                cull_mask = np.sum(
                    semantic_instance[None, ...]
                    == self.cull_instance_ids[:, None, None],
                    axis=0,
                ).astype(bool)
                qryMasks = utils.nodes2key(
                    self.semantic_instance_predicted, "segmentation"
                )
                areaThresh = np.ceil(
                    0.001 * semantic_instance.shape[0] * semantic_instance.shape[1]
                )
                cull_inds = []
                for mi, mask in enumerate(qryMasks):
                    # or mask.sum() <= areaThresh:
                    if mask[cull_mask].sum() >= areaThresh:
                        cull_inds.append(mi)
                logger.info(f"{len(cull_inds)}/{len(qryMasks)} instances to cull")
                if len(cull_inds) == len(qryMasks):
                    logger.warning("Skipped culling as len(cull_inds) == len(qryMasks)")
                else:
                    self.semantic_instance_predicted = [
                        self.semantic_instance_predicted[mi]
                        for mi in range(len(self.semantic_instance_predicted))
                        if mi not in cull_inds
                    ]

            if self.args.use_gt_localization:
                self.goalie.localizer.localizedImgIdx = (
                    goal_img_idx - 1 if self.args.reverse else goal_img_idx + 1
                )
                self.goalie.localizer.lost = False

            self.goal_mask = self.goalie.get_goal_mask(
                qryImg=rgb[:, :, :3],
                qryNodes=self.semantic_instance_predicted,
                qryPosition=(
                    self.agent.get_state().position
                    if self.args.debug and self.args.env == "sim"
                    else None
                ),
                remove_mask=remove_mask,
                refImgInds=localizedImgInds,
            )
            self.control_input_robohop = [self.goalie.pls, self.goalie.coords]
            self.control_input_learnt = [
                # self.goalie.qryMasks[self.goalie.matchPairs[:, 0]], self.goalie.pls]
                self.goalie.qryMasks,
                self.goalie.pls_min,
            ]
            if (
                self.goalie.qryMasks is not None
                and self.control_input_learnt[0] is not None
                and self.control_input_learnt[1] is not None
            ):
                assert len(self.control_input_learnt[0]) == len(
                    self.control_input_learnt[1]
                )

        elif goal_source == 'image_topological':
            if control_method == 'learnt':
                plan_shift = 1
                if self.args.use_gt_localization:
                    if self.args.reverse:
                        plan_shift = -1
                    self.goalie.goal_idx = goal_img_idx + plan_shift

                if self.args.use_gt_localization and self.goalie.config['fixed_plan']:
                    img_goal = self.goalie.map_images[min(
                        self.goalie.goal_idx, self.goalie.num_map_images - 1)]
                else:
                    start = max(self.goalie.goal_idx -
                                self.goalie.loc_radius, 0)
                    end = min(
                        self.goalie.goal_idx + self.goalie.loc_radius + 1, self.goalie.num_map_images)
                    img_goal_list = self.goalie.map_images[start:end]
                    self.goalie.goal_idx = self.goal_controller.predict_goal_idx(
                        rgb, img_goal_list, self.args.reverse)
                    img_goal = img_goal_list[self.goalie.goal_idx]
                    self.goalie.goal_idx += start
                self.control_input_learnt = img_goal
                self.goal_mask = img_goal.copy()
            else:
                raise NotImplementedError(
                    f'{goal_source=} only defined for {control_method=}...')

        elif goal_source == "lang_e3d":
            from PIL import Image as _PIL_Image

            segmentor = self.preload_data.get("segmentor")
            lange3d = self.preload_data["lange3d"]
            clip_processor = self.preload_data["clip_processor"]
            device = self.device

            # ── Resolve instruction for this step ─────────────────────────────
            # "episodic"     : same episode-level instruction every step.
            # "next_action"  : look up per-frame NAI from the run-level LMDB
            #                  (built in run() by build_run_instructions_lmdb).
            #                  Falls back to episodic when entry is missing/empty.
            instruction = self.episode_instruction
            if (self.instruction_type == "next_action"
                    and self.instr_lmdb is not None
                    and goal_img_idx is not None):
                _key = f"{self.ep_num}/{goal_img_idx:03d}".encode("utf-8")
                with self.instr_lmdb.begin() as _txn:
                    _raw = _txn.get(_key)
                if _raw is not None:
                    _rec = pickle.loads(_raw)
                    nai = _rec.get("next_action_instruction", "")
                    if nai:
                        instruction = nai
                        logger.debug(f"[ep {self.ep_num} frame {goal_img_idx:03d}] "
                                     f"NAI: {nai[:60]}…")
                    else:
                        logger.debug(f"[ep {self.ep_num} frame {goal_img_idx:03d}] "
                                     "empty NAI — using episodic")

            # ── FastSAM segmentation ──────────────────────────────────────────
            if segmentor is not None:
                seg_result = segmentor.segment(rgb[:, :, :3], retMaskAsDict=False)
                if seg_result[0] is None:
                    masks_np = np.zeros((0, rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
                else:
                    masks_np = seg_result[0].astype(np.uint8)  # [K, H, W]
            else:
                masks_np = np.zeros((0, rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

            K = masks_np.shape[0]

            if K == 0:
                # No masks: reuse last valid tensor so agent keeps moving.
                # goal_mask=1 everywhere → visual servoing goes straight.
                self.control_input_learnt = getattr(self, '_last_goal_tensor', [None, None])
                self.control_input_robohop = semantic_instance
                self.goal_mask = np.full(
                    (rgb.shape[0], rgb.shape[1]), fill_value=1.0, dtype=np.float32
                )
            else:
                # ── CLIP image + instruction tokenization ─────────────────────
                clip_inputs = clip_processor(
                    images=_PIL_Image.fromarray(rgb[:, :, :3].astype(np.uint8)),
                    text=[instruction],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                )
                pixel_values = clip_inputs["pixel_values"].to(device)
                input_ids = clip_inputs["input_ids"].to(device)
                attn_mask = clip_inputs["attention_mask"].to(device)

                masks_tensor = torch.from_numpy(masks_np.astype(bool)).to(device)

                # ── LangGeoNetV2 → per-object cost (lower = closer to goal) ──
                with torch.no_grad():
                    lang_preds, _ = lange3d(
                        pixel_values, [masks_tensor], input_ids, attn_mask)

                # lang_preds is a list[B] of [K_b] tensors; B==1 here.
                logits = lang_preds[0].detach().float().cpu().numpy()
                if logits.shape[0] != K:
                    raise RuntimeError(
                        f"LangGeoNet returned {logits.shape[0]} costs for {K} masks")

                # ── Build TopoPaths-style goal tensor for learnt controller ───
                # Use continuous near/far positional-encoding blend by cost.
                # This matches TopoPaths.build_differentiable_goal() and routes
                # predict_from_goal_enc() (not predict() with corrupt PL scale).
                topopaths = self.preload_data.get("topopaths")
                rank_enc_np = topopaths.rank_enc          # [maxRank+1, dims]
                near_enc = rank_enc_np[0]                 # [dims] lowest PL
                far_enc  = rank_enc_np[-1]                # [dims] highest PL

                lo_t, hi_t = float(logits.min()), float(logits.max())
                if hi_t - lo_t > 1e-8:
                    pls = (logits - lo_t) / (hi_t - lo_t)  # [K] in [0,1]
                else:
                    pls = np.zeros_like(logits)
                goal_mask = np.ones(
                    (rgb.shape[0], rgb.shape[1]), dtype=np.float32
                )  # default = max cost
                masks_bool = masks_np.astype(bool)  # [K, H, W]
                order = np.argsort(pls)[::-1]  # descending pls
                for ki in order:
                    goal_mask[masks_bool[ki]] = pls[ki]

                self.goal_mask = goal_mask  # (H, W) float32, low = near goal

                # ── control_input_robohop: semantic_instance for robohop/tango
                # self.control_input_robohop = semantic_instance

                # ── control_input_learnt: Tensor [1, dims, H//4, W//4] for learnt controller ──
                # Per-object encoding: blend near_enc and far_enc by normalized cost
                
                # Continuous near/far blend → [1, dims, H//4, W//4] Tensor
                # Matches TopoPaths.build_differentiable_goal() convention:
                #   cost≈0 (near goal) → near_enc = rank_enc[0]
                #   cost≈1 (far)       → far_enc  = rank_enc[-1]
                costs_np = pls  # [K] in [0,1], 0=near goal
                obj_enc = ((1.0 - costs_np[:, None]) * near_enc
                           + costs_np[:, None] * far_enc)  # [K, dims]

                H_full, W_full = rgb.shape[0], rgb.shape[1]
                masks_f = torch.from_numpy(masks_np.astype(np.float32))
                masks_half = torch.nn.functional.interpolate(
                    masks_f.unsqueeze(1),
                    size=(H_full // 4, W_full // 4),
                    mode='nearest',
                ).squeeze(1).numpy()  # [K, H//4, W//4]

                img_enc = np.einsum("kd,khw->dhw", obj_enc, masks_half)  # [dims, H//4, W//4]
                goal_tensor = torch.from_numpy(img_enc.astype(np.float32)).unsqueeze(0)
                self._last_goal_tensor = goal_tensor        # cache for K==0 fallback
                self.control_input_learnt = goal_tensor     # Tensor → predict_from_goal_enc
                #NOTE: Discrete encoding--> going through .predict()
                # self.control_input_learnt = [masks_bool, pls.astype(np.float32)]

        else:
            raise NotImplementedError(f"{self.args.goal_source} is not available...")

    def get_control_signal(self, step, rgb, depth):
        control_method = self.args.method.lower()
        goals_image = None

        if control_method == "robohop":  # the og controller
            self.velocity_control, self.theta_control, goals_image = control_with_mask(
                self.control_input_robohop,
                self.goal_mask,
                v=self.velocity_control,
                gain=1,
                tao=5,
            )
            self.theta_control = -self.theta_control
            self.vis_img = (
                255.0
                - 255 * (utils_viz.goal_mask_to_vis(goals_image, outlier_min_val=255))
            ).astype(np.uint8)

        elif control_method == "tango":
            self.velocity_control, self.theta_control, goals_image_ = (
                self.goal_controller.control(
                    depth,
                    self.control_input_robohop,
                    self.goal_mask,
                    self.traversable_mask,
                )
            )
            if goals_image_ is not None:
                self.vis_img = (
                    255.0
                    - 255
                    * (utils_viz.goal_mask_to_vis(goals_image_, outlier_min_val=255))
                ).astype(np.uint8)
            else:
                self.vis_img = self.vis_img_default.copy()

        elif control_method == "pixnav":
            self.pixnav_goal_mask = utils.robohop_to_pixnav_goal_mask(
                self.goal_mask, depth
            )
            if not (step % 63) or self.discrete_action == 0:
                self.goal_controller.reset(rgb, self.pixnav_goal_mask.astype(np.uint8))
            self.discrete_action, predicted_mask = self.goal_controller.step(
                rgb, self.collided)

        elif control_method == 'learnt':
            if isinstance(self.control_input_learnt, torch.Tensor):
                # lang_e3d: goal already encoded as tensor [1, dims, H, W]
                self.velocity_control, self.theta_control, self.vis_img = \
                    self.goal_controller.predict_from_goal_enc(
                        rgb, self.control_input_learnt)
            elif self.control_input_learnt[0] is None or self.control_input_learnt[1] is None:
                self.velocity_control, self.theta_control, self.vis_img = 0, 0, self.vis_img_default.copy()
            else:
                self.velocity_control, self.theta_control, self.vis_img = self.goal_controller.predict(
                    rgb, self.control_input_learnt)
            self.controller_logs = self.goal_controller.controller_logs

        elif control_method == 'disabled':
            # Visual servoing: bypass GNM entirely.
            # Uses goal_mask from LangGeoNetV2 (low=near goal) to steer toward
            # the centroid of the minimum-cost object region.
            gm = self.goal_mask  # [H, W] float32; 0=near goal, 1=background
            H_im, W_im = gm.shape
            min_cost = float(gm.min())
            if min_cost < 0.95:  # at least one object identified below background
                thresh = min_cost + 0.15 * (1.0 - min_cost)
                near_mask = gm <= thresh
                if near_mask.sum() > 0:
                    _, xs = np.where(near_mask)
                    cx = float(xs.mean())
                    # cx < W/2 → object left → turn left (positive w)
                    angular_err = (W_im / 2.0 - cx) / (W_im / 2.0)
                    if abs(cx - W_im / 2.0) < W_im * 0.05:  # dead-band ±5%
                        angular_err = 0.0
                    w = float(np.clip(0.5 * angular_err, -0.5, 0.5))
                    v = 0.06
                else:
                    v, w = 0.05, 0.0
            else:
                v, w = 0.05, 0.0  # no near-goal object visible; go straight
            self.velocity_control = v
            self.theta_control = -w  # convention: theta_control = -w
            self.vis_img = self.vis_img_default.copy()
            self.controller_logs = []



        else:
            raise NotImplementedError(f"{self.args.method} is not available...")
        return goals_image

    def execute_action(self):
        control_method = self.args.method.lower()

        if control_method == "pixnav":
            action_dict = {
                0: "stop",
                1: "move_forward",
                2: "turn_left",
                3: "turn_right",
                4: "look_up",
                5: "look_down",
            }
            previous_state = self.agent.state
            action = action_dict[self.discrete_action]
            _ = self.sim.step(action)
            current_state = self.agent.state
            self.collided = utils.has_collided(self.sim, previous_state, current_state)
        else:
            # ── Collision escape: reverse + hard-turn after 5 stuck steps ─────
            # Track on self so this works even when goal_controller is None
            # (e.g. method='disabled' visual-servoing path).
            _ESCAPE_THRESHOLD = 5
            if self.collided:
                self._consecutive_collisions = getattr(self, '_consecutive_collisions', 0) + 1
            else:
                self._consecutive_collisions = 0
            # Mirror onto goal_controller if it exists (backward compat)
            if self.goal_controller is not None:
                self.goal_controller.consecutive_collisions = self._consecutive_collisions

            if self._consecutive_collisions >= _ESCAPE_THRESHOLD:
                logger.warning(
                    f"Collision escape triggered after "
                    f"{self._consecutive_collisions} stuck steps"
                )
                escape_v = -0.05
                escape_w = 0.5
                if self.goal_controller is not None:
                    self.goal_controller._ema_wp = None
                self._consecutive_collisions = 0
                self.agent, self.sim, self.collided = utils.apply_velocity(
                    vel_control=self.vel_control,
                    agent=self.agent,
                    sim=self.sim,
                    velocity=escape_v,
                    steer=escape_w,
                    time_step=self.time_delta,
                )
            else:
                self.agent, self.sim, self.collided = utils.apply_velocity(
                    vel_control=self.vel_control,
                    agent=self.agent,
                    sim=self.sim,
                    velocity=self.velocity_control,
                    steer=-self.theta_control,  # opposite y axis
                    time_step=self.time_delta,
                )


    def is_done(self):
        done = False
        current_robot_state = self.agent.get_state()  # world coordinates
        self.distance_to_goal = ust.find_shortest_path(
            self.sim, p1=current_robot_state.position, p2=self.final_goal_position
        )[0]
        if self.distance_to_goal <= self.args.threshold_goal_distance:
            logger.info(f"\nWinner! dist to goal: {self.distance_to_goal:.6f}\n")
            self.success_status = "success"
            done = True
        return done

    def set_logging(self):
        self.dirname_vis_episode = self.path_episode_results / "vis"
        self.dirname_vis_episode.mkdir(exist_ok=True, parents=True)

        self.filename_metadata_episode = self.path_episode_results / "metadata.txt"
        self.filename_results_episode = self.path_episode_results / "results.csv"

        utils.initialize_results(
            self.filename_metadata_episode,
            self.filename_results_episode,
            self.args,
            self.pid_steer_values,
            self.hfov_radians,
            self.time_delta,
            self.velocity_control,
            self.final_goal_position,
            self.traversable_class_indices,
        )

        results_dict_keys = [
            "step",
            "distance_to_goal",
            "velocity_control",
            "theta_control",
            "collided",
            "discrete_action",
            "agent_states",
            "controller_logs",
        ]
        self.results_dict = {k: [] for k in results_dict_keys}

    def log_results(self, step, final=False):
        if not final:
            utils.write_results(
                self.filename_results_episode,
                step,
                self.agent.get_state() if self.agent is not None else None,
                self.distance_to_goal,
                self.velocity_control,
                self.theta_control,
                self.collided,
                self.discrete_action,
            )
            if self.vis is not None:
                if self.args.env == "sim":
                    self.update_vis_sim()
                else:
                    self.update_vis()

            results_dict_curr = {
                "step": step,
                "distance_to_goal": self.distance_to_goal,
                "velocity_control": self.velocity_control,
                "theta_control": self.theta_control,
                "collided": self.collided,
                "discrete_action": self.discrete_action,
                "agent_states": self.agent.get_state() if self.agent is not None else None,
                "controller_logs": self.controller_logs[-1] if self.controller_logs is not None and len(self.controller_logs) > 0 else None,
            }

            self.update_results_dict(results_dict_curr)

        else:
            utils.write_final_meta_results(
                filename_metadata_episode=self.filename_metadata_episode,
                success_status=self.success_status,
                final_distance=self.distance_to_goal,
                step=step,
                distance_to_final_goal=self.distance_to_final_goal,
            )

            np.savez(
                self.path_episode_results / "results_dict.npz", **self.results_dict
            )

    def update_results_dict(self, curr_dict):
        for k, v in curr_dict.items():
            self.results_dict[k].append(v)

    def update_vis_sim(self):
        # if this is the first call, init video
        ratio = self.vis_img.shape[1] / self.vis.tdv.shape[1]
        if self.vis.video is None:
            # resize tdv to match the rgb image
            self.tdv = cv2.resize(self.vis.tdv, dsize=None, fx=ratio, fy=ratio)
            self.video_cfg["width"] = self.vis_img.shape[1]
            self.video_cfg["height"] = self.vis_img.shape[0] + self.tdv.shape[0]
            self.vis.init_video(self.video_cfg)

        self.vis.draw_infer_step(self.agent.get_state())
        self.tdv = cv2.resize(self.vis.tdv, dsize=None, fx=ratio, fy=ratio)
        combined_img = np.concatenate((self.tdv, self.vis_img), axis=0)
        self.vis.save_video_frame(combined_img)

    def update_vis(self):
        # if this is the first call, init video
        if self.vis.video is None:
            self.video_cfg["width"] = self.vis_img.shape[1]
            self.video_cfg["height"] = self.vis_img.shape[0]
            self.vis.init_video(self.video_cfg)

        self.vis.save_video_frame(self.vis_img)

    def init_plotting(self):
        # TODO: better handle 'plt'
        import matplotlib
        import matplotlib.pyplot as plt

        if self.args.save_vis:
            matplotlib.use("Agg")  # Use the Agg backend to suppress plots

        import matplotlib.style as mplstyle

        mplstyle.use("fast")
        mplstyle.use(["dark_background", "ggplot", "fast"])
        fig, ax = utils_viz.setup_sim_plots()
        return ax, plt

    def plot(self, ax, plt, step, rgb, depth, semantic_instance):
        goals_image = None

        if self.args.goal_source.lower() == "topological":
            if self.semantic_instance_predicted is None:
                semantic_instance_vis = np.zeros(rgb.shape[:2])
            else:
                semantic_instance_vis = utils_viz.show_anns(
                    None, self.semantic_instance_predicted, borders=False
                )
        else:
            semantic_instance_vis = semantic_instance

        goal_mask_vis = utils_viz.goal_mask_to_vis(self.goal_mask)
        goal = self.goal_mask == self.goal_mask.min()  # .astype(int)
        if self.args.method.lower() == "pixnav":
            goal += (self.pixnav_goal_mask / self.pixnav_goal_mask.max()).astype(
                int
            ) * 2
        utils_viz.plot_sensors(
            ax=ax,
            display_img=rgb,
            semantic=semantic_instance_vis,
            depth=depth,
            goal=goal,
            goal_mask=goal_mask_vis,
            flow_goal=(
                goals_image if goals_image is not None else np.zeros(rgb.shape[:2])
            ),
            trav_mask=self.traversable_mask,
        )
        if self.args.method.lower() == "tango":
            utils_viz.plot_path_points(
                ax=[ax[1, 2], ax[1, 0]],
                points=self.goal_controller.point_poses,
                cost_map_relative_bev=self.goal_controller.planning_cost_map_relative_bev_safe,
                colour="red",
            )

        if self.args.save_vis:
            plt.tight_layout()
            plt.savefig(
                self.dirname_vis_episode / f"{step:04d}.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
        else:
            plt.pause(0.05)  # pause a bit so that plots are updated

    def close(self, step=-1):
        # if self.args.plot:
        # plt.close()
        if self.args.log_robot:
            self.log_results(step, final=True)

        if hasattr(self, "vis") and self.vis:
            self.vis.close()
        if hasattr(self, "sim") and self.sim:
            self.sim.close()


def setup_wandb_logging(args):
    wandb.login()
    wandb.init(project="obj_rel_nav")
    wandb.config.update(args)
    wandb.run.name = (
        f"{args.exp_name}_{args.task_type}_{args.method}_{args.goal_source}"
    )


def wandb_log_episode(epsiode_name, results_dict, video_path=None):
    for step in range(len(results_dict["step"])):
        wandb.log(
            {
                f"{epsiode_name}/{key}": results_dict[key][step]
                for key in results_dict.keys()
                if key not in ["step", "agent_states"]
            },
            commit=False,
        )
    wandb.log({})

    if video_path is not None and os.path.exists(video_path):
        video_path_2 = video_path[:-4] + "_2.mp4"
        os.system(
            f"ffmpeg -y -loglevel 0 -i {video_path} -vcodec libx264 {video_path_2}"
        )
        wandb.log({"video": wandb.Video(video_path_2)}, commit=True)


def load_run_list(args, path_episode_root) -> list:
    if args.run_list == "":
        path_episodes = sorted(path_episode_root.glob("*"))
    else:
        path_episodes = []
        if args.path_run == "":
            raise ValueError("Run path must be specified when using run list!")
        if args.run_list.lower() in ["winners", "failures", "no_good", "custom"]:
            if args.run_list.lower() not in ["no_good", "custom"]:
                logger.info(
                    f"Setting logging to False when running winner or failure list! - arg.log_robot:{args.log_robot}"
                )
                args.log_robot = False
            with open(
                str(Path(args.path_run) / "summary" / f"{args.run_list.lower()}.csv"),
                "r",
            ) as f:
                for line in f.readlines():
                    path_episodes.append(
                        path_episode_root
                        / line[: line.rfind(f"_{args.method}")].strip("\n")
                    )
        else:
            raise ValueError(f"{args.run_list} is not a valid option.")
    return path_episodes


def init_results_dir_and_save_cfg(args, default_logger=None):
    if (args.log_robot or args.save_vis) and args.run_list == "":
        path_results = Path(args.path_results)
        task_str = args.task_type
        if args.reverse:
            task_str += "_reverse"
        path_results_folder = (
            path_results
            / task_str
            / args.exp_name
            / args.split
            / args.max_start_distance
            / f'{datetime.now().strftime("%Y%m%d-%H-%M-%S")}_{args.method.lower()}_{args.goal_source}'
        )
        path_results_folder.mkdir(exist_ok=True, parents=True)
        if default_logger is not None:
            default_logger.update_file_handler_root(path_results_folder / "output.log")
        print(f"Logging to: {str(path_results_folder)}")
    elif args.run_list != "":
        path_results_folder = Path(args.path_run)
        print(f"(Overwrite) Logging to: {str(path_results_folder)}")

    if args.log_robot:
        save_dict(path_results_folder / "args.yaml", vars(args))
        if args.method.lower() == "learnt":
            args_filepath = args.controller["config_file"]
            shutil.copyfile(
                args_filepath, path_results_folder / Path(args_filepath).name
            )

    return path_results_folder


def build_run_instructions_lmdb(episodes: list, path_results_folder: Path,
                                preload_data: dict) -> None:
    """Build a single per-run LMDB of instructions from episode folders.

    Called once in run() after the episode list is known.  Reads each
    episode's ``instruction.txt`` (episodic) and ``next_action_instructions.json``
    (per-frame NAI, written by build_mp3d_iin_from_h5.py) and stores them in

        <path_results_folder>/instructions.lmdb

    LMDB key layout:
        b"{ep_num}/{seq_idx:03d}"   →  pickle({"episodic_instruction": str,
                                                "next_action_instruction": str})

    The opened read-only env is stored in ``preload_data["instr_lmdb"]``.
    Episodes whose folders lack instruction files are skipped silently.
    """
    try:
        import lmdb as _lmdb
    except ImportError:
        logger.warning("lmdb not installed — instruction LMDB disabled. "
                       "pip install lmdb to enable next_action mode.")
        return

    lmdb_path = path_results_folder / "instructions.lmdb"
    lmdb_path.mkdir(parents=True, exist_ok=True)
    map_size = int(512 * (1 << 20))  # 512 MB virtual (sparse on Linux)
    all_keys: list[bytes] = []
    n_ep, n_nai = 0, 0

    with _lmdb.open(str(lmdb_path), map_size=map_size, subdir=True,
                    meminit=False, map_async=True) as env:
        for ep_path in episodes:
            instr_txt = ep_path / "instruction.txt"
            if not instr_txt.exists():
                continue
            episodic = instr_txt.read_text().strip()

            # Parse episode numeric ID ("17DRP5sb8fy_0011458_..." → "11458")
            parts = ep_path.name.split("_")
            raw_id = parts[1] if len(parts) > 1 else ep_path.name
            ep_num = str(int(raw_id)) if raw_id.isdigit() else raw_id

            # Per-frame NAI map {"000": "...", "001": "..."}
            nai_json = ep_path / "next_action_instructions.json"
            nai_map: dict = {}
            if nai_json.exists():
                try:
                    nai_map = json.loads(nai_json.read_text())
                    n_nai += 1
                except Exception:
                    pass

            # Determine frame count from agent_states
            agent_states_path = ep_path / "agent_states.npy"
            if not agent_states_path.exists():
                continue
            n_frames = len(np.load(str(agent_states_path), allow_pickle=True))

            with env.begin(write=True) as txn:
                for seq_idx in range(n_frames):
                    key = f"{ep_num}/{seq_idx:03d}".encode("utf-8")
                    nai = nai_map.get(f"{seq_idx:03d}", "")
                    record = {
                        "episodic_instruction": episodic,
                        "next_action_instruction": nai,
                    }
                    txn.put(key, pickle.dumps(record, protocol=4))
                    all_keys.append(key)
            n_ep += 1

        with env.begin(write=True) as txn:
            txn.put(b"__keys__", pickle.dumps(sorted(all_keys), protocol=4))

    logger.info(f"[instructions LMDB] built {len(all_keys)} records "
                f"from {n_ep} episodes ({n_nai} with NAI) → {lmdb_path}")

    # Open read-only for use during episode navigation
    preload_data["instr_lmdb"] = _lmdb.open(
        str(lmdb_path), readonly=True, lock=False,
        readahead=False, meminit=False,
    )


def preload_models(args):
    # preload some models before iterating over the episodes
    goal_controller = model_loader.get_controller_model(
        args.method, args.goal_source, args.controller["config_file"])

    segmentor = None
    if args.goal_source == "topological":

        # use predefined traversable classes with fast_sam predictions only if it is tango and infer_traversable is True
        traversable_class_names = (
            args.traversable_class_names
            if args.method.lower() == "tango" and args.infer_traversable
            else None
        )

        segmentor = model_loader.get_segmentor(
            args.segmentor,
            args.sim["width"],
            args.sim["height"],
            path_models=args.path_models,
            traversable_class_names=traversable_class_names,
        )

    depth_model = None
    if args.infer_depth:
        depth_model = model_loader.get_depth_model()

    # collect preload data that each episode instance can reuse
    preload_data = {
        "goal_controller": goal_controller,
        "segmentor": segmentor,
        "depth_model": depth_model,
    }

    # ── lang_e3d: joint checkpoint + segmentor + optional H5 file ────────────
    if args.goal_source == "lang_e3d":
        # Segmentor is needed at every step for online segmentation
        if preload_data["segmentor"] is None:
            traversable_class_names = (
                args.traversable_class_names
                if args.method.lower() == "tango" and args.infer_traversable
                else None
            )
            preload_data["segmentor"] = model_loader.get_segmentor(
                getattr(args, "segmentor", "fast_sam"),
                args.sim["width"],
                args.sim["height"],
                path_models=getattr(args, "path_models", None),
                traversable_class_names=traversable_class_names,
            )

        # Load joint models (LangGeoNetV2 + GNM + TopoPaths + CLIPProcessor)
        joint_checkpoint = getattr(args, "joint_checkpoint", None)
        if joint_checkpoint is None:
            raise ValueError(
                "joint_checkpoint must be set in config for goal_source=lang_e3d")
        joint = model_loader.get_joint_models(joint_checkpoint)

        # Replace the controller's GNM weights with the joint checkpoint's GNM
        if goal_controller is not None:
            goal_controller.model = joint["gnm_joint"]

        preload_data["lange3d"] = joint["lange3d"]
        preload_data["topopaths"] = joint["topopaths"]
        preload_data["clip_processor"] = joint["clip_processor"]

        # Open H5 file so each episode can read its instruction
        h5_path = getattr(args, "h5_path", None)
        if h5_path:
            import h5py as _h5py
            preload_data["h5_file"] = _h5py.File(h5_path, "r")

    return preload_data


def set_start_state_reverse_orientation(agent_states, start_index):
    start_state = agent_states[start_index]
    # compute orientation, looking at the next GT forward step
    lookat_index = start_index - 1
    if lookat_index < 0:
        print("Cannot reverse orientation at the start of the episode.")
        return None

    # search/validate end_idx in reverse direction
    for k in range(lookat_index, -1, -1):
        # keep looking if agent hasn't moved
        if np.linalg.norm(start_state.position - agent_states[k].position) <= 0.1:
            continue
        else:
            lookat_index = k
            break
    # looking in the reverse direction
    start_state.rotation = ust.get_agent_rotation_from_two_positions(
        start_state.position, agent_states[lookat_index].position
    )
    return start_state


def closest_state(sim, agent_states, distance_threshold: float, final_position=None):
    distances = np.zeros_like(agent_states)
    final_position = (
        agent_states[-1].position if final_position is None else final_position
    )
    for i, p in enumerate(agent_states):
        distances[i] = ust.find_shortest_path(sim, final_position, p.position)[0]
    start_index = ((distances - distance_threshold) ** 2).argmin()
    return start_index


def select_starting_state(sim, args, agent_states, final_position=None):
    # reverse traverse episodes end 1m before the original start, offset that
    distance_threshold_offset = 1 if args.reverse else 0
    if args.max_start_distance.lower() == "easy":
        start_index = closest_state(
            sim, agent_states, 3 + distance_threshold_offset, final_position
        )
    elif args.max_start_distance.lower() == "hard":
        if args.task_type == "via_alt_goal":
            distance_threshold_offset += 3

        start_index = closest_state(
            sim, agent_states, 5 + distance_threshold_offset, final_position
        )
    elif args.max_start_distance.lower() == "full":
        start_index = 0 if not args.reverse else len(agent_states) - 1
    else:
        raise NotImplementedError(
            f"max start distance: {args.max_start_distance} is not an available start."
        )
    start_state = agent_states[start_index]
    if args.reverse:
        start_state = set_start_state_reverse_orientation(agent_states, start_index)
    return start_state


def save_dict(full_save_path, config_dict):
    with open(full_save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def get_semantic_filters(sim, traversable_class_names, cull_categories):
    # setup is/is not traversable and which goals are banned (for the simulator runs)
    instance_index_to_name_map = utils.get_instance_index_to_name_mapping(
        sim.semantic_scene
    )
    traversable_class_indices = instance_index_to_name_map[:, 0][
        np.isin(instance_index_to_name_map[:, 1], traversable_class_names)
    ]
    traversable_class_indices = np.unique(traversable_class_indices).astype(int)
    bad_goal_categories = ["ceiling", "ceiling lower"]
    bad_goal_cat_idx = instance_index_to_name_map[:, 0][
        np.isin(instance_index_to_name_map[:, 1], bad_goal_categories)
    ]
    bad_goal_classes = np.unique(bad_goal_cat_idx).astype(int)

    cull_instance_ids = (
        instance_index_to_name_map[:, 0][
            np.isin(instance_index_to_name_map[:, 1], cull_categories)
        ]
    ).astype(int)
    return traversable_class_indices, bad_goal_classes, cull_instance_ids
