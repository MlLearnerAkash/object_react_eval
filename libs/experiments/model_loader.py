import torch
from pathlib import Path

# ignore FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="contextlib")


def get_depth_model():
    from libs.depth.depth_anything_metric_model import DepthAnythingMetricModel

    depth_model_name = 'zoedepth'
    path_zoe_depth = Path.cwd() / 'model_weights' / \
        'depth_anything_metric_depth_indoor.pt'
    if not path_zoe_depth.exists():
        raise FileNotFoundError(f'{path_zoe_depth} not found...')
    depth_model = DepthAnythingMetricModel(
        depth_model_name, pretrained_resource=str(path_zoe_depth))
    return depth_model


def get_controller_model(method, goal_source, config_filepath):
    goal_controller = None
    if method == 'learnt':
        from libs.control.objectreact import ObjRelLearntController
        goal_controller = ObjRelLearntController(
            config_filepath, goal_source=goal_source)
    return goal_controller


def get_segmentor(segmentor_name, image_width, image_height, device=None,
                  path_models=None, traversable_class_names=None):
    if device is None:
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

    segmentor = None

    if segmentor_name == 'sam':
        from libs.segmentor import sam

        segmentor = sam.Seg_SAM(
            path_models, device,
            resize_w=image_width,
            resize_h=image_height
        )

    elif segmentor_name == 'fast_sam':
        from libs.segmentor import fast_sam_module

        segmentor = fast_sam_module.FastSamClass(
            {'width': image_width, 'height': image_height,
             'mask_height': image_height, 'mask_width': image_width,
             'conf': 0.5, 'model': 'FastSAM-s.pt',
             'imgsz': int(max(image_height, image_width, 480))},
            device=device, traversable_categories=traversable_class_names
        )  # imgsz < 480 gives poorer results

    elif segmentor_name == 'sam2':
        from libs.segmentor import sam2_seg
        assert path_models is not None, f'{path_models=} must be provided for {segmentor_name=}!'
        segmentor = sam2_seg.Seg_SAM2(
            model_checkpoint=path_models, resize_w=image_width, resize_h=image_height)

    elif 'sam21' in segmentor_name:
        sam_kwargs = {}
        if 'pps' in segmentor_name:
            sam_kwargs = {"points_per_side": int(
                segmentor_name.split("_")[-1][3:])}
        from libs.segmentor import sam21
        segmentor = sam21.Seg_SAM21(
            resize_w=image_width, resize_h=image_height, sam_kwargs=sam_kwargs)

    elif segmentor_name == 'sim':
        raise ValueError(
            'Simulator segments not supported in topological mode...')

    else:
        raise NotImplementedError(f'{segmentor_name=} not implemented...')

    return segmentor


def get_joint_models(joint_checkpoint_path, device=None):
    """Load LangGeoNetV2 + GNM (joint checkpoint) for lang_e3d goal source.

    Returns a dict with keys:
        lange3d      : LangGeoNetV2 model (eval mode, on device)
        gnm_joint    : GNM model (eval mode, on device)
        topopaths    : TopoPaths instance
        clip_processor : CLIPProcessor
    """
    import sys
    import warnings
    from pathlib import Path as _Path

    TRAIN_DIR = _Path("/data/ws/VLN-CE/controller/object_react/train")
    for _p in [str(TRAIN_DIR), str(TRAIN_DIR / "lange3dnet_train")]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    # Install mock _magnum so graphs pickled with Habitat C++ types can be
    # loaded without the native extension.
    from joint_dataset import _install_mock_magnum
    _install_mock_magnum()

    from vint_train.models.gnm.gnm import GNM
    from vint_train.models.object_react.dataloader import TopoPaths
    from model import LangGeoNetV2
    from transformers import CLIPProcessor

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Architecture must match training (MODEL_CFG in joint_val_app.py)
    lange3d = LangGeoNetV2(
        d_model=256, n_heads=8, n_layers=2,
        clip_model_name="openai/clip-vit-base-patch16",
        dino_model_name="facebook/dinov2-small",
        freeze_clip=True, freeze_dino=True,
    ).to(device)

    gnm = GNM(
        context_size=5, len_traj_pred=10, learn_angle=True,
        obs_encoding_size=1024, goal_encoding_size=1024,
        goal_type="image_mask_enc", obs_type="disabled",
        dims=8, use_mask_grad=False, goal_uses_context=False,
    ).to(device)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ckpt = torch.load(joint_checkpoint_path, map_location=device,
                          weights_only=False)

    lange3d.load_state_dict(ckpt["lange3d"], strict=False)
    gnm.load_state_dict(ckpt["gnm"], strict=False)
    lange3d.eval()
    gnm.eval()
    print(f"[get_joint_models] loaded epoch {ckpt.get('epoch', '?')} "
          f"val_spearman={ckpt.get('val_spearman', float('nan')):.4f}")

    topopaths = TopoPaths(dims=8, w=160, h=120)
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch16")

    return {
        "lange3d": lange3d,
        "gnm_joint": gnm,
        "topopaths": topopaths,
        "clip_processor": clip_processor,
    }
