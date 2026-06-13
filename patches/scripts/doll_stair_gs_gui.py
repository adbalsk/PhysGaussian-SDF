"""
doll_stair_gs_gui.py — real-time interactive 3DGS GUI
布娃娃（warp MPM）滚下楼梯，diff-gaussian-rasterization 实时渲染，cv2 交互窗口。

两种渲染模式（--render hybrid|gaussian，默认 hybrid）：
  hybrid   : pyrender 渲染立方体房间+楼梯（真实 Phong 光照），逐像素深度合成高斯布娃娃
  gaussian : 楼梯烘焙成各向异性扁盘高斯，单 pass 光栅化（无 pyrender）

两种物理模式（--mode live|playback）：
  live     : 每帧实时跑 MPM，substeps/帧可运行时调节
  playback : 回放预录轨迹 npz

Controls:  LMB drag=orbit  wheel=zoom  SPACE=pause  R=reset  [ / ]=substeps
           L=lighting  S=save PNG  Alt+LMB=kick doll  ESC/q=quit

Usage:
  python scripts/doll_stair_gs_gui.py --model_path ./model/doll.ply \\
      --config ./config/doll_stair_config.json
  python scripts/doll_stair_gs_gui.py --mode playback --traj ./output/doll_traj.npz \\
      --model_path ./model/doll.ply --config ./config/doll_stair_config.json
"""

import sys, os, math, time, json, argparse

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "gaussian-splatting"))

import numpy as np
import cv2
import torch
import warp as wp
wp.init()

from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.cameras import Camera as GSCamera
from utils.graphics_utils import focal2fov
from utils.system_utils import searchForMaxIteration
from utils.sh_utils import eval_sh

from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from utils.decode_param import decode_param_json, set_boundary_conditions
from utils.transformation_utils import (
    generate_rotation_matrices, apply_rotations, apply_cov_rotations,
    transform2origin, shift2center111,
    apply_inverse_rotations, apply_inverse_cov_rotations,
    undotransform2origin, undoshift2center111,
    get_center_view_worldspace_and_observant_coordinate,
)
from utils.camera_view_utils import get_camera_position_and_rotation
from utils.render_utils import load_params_from_gs, convert_SH
from utils.mesh_sdf import load_obj, bake_sdf_mpm_grid
from utils.mesh_to_gs import bake_mesh_to_gaussians


def _rot_matrix(degree, axis):
    th = math.radians(degree)
    c, s = math.cos(th), math.sin(th)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def place_stair_in_sim_space(verts, sp):
    v = verts.astype(np.float64).copy()
    for deg, ax in zip(sp.get("rotation_degree", []), sp.get("rotation_axis", [])):
        v = v @ _rot_matrix(deg, ax).T
    center = (v.min(0) + v.max(0)) * 0.5
    max_ext = float((v.max(0) - v.min(0)).max())
    scale = (float(sp["fit_size"]) / max_ext
             if sp.get("fit_size") is not None else float(sp.get("scale", 1.0)))
    translate = np.asarray(sp.get("translate", [1.0, 1.0, 1.0]), dtype=np.float64)
    return ((v - center) * scale + translate).astype(np.float32)


def _detect_sh_degree(ply_path):
    from plyfile import PlyData
    el = PlyData.read(ply_path).elements[0]
    n_rest = sum(1 for p in el.properties if p.name.startswith("f_rest_"))
    K = (n_rest + 3) // 3
    return int(round(math.sqrt(K))) - 1


def load_doll_gaussians(model_path):
    ply_path = None
    if model_path.endswith(".ply"):
        ply_path = model_path
    else:
        ckpt = os.path.join(model_path, "point_cloud")
        if os.path.isdir(ckpt):
            it = searchForMaxIteration(ckpt)
            ply_path = os.path.join(ckpt, f"iteration_{it}", "point_cloud.ply")
        else:
            plys = [f for f in os.listdir(model_path) if f.endswith(".ply")]
            if plys:
                ply_path = os.path.join(model_path, plys[0])
    if ply_path is None or not os.path.exists(ply_path):
        raise FileNotFoundError(f"No GS .ply at {model_path}")
    sh = _detect_sh_degree(ply_path)
    print(f"  doll ply: {ply_path} (sh_degree={sh})")
    g = GaussianModel(sh)
    g.load_ply(ply_path)
    return g


def build_orbit_camera(az_deg, el_deg, radius, center_world, observant,
                       width, height, fov_deg=50.0):
    """(azimuth, elevation, radius) -> GSCamera, mirrors gs_simulation_stair.py."""
    fov = math.radians(fov_deg)
    focal = 0.5 * width / math.tan(0.5 * fov)
    position, Rmat = get_camera_position_and_rotation(
        az_deg, el_deg, radius, center_world, observant)
    tmp = np.zeros((4, 4))
    tmp[:3, :3] = Rmat
    tmp[:3, 3] = position
    tmp[3, 3] = 1
    C2W = np.linalg.inv(tmp)
    Rc = C2W[:3, :3].transpose()
    Tc = C2W[:3, 3]
    return GSCamera(colmap_id=0, R=Rc, T=Tc,
                    FoVx=focal2fov(focal, width), FoVy=focal2fov(focal, height),
                    image=torch.zeros((3, height, width)), gt_alpha_mask=None,
                    image_name="orbit", uid=0)


def _make_spotlight_pose(pos_world, target_world):
    """OpenGL-convention 4x4 pose for a spotlight at pos looking at target."""
    pos = np.asarray(pos_world, dtype=np.float64)
    tgt = np.asarray(target_world, dtype=np.float64)
    z = pos - tgt
    z /= (np.linalg.norm(z) + 1e-12)
    up = np.array([0., 0., 1.]) if abs(z[2]) < 0.99 else np.array([0., 1., 0.])
    x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x; pose[:3, 1] = y; pose[:3, 2] = z; pose[:3, 3] = pos
    return pose


def build_room_scene(stair_v_world, stair_f, scene_center, scene_radius,
                     room_color=(0.72, 0.72, 0.74), stair_color=(0.55, 0.56, 0.60)):
    """pyrender Scene = stair mesh only (no room box); white background."""
    import pyrender, trimesh
    scene = pyrender.Scene(bg_color=[1., 1., 1., 1.], ambient_light=[0.28, 0.28, 0.30])
    smesh = trimesh.Trimesh(vertices=stair_v_world, faces=stair_f.astype(np.int32), process=False)
    smesh.fix_normals()
    stair_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*stair_color, 1.0],
        metallicFactor=0.0, roughnessFactor=0.6, doubleSided=True, smooth=False)
    scene.add(pyrender.Mesh.from_trimesh(smesh, material=stair_mat, smooth=False), name="stair")
    return scene, None


def add_scene_lights(scene, mode, scene_center, scene_radius):
    """Add spot (mode='spot') or 3-dir (mode='flat') lights; return node list."""
    import pyrender
    nodes = []
    c = np.asarray(scene_center, dtype=np.float64)
    r = float(scene_radius)
    if mode == "spot":
        lpos = c + np.array([0.6 * r, -1.3 * r, 2.0 * r])
        spot = pyrender.SpotLight(color=[1.0, 0.98, 0.95], intensity=18.0 * r * r,
                                  innerConeAngle=np.pi / 7.0, outerConeAngle=np.pi / 3.2,
                                  range=r * 12.0)
        nodes.append(scene.add(spot, pose=_make_spotlight_pose(lpos, c), name="spot"))
    else:
        def _dir_pose(d):
            z = -np.asarray(d, dtype=np.float64)
            z /= np.linalg.norm(z) + 1e-12
            up = np.array([0., 0., 1.]) if abs(z[2]) < 0.99 else np.array([0., 1., 0.])
            x = np.cross(up, z); x /= np.linalg.norm(x) + 1e-12
            y = np.cross(z, x)
            m = np.eye(4, dtype=np.float32)
            m[:3, 0] = x; m[:3, 1] = y; m[:3, 2] = z
            return m
        for d, inten, col in [([0.6, -0.8, 1.2], 3.6, [1.0, 1.0, 0.97]),
                               ([-0.8, -0.4, 0.7], 1.6, [0.7, 0.78, 1.0]),
                               ([-0.4, 0.9, -0.4], 0.7, [1.0, 0.95, 0.9])]:
            nodes.append(scene.add(pyrender.DirectionalLight(color=col, intensity=inten),
                                   pose=_dir_pose(d), name="dir"))
    return nodes


def render_background(scene, renderer, gs_cam):
    """Render mesh scene from gs_cam; returns (rgb[H,W,3] float32, depth[H,W] float32 m).
    Uses RenderFlags.NONE — SHADOWS_SPOT crashes glGenTextures on this PyOpenGL stack."""
    import pyrender
    H, W = gs_cam.image_height, gs_cam.image_width
    fx = 0.5 * W / math.tan(0.5 * gs_cam.FoVx)
    fy = 0.5 * H / math.tan(0.5 * gs_cam.FoVy)
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=W / 2.0, cy=H / 2.0,
                                    znear=0.02, zfar=1000.0)
    W2C = np.eye(4, dtype=np.float64)
    W2C[:3, :3] = gs_cam.R.T
    W2C[:3, 3] = gs_cam.T
    C2W_gl = (np.linalg.inv(W2C) @ np.diag([1., -1., -1., 1.])).astype(np.float32)
    cam_node = scene.add(cam, pose=C2W_gl)
    color_u8, depth = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
    scene.remove_node(cam_node)
    return color_u8.astype(np.float32) / 255.0, depth.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./model/doll.ply")
    parser.add_argument("--config", type=str, default="./config/doll_stair_config.json")
    parser.add_argument("--mode", choices=["live", "playback"], default="live")
    parser.add_argument("--traj", type=str, default=None)
    parser.add_argument("--record", type=str, default=None,
                        help="record world-space doll trajectory to npz (live only)")
    parser.add_argument("--no-physics", action="store_true")
    parser.add_argument("--no-light", action="store_true", help="start with flat stair color")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--substeps", type=int, default=25,
                        help="substeps per displayed frame (live)")
    parser.add_argument("--substep_dt", type=float, default=None,
                        help="override config substep_dt (validated stable up to ~5e-4)")
    parser.add_argument("--render", choices=["hybrid", "gaussian"], default="hybrid",
                        help="hybrid: pyrender room+stairs depth-composited with gaussian doll; "
                             "gaussian: all-gaussian single pass")
    parser.add_argument("--selftest", type=int, default=0,
                        help="headless: step N frames, save PNG, exit (validation)")
    args = parser.parse_args()

    device = "cuda:0"
    W, H = args.width, args.height

    (material_params, bc_params, time_params,
     preprocessing_params, camera_params) = decode_param_json(args.config)
    with open(args.config) as f:
        _cfg = json.load(f)
    sp = _cfg.get("stair_params", {})
    fp = _cfg.get("floor_params", {})
    n_grid = material_params["n_grid"]
    grid_lim = material_params["grid_lim"]

    print("Loading doll gaussians...")
    gaussians = load_doll_gaussians(args.model_path)
    pipeline = type('P', (), {'convert_SHs_python': False,
                              'compute_cov3D_python': True, 'debug': False})()
    params = load_params_from_gs(gaussians, pipeline)
    init_pos, init_opacity = params["pos"], params["opacity"]
    init_shs = params["shs"]

    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask]
    init_opacity = init_opacity[mask]
    init_cov = params["cov3D_precomp"][mask]
    init_shs = init_shs[mask]

    rotm = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"])
    rp = apply_rotations(init_pos, rotm)
    tp, scale_origin, omp = transform2origin(rp, preprocessing_params["scale"])
    tp = shift2center111(tp)
    init_cov = scale_origin * scale_origin * apply_cov_rotations(init_cov, rotm)
    gs_num = tp.shape[0]
    mpm_init_pos = tp.to(device)
    mpm_init_cov = torch.zeros((gs_num, 6), device=device)
    mpm_init_cov[:gs_num] = init_cov
    print(f"  doll: {gs_num} particles")

    obj = sp.get("obj", "./model/stair.obj")
    if not os.path.isabs(obj):
        obj = os.path.join(_root, obj)
    print(f"Loading stair: {obj}")
    sv_mesh, sf = load_obj(obj)
    sv_sim = place_stair_in_sim_space(sv_mesh, sp)
    print("Baking stair SDF...")
    stair_sdf = bake_sdf_mpm_grid(sv_sim, sf, n_grid=n_grid, grid_lim=grid_lim, verbose=True)

    sv_sim_t = torch.tensor(sv_sim, device=device)
    stair_v_world = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(sv_sim_t), scale_origin, omp),
        rotm).detach().cpu().numpy()

    # floor geometry in world space (a flat quad at floor_z in sim space)
    floor_z_sim = float(fp.get("z", 0.10))
    floor_half = float(fp.get("half_extent", 1.8))
    floor_col = tuple(fp.get("color", [0.82, 0.82, 0.80]))
    _cx, _cy = 1.0, 1.0  # floor XY center in sim space
    _fv_sim = np.array([
        [_cx - floor_half, _cy - floor_half, floor_z_sim],
        [_cx + floor_half, _cy - floor_half, floor_z_sim],
        [_cx + floor_half, _cy + floor_half, floor_z_sim],
        [_cx - floor_half, _cy + floor_half, floor_z_sim],
    ], dtype=np.float32)
    _ff = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    _fv_sim_t = torch.tensor(_fv_sim, device=device)
    floor_v_world = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(_fv_sim_t), scale_origin, omp),
        rotm).detach().cpu().numpy()

    if args.render == "gaussian":
        print("Baking stair static gaussians (anisotropic + lit)...")
        stair_gs = bake_mesh_to_gaussians(
            sv_sim, sf, sh_degree=gaussians.max_sh_degree,
            color=tuple(sp.get("color", [0.55, 0.56, 0.60])),
            samples_per_area=sp.get("samples_per_area", 8000.0),
            gaussian_scale=sp.get("gaussian_scale"),
            opacity=sp.get("opacity", 0.995),
            device=device, sampling="stratified", anisotropic=True,
            flatten_ratio=0.15, ambient=0.4)
        stair_pos_world = apply_inverse_rotations(
            undotransform2origin(undoshift2center111(stair_gs["pos"]), scale_origin, omp), rotm)
        stair_cov_world = apply_inverse_cov_rotations(
            stair_gs["cov3D"] / (scale_origin * scale_origin), rotm)
        stair_opacity = stair_gs["opacity"]
        stair_col_lit = stair_gs["colors_lit"]
        stair_col_flat = stair_gs["colors_flat"]
        # bake floor quad as gaussians and concatenate with stair
        print("Baking floor gaussians...")
        floor_gs = bake_mesh_to_gaussians(
            _fv_sim, _ff, sh_degree=gaussians.max_sh_degree,
            color=floor_col,
            samples_per_area=fp.get("samples_per_area", 4000.0),
            opacity=fp.get("opacity", 0.995),
            device=device, sampling="stratified", anisotropic=True,
            flatten_ratio=0.08, ambient=0.5)
        floor_pos_world = apply_inverse_rotations(
            undotransform2origin(undoshift2center111(floor_gs["pos"]), scale_origin, omp), rotm)
        floor_cov_world = apply_inverse_cov_rotations(
            floor_gs["cov3D"] / (scale_origin * scale_origin), rotm)
        stair_pos_world = torch.cat([stair_pos_world, floor_pos_world], 0)
        stair_cov_world = torch.cat([stair_cov_world, floor_cov_world], 0)
        stair_opacity = torch.cat([stair_opacity, floor_gs["opacity"]], 0)
        stair_col_lit = torch.cat([stair_col_lit, floor_gs["colors_lit"]], 0)
        stair_col_flat = torch.cat([stair_col_flat, floor_gs["colors_flat"]], 0)
        n_stair = stair_pos_world.shape[0]
    else:
        n_stair = 0
        stair_pos_world = stair_cov_world = stair_opacity = None
        stair_col_lit = stair_col_flat = None

    traj = None
    if args.mode == "playback":
        if args.traj is None or not os.path.exists(args.traj):
            raise FileNotFoundError(f"playback mode needs --traj <npz>; got {args.traj}")
        data = np.load(args.traj)
        traj = {"pos": data["pos"], "cov": data["cov"]}
        print(f"  playback: {traj['pos'].shape[0]} frames from {args.traj}")
        mpm_solver = None
    else:
        print("Initializing MPM solver...")
        dx = grid_lim / n_grid
        cell_vol = dx ** 3
        idx = torch.clamp((mpm_init_pos / dx).long(), 0, n_grid - 1)
        flat = idx[:, 0] * (n_grid * n_grid) + idx[:, 1] * n_grid + idx[:, 2]
        cnts = torch.zeros(n_grid ** 3, device=device).scatter_add(
            0, flat, torch.ones(gs_num, device=device))
        mpm_init_vol = torch.where(cnts > 0, cell_vol / cnts,
                                   torch.zeros_like(cnts))[flat]
        mpm_solver = MPM_Simulator_WARP(10)
        mpm_solver.load_initial_data_from_torch(
            mpm_init_pos, mpm_init_vol, mpm_init_cov, n_grid=n_grid, grid_lim=grid_lim)
        mpm_solver.set_parameters_dict(material_params)
        set_boundary_conditions(mpm_solver, bc_params, time_params)
        mpm_solver.add_sdf_collider(
            stair_sdf, friction=float(sp.get("friction", 0.35)),
            restitution=float(sp.get("restitution", 0.0)),
            threshold=sp.get("threshold", None), device=device)
        mpm_solver.add_surface_collider(
            point=[_cx, _cy, floor_z_sim], normal=[0.0, 0.0, 1.0],
            surface="frictional", friction=float(fp.get("friction", 0.4)))
        mpm_solver.finalize_mu_lam()
        reset_x = mpm_solver.export_particle_x_to_torch().cpu().numpy().copy()
        reset_vol = mpm_solver.mpm_state.particle_vol.numpy().copy()
        reset_cov = mpm_solver.mpm_state.particle_init_cov.numpy().copy()

    mpm_center = torch.tensor(
        camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    mpm_up = torch.tensor(
        camera_params["mpm_space_vertical_upward_axis"]).reshape((1, 3)).cuda()
    center_world, observant = get_center_view_worldspace_and_observant_coordinate(
        mpm_center, mpm_up, rotm, scale_origin, omp)

    bg_color = torch.tensor([1., 1., 1.], dtype=torch.float32, device=device)
    screen_all = torch.zeros((gs_num + n_stair, 3), device=device, requires_grad=True)
    opacity_all = (torch.cat([init_opacity, stair_opacity], 0)
                   if args.render == "gaussian" else init_opacity)
    shs_render = init_shs

    _allw = stair_v_world
    scene_center_np = ((_allw.min(0) + _allw.max(0)) * 0.5).astype(np.float64)
    scene_radius_np = float(np.linalg.norm(_allw.max(0) - _allw.min(0)) * 0.5)

    pr_scene = pr_renderer = None
    light_nodes = []
    if args.render == "hybrid":
        import pyrender, trimesh
        print("Building pyrender stair+floor scene...")
        pr_scene, _ = build_room_scene(
            stair_v_world, sf, scene_center_np, scene_radius_np,
            room_color=tuple(sp.get("room_color", [0.72, 0.72, 0.74])),
            stair_color=tuple(sp.get("color", [0.55, 0.56, 0.60])))
        # add floor plane
        floor_mesh = trimesh.Trimesh(
            vertices=floor_v_world, faces=_ff.astype(np.int32), process=False)
        floor_mesh.fix_normals()
        floor_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[*floor_col, 1.0],
            metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True, smooth=False)
        pr_scene.add(pyrender.Mesh.from_trimesh(floor_mesh, material=floor_mat, smooth=False),
                     name="floor")
        light_nodes = add_scene_lights(
            pr_scene, "spot" if not args.no_light else "flat",
            scene_center_np, scene_radius_np)
        pr_renderer = pyrender.OffscreenRenderer(W, H)

    substep_dt = args.substep_dt if args.substep_dt else time_params["substep_dt"]
    print(f"  substep_dt = {substep_dt:.1e}  "
          f"(physics-speed ceiling ~{substep_dt / 0.0006 * 100:.0f}% realtime)")

    state = {
        "az": float(camera_params.get("init_azimuthm", 90)),
        "el": float(camera_params.get("init_elevation", 8)),
        "radius": float(camera_params.get("init_radius", 3.0)),
        "dragging": False, "last_x": 0, "last_y": 0,
        "paused": args.no_physics, "lit": not args.no_light,
        "spf": max(1, args.substeps), "reset": False, "save": False, "quit": False,
        "kick": False, "kick_strength": 3.0,
    }

    def on_mouse(event, x, y, flags, _):
        alt_held = bool(flags & cv2.EVENT_FLAG_ALTKEY)
        if event == cv2.EVENT_LBUTTONDOWN and alt_held:
            # Alt+LMB: kick doll along camera forward direction (sim space)
            if args.mode == "live" and not args.no_physics:
                state["kick"] = True
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["last_x"], state["last_y"] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["az"] -= (x - state["last_x"]) * 0.4
            state["el"] = max(-89.0, min(89.0, state["el"] + (y - state["last_y"]) * 0.4))
            state["last_x"], state["last_y"] = x, y
        elif event == cv2.EVENT_MOUSEWHEEL:
            state["radius"] = max(0.5, min(20.0, state["radius"] + (-0.3 if flags > 0 else 0.3)))

    win = "3DGS Sim"
    if not args.selftest:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, on_mouse)

    rec_pos, rec_cov = ([], []) if args.record else (None, None)
    play_idx = 0

    def get_doll_world():
        nonlocal play_idx
        if args.mode == "playback":
            i = min(play_idx, traj["pos"].shape[0] - 1)
            return (torch.tensor(traj["pos"][i], device=device),
                    torch.tensor(traj["cov"][i], device=device), None)
        pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
        cov = mpm_solver.export_particle_cov_to_torch().view(-1, 6)[:gs_num].to(device)
        rot = mpm_solver.export_particle_R_to_torch().view(-1, 3, 3)[:gs_num].to(device)
        pos = apply_inverse_rotations(
            undotransform2origin(undoshift2center111(pos), scale_origin, omp), rotm)
        cov = apply_inverse_cov_rotations(cov / (scale_origin * scale_origin), rotm)
        return pos, cov, rot

    def _make_rast(cam, bg):
        rs = GaussianRasterizationSettings(
            image_height=H, image_width=W,
            tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg, scale_modifier=1.0,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            sh_degree=gaussians.active_sh_degree, campos=cam.camera_center,
            prefiltered=False, debug=False)
        return GaussianRasterizer(raster_settings=rs)

    def render_gaussian(cam):
        pos, cov, rot = get_doll_world()
        doll_col = convert_SH(shs_render, cam, gaussians, pos, rot)
        stair_col = stair_col_lit if state["lit"] else stair_col_flat
        all_pos = torch.cat([pos, stair_pos_world], 0)
        all_cov = torch.cat([cov, stair_cov_world], 0)
        all_col = torch.cat([doll_col, stair_col], 0)
        rast = _make_rast(cam, bg_color)
        img = rast(means3D=all_pos, means2D=screen_all, shs=None,
                   colors_precomp=all_col, opacities=opacity_all,
                   scales=None, rotations=None, cov3D_precomp=all_cov)[0]
        rgb = img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        return (rgb * 255).astype(np.uint8)[:, :, ::-1].copy()

    znear_d = max(0.05, scene_radius_np * 0.2)
    zfar_d = scene_radius_np * 6.0 + float(camera_params.get("init_radius", 3.0))

    def render_hybrid(cam):
        """doll 3-pass (color/alpha/depth) + pyrender background, depth-aware composite."""
        pos, cov, rot = get_doll_world()
        doll_col = convert_SH(shs_render, cam, gaussians, pos, rot)
        screen_doll = screen_all[:gs_num]
        common = dict(means2D=screen_doll, shs=None, opacities=init_opacity,
                      scales=None, rotations=None, cov3D_precomp=cov)
        # render doll against black and white to extract alpha
        rast_b = _make_rast(cam, torch.zeros(3, device=device))
        rast_w = _make_rast(cam, torch.ones(3, device=device))
        doll_b = rast_b(means3D=pos, colors_precomp=doll_col, **common)[0]
        doll_w = rast_w(means3D=pos, colors_precomp=doll_col, **common)[0]
        db = doll_b.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        dw = doll_w.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        alpha = 1.0 - np.clip(dw - db, 0, 1).mean(-1, keepdims=True)

        # depth as view-space z along optical axis (not euclidean distance)
        campos = cam.camera_center.to(device)
        fwd = torch.tensor(center_world, dtype=torch.float32, device=device) - campos
        fwd = fwd / (fwd.norm() + 1e-8)
        viewz = ((pos - campos.unsqueeze(0)) * fwd.unsqueeze(0)).sum(dim=1, keepdim=True)
        depth_col = ((viewz - znear_d) / (zfar_d - znear_d)).clamp(0, 1).repeat(1, 3)
        doll_d = rast_b(means3D=pos, colors_precomp=depth_col, **common)[0]
        dd = doll_d.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()[..., 0]
        doll_depth_m = znear_d + dd * (zfar_d - znear_d)

        bg_rgb, bg_depth = render_background(pr_scene, pr_renderer, cam)
        # pyrender bg is white; pixels with no mesh hit have depth=0 -> treat as far
        bg_depth_cmp = np.where(bg_depth > 1e-6, bg_depth, 1e9)
        doll_vis3 = ((alpha[..., 0] > 0.5) & (doll_depth_m < bg_depth_cmp))[..., None].astype(np.float32)
        a = alpha * doll_vis3
        out = np.clip(db * a + bg_rgb * (1.0 - a), 0, 1)
        return (out * 255).astype(np.uint8)[:, :, ::-1].copy()

    def render(cam):
        return render_hybrid(cam) if args.render == "hybrid" else render_gaussian(cam)

    print(f"\nMode: {args.mode}  |  {gs_num + n_stair} gaussians  |  render: {args.render}")
    print("LMB drag=orbit  wheel=zoom  SPACE=pause  R=reset  [ / ]=substeps")
    print("L=lighting  S=save PNG  Alt+LMB=kick doll  ESC/q=quit\n")

    sim_frame = 0
    fps_ema = 0.0
    while not state["quit"]:
        t0 = time.time()

        if args.mode == "live" and not state["paused"] and not args.no_physics:
            for _ in range(state["spf"]):
                mpm_solver.p2g2p(sim_frame, substep_dt, device=device)
            sim_frame += 1
            if args.record:
                p, c, _ = get_doll_world()
                rec_pos.append(p.detach().cpu().numpy().astype(np.float32))
                rec_cov.append(c.detach().cpu().numpy().astype(np.float32))
        elif args.mode == "playback" and not state["paused"]:
            play_idx = min(play_idx + 1, traj["pos"].shape[0] - 1)

        if state["reset"] and args.mode == "live":
            mpm_solver.load_initial_data_from_torch(
                torch.tensor(reset_x, device=device),
                torch.tensor(reset_vol, device=device),
                torch.tensor(reset_cov.reshape(-1, 6), device=device),
                n_grid=n_grid, grid_lim=grid_lim, device=device)
            mpm_solver.set_parameters_dict(material_params)
            set_boundary_conditions(mpm_solver, bc_params, time_params)
            mpm_solver.add_sdf_collider(
                stair_sdf, friction=float(sp.get("friction", 0.35)),
                restitution=float(sp.get("restitution", 0.0)),
                threshold=sp.get("threshold", None), device=device)
            mpm_solver.add_surface_collider(
                point=[_cx, _cy, floor_z_sim], normal=[0.0, 0.0, 1.0],
                surface="frictional", friction=float(fp.get("friction", 0.4)))
            mpm_solver.finalize_mu_lam()
            sim_frame = 0
            state["reset"] = False
            print("Reset.")

        if state["kick"] and args.mode == "live" and mpm_solver is not None:
            # compute camera forward in world space, then transform to sim (MPM) space
            cam_tmp = build_orbit_camera(state["az"], state["el"], state["radius"],
                                         center_world, observant, W, H)
            campos_w = cam_tmp.camera_center.cpu().numpy()           # [3] world
            center_w = np.asarray(center_world, dtype=np.float64).flatten()
            fwd_w = center_w - campos_w
            fwd_w = fwd_w / (np.linalg.norm(fwd_w) + 1e-8)          # unit forward

            # world -> sim: apply rotation (rotm), then scale+shift are isotropic so
            # direction only needs the rotation part
            # rotm is a list of (degree, axis) tuples already applied via generate_rotation_matrices
            fwd_t = torch.tensor(fwd_w, dtype=torch.float32, device=device).unsqueeze(0)
            fwd_sim = apply_rotations(fwd_t, rotm).squeeze(0) * float(scale_origin)
            fwd_sim = fwd_sim / (fwd_sim.norm() + 1e-8)

            v_cur = mpm_solver.export_particle_v_to_torch()          # [N, 3]
            v_new = v_cur + fwd_sim.unsqueeze(0) * state["kick_strength"]
            mpm_solver.import_particle_v_from_torch(v_new, device=device)
            print(f"Kick! dir_sim={fwd_sim.cpu().numpy().round(3)}  "
                  f"strength={state['kick_strength']:.1f}")
            state["kick"] = False

        cam = build_orbit_camera(state["az"], state["el"], state["radius"],
                                 center_world, observant, W, H)
        frame = render(cam)

        dt = time.time() - t0
        fps_ema = 1.0 / max(dt, 1e-4) if fps_ema == 0 else 0.9 * fps_ema + 0.1 / max(dt, 1e-4)
        hud_parts = [f"FPS {fps_ema:4.1f}", args.render,
                     f"spf={state['spf']} frame={sim_frame}" if args.mode == "live"
                     else f"play={play_idx}/{traj['pos'].shape[0]-1}",
                     "PAUSED" if state["paused"] else "running",
                     f"light={'spot' if state['lit'] else 'flat'}" if args.render == "hybrid"
                     else f"light={'on' if state['lit'] else 'off'}"]
        cv2.putText(frame, "  ".join(hud_parts), (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 255, 120), 1, cv2.LINE_AA)

        if state["save"]:
            outp = os.path.join(_root, "output", f"gui_frame_{int(time.time())}.png")
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            cv2.imwrite(outp, frame)
            print(f"Saved {outp}")
            state["save"] = False

        if args.selftest:
            if sim_frame >= args.selftest or play_idx >= args.selftest or args.no_physics:
                outp = os.path.join(_root, "output", "gui_selftest.png")
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                cv2.imwrite(outp, frame)
                print(f"SELFTEST_OK {outp}  fps={fps_ema:.1f}")
                state["quit"] = True
            continue

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            state["quit"] = True
        elif k == ord(' '):
            state["paused"] = not state["paused"]
        elif k == ord('r'):
            state["reset"] = True
        elif k == ord('['):
            state["spf"] = max(1, state["spf"] - 5)
        elif k == ord(']'):
            state["spf"] = min(400, state["spf"] + 5)
        elif k == ord('l'):
            state["lit"] = not state["lit"]
            if args.render == "hybrid":
                for nd in light_nodes:
                    pr_scene.remove_node(nd)
                light_nodes = add_scene_lights(
                    pr_scene, "spot" if state["lit"] else "flat",
                    scene_center_np, scene_radius_np)
        elif k == ord('s'):
            state["save"] = True
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            state["quit"] = True

    if not args.selftest:
        cv2.destroyAllWindows()

    if args.record and rec_pos:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        np.savez_compressed(args.record, pos=np.stack(rec_pos), cov=np.stack(rec_cov))
        print(f"Recorded {len(rec_pos)} frames -> {args.record}")

    os._exit(0)


if __name__ == "__main__":
    main()
