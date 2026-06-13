"""
Local GGUI Demo: spheres tumbling down an SDF staircase, with mouse-drag camera.

Run locally (needs a GPU + Vulkan for Taichi GGUI):
  python scripts/falling_spheres_gui.py
  python scripts/falling_spheres_gui.py --num-spheres 80 --friction 0.0

Controls:
  Left mouse drag : orbit camera
  Mouse wheel     : (not bound) -- use --cam-dist
  R               : reset spheres
  SPACE           : pause / resume
  ESC             : quit
"""

import sys, os, math, argparse
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.mesh_sdf import load_obj

import taichi as ti

ti.init(arch=ti.gpu, random_seed=int.from_bytes(os.urandom(4), 'little') & 0x7FFFFFFF)

_MAX_TRIS = 10000
_MAX_SPHERES = 1000
_MAX_GRID = 129

# ---- mesh triangles ----
n_faces = ti.field(ti.i32, ())
tri_v0 = ti.Vector.field(3, ti.f32, shape=_MAX_TRIS)
tri_v1 = ti.Vector.field(3, ti.f32, shape=_MAX_TRIS)
tri_v2 = ti.Vector.field(3, ti.f32, shape=_MAX_TRIS)

# ---- SDF grid ----
R = ti.field(ti.i32, ())
grid_min = ti.Vector.field(3, ti.f32, ())
grid_max = ti.Vector.field(3, ti.f32, ())
sdf = ti.field(ti.f32, shape=(_MAX_GRID, _MAX_GRID, _MAX_GRID))

# ---- spheres ----
n_spheres = ti.field(ti.i32, ())
sphere_pos = ti.Vector.field(3, ti.f32, shape=_MAX_SPHERES)
sphere_vel = ti.Vector.field(3, ti.f32, shape=_MAX_SPHERES)
sphere_color = ti.Vector.field(3, ti.f32, shape=_MAX_SPHERES)
sphere_radius = ti.field(ti.f32, ())

# ---- physics params ----
gravity = ti.Vector.field(3, ti.f32, ())
restitution = ti.field(ti.f32, ())
friction_coeff = ti.field(ti.f32, ())
ground_z = ti.field(ti.f32, ())

# ---- spawn region ----
spawn_lo = ti.Vector.field(3, ti.f32, ())
spawn_hi = ti.Vector.field(3, ti.f32, ())
spawn_vy = ti.field(ti.f32, ())

# PLACEHOLDER_KERNELS

# ===================================================================
# Triangle distance + ray-triangle intersection
# ===================================================================

@ti.func
def _closest_on_edge(p: ti.math.vec3, p0: ti.math.vec3, p1: ti.math.vec3) -> ti.math.vec3:
    e = p1 - p0
    ee = e.dot(e)
    res = p0
    if ee >= 1e-15:
        t = (p - p0).dot(e) / ee
        t = ti.max(0.0, ti.min(1.0, t))
        res = p0 + t * e
    return res


@ti.func
def point_triangle_distance(p: ti.math.vec3, a: ti.math.vec3,
                            b: ti.math.vec3, c: ti.math.vec3):
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1)
    dot11 = v1.dot(v1)
    denom = dot00 * dot11 - dot01 * dot01

    best_q = a
    best_d2 = v2.dot(v2)
    need_fallback = 1
    if denom >= 1e-15:
        dot02 = v2.dot(v0)
        dot12 = v2.dot(v1)
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom
        if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
            q = a + u * v0 + v * v1
            best_q = q
            best_d2 = (p - q).dot(p - q)
            need_fallback = 0

    if need_fallback:
        q_ab = _closest_on_edge(p, a, b)
        d2_ab = (p - q_ab).dot(p - q_ab)
        q_bc = _closest_on_edge(p, b, c)
        d2_bc = (p - q_bc).dot(p - q_bc)
        q_ca = _closest_on_edge(p, c, a)
        d2_ca = (p - q_ca).dot(p - q_ca)
        best_q, best_d2 = q_ab, d2_ab
        if d2_bc < best_d2:
            best_q, best_d2 = q_bc, d2_bc
        if d2_ca < best_d2:
            best_q, best_d2 = q_ca, d2_ca
        if v2.dot(v2) < best_d2:
            best_q, best_d2 = a, v2.dot(v2)
        if (p - b).dot(p - b) < best_d2:
            best_q, best_d2 = b, (p - b).dot(p - b)
        if (p - c).dot(p - c) < best_d2:
            best_q, best_d2 = c, (p - c).dot(p - c)
    return best_q, best_d2


@ti.func
def ray_triangle_hit(orig: ti.math.vec3, dir: ti.math.vec3,
                     a: ti.math.vec3, b: ti.math.vec3, c: ti.math.vec3) -> ti.i32:
    eps = 1e-9
    e1 = b - a
    e2 = c - a
    pvec = dir.cross(e2)
    det = e1.dot(pvec)
    hit = 0
    if det > eps or det < -eps:
        inv_det = 1.0 / det
        tvec = orig - a
        u = tvec.dot(pvec) * inv_det
        if u >= 0.0 and u <= 1.0:
            qvec = tvec.cross(e1)
            v = dir.dot(qvec) * inv_det
            if v >= 0.0 and u + v <= 1.0:
                t = e2.dot(qvec) * inv_det
                if t > eps:
                    hit = 1
    return hit


@ti.kernel
def compute_sdf_kernel():
    r = R[None]
    gmin = grid_min[None]
    gmax = grid_max[None]
    d = (gmax - gmin) / r
    nf = n_faces[None]
    for i, j, k in ti.ndrange(r, r, r):
        p = gmin + ti.Vector([float(i), float(j), float(k)]) * d
        best_d2 = 1e20
        cx = 0
        cy = 0
        cz = 0
        for fi in range(nf):
            a = tri_v0[fi]
            b = tri_v1[fi]
            c = tri_v2[fi]
            q, d2 = point_triangle_distance(p, a, b, c)
            if d2 < best_d2:
                best_d2 = d2
            cx += ray_triangle_hit(p, ti.Vector([1.0, 0.0, 0.0]), a, b, c)
            cy += ray_triangle_hit(p, ti.Vector([0.0, 1.0, 0.0]), a, b, c)
            cz += ray_triangle_hit(p, ti.Vector([0.0, 0.0, 1.0]), a, b, c)
        votes = (cx % 2) + (cy % 2) + (cz % 2)
        sd = best_d2 ** 0.5
        if votes >= 2:
            sd = -sd
        sdf[i, j, k] = sd


@ti.func
def sdf_query(pos: ti.math.vec3) -> ti.f32:
    r = R[None]
    gmin = grid_min[None]
    gmax = grid_max[None]
    d = (gmax - gmin) / r
    result = 1e6
    inside_box = (pos[0] >= gmin[0] and pos[0] <= gmax[0] and
                  pos[1] >= gmin[1] and pos[1] <= gmax[1] and
                  pos[2] >= gmin[2] and pos[2] <= gmax[2])
    if inside_box:
        gx = (pos[0] - gmin[0]) / d[0]
        gy = (pos[1] - gmin[1]) / d[1]
        gz = (pos[2] - gmin[2]) / d[2]
        ix = ti.cast(ti.floor(gx), ti.i32)
        iy = ti.cast(ti.floor(gy), ti.i32)
        iz = ti.cast(ti.floor(gz), ti.i32)
        fx = gx - float(ix)
        fy = gy - float(iy)
        fz = gz - float(iz)
        rr = r - 1
        i0 = ti.max(0, ti.min(rr, ix))
        i1 = ti.max(0, ti.min(rr, ix + 1))
        j0 = ti.max(0, ti.min(rr, iy))
        j1 = ti.max(0, ti.min(rr, iy + 1))
        k0 = ti.max(0, ti.min(rr, iz))
        k1 = ti.max(0, ti.min(rr, iz + 1))
        c000 = sdf[i0, j0, k0]
        c100 = sdf[i1, j0, k0]
        c010 = sdf[i0, j1, k0]
        c110 = sdf[i1, j1, k0]
        c001 = sdf[i0, j0, k1]
        c101 = sdf[i1, j0, k1]
        c011 = sdf[i0, j1, k1]
        c111 = sdf[i1, j1, k1]
        c00 = c000 + (c100 - c000) * fx
        c10 = c010 + (c110 - c010) * fx
        c01 = c001 + (c101 - c001) * fx
        c11 = c011 + (c111 - c011) * fx
        c0 = c00 + (c10 - c00) * fy
        c1 = c01 + (c11 - c01) * fy
        result = c0 + (c1 - c0) * fz
    return result


@ti.func
def sdf_gradient(pos: ti.math.vec3, eps: ti.f32) -> ti.math.vec3:
    gx = (sdf_query(pos + ti.Vector([eps, 0.0, 0.0]))
          - sdf_query(pos - ti.Vector([eps, 0.0, 0.0]))) / (2.0 * eps)
    gy = (sdf_query(pos + ti.Vector([0.0, eps, 0.0]))
          - sdf_query(pos - ti.Vector([0.0, eps, 0.0]))) / (2.0 * eps)
    gz = (sdf_query(pos + ti.Vector([0.0, 0.0, eps]))
          - sdf_query(pos - ti.Vector([0.0, 0.0, eps]))) / (2.0 * eps)
    nn = (gx * gx + gy * gy + gz * gz) ** 0.5
    result = ti.Vector([0.0, 0.0, 1.0])
    if nn > 1e-10:
        result = ti.Vector([gx / nn, gy / nn, gz / nn])
    return result


@ti.kernel
def init_spheres_kernel(n: ti.i32):
    n_spheres[None] = n
    lo = spawn_lo[None]
    hi = spawn_hi[None]
    vy = spawn_vy[None]
    for i in range(n):
        sphere_pos[i] = ti.Vector([
            lo[0] + ti.random() * (hi[0] - lo[0]),
            lo[1] + ti.random() * (hi[1] - lo[1]),
            lo[2] + ti.random() * (hi[2] - lo[2]),
        ])
        sphere_vel[i] = ti.Vector([0.0, vy, 0.0])
        # color gradient by index
        t = float(i) / float(n)
        sphere_color[i] = ti.Vector([0.2 + 0.8 * t, 0.45, 0.9 - 0.6 * t])


@ti.func
def collide_response(vel: ti.math.vec3, n: ti.math.vec3,
                     rest: ti.f32, fric: ti.f32) -> ti.math.vec3:
    """
    Resolve a sphere-surface collision with proper Coulomb friction.

    n must be the unit outward surface normal.  Returns the post-collision
    velocity.  Key physics:
      - normal component is reflected with coefficient of restitution `rest`
      - tangential component loses a friction impulse capped at mu*|jn|,
        i.e. it is REDUCED by that amount (not clamped down to it).
        With fric=0 the tangential velocity is fully preserved -> the ball
        slides / rolls freely down the steps instead of sticking.
    """
    out = vel
    vn = vel.dot(n)
    if vn < 0.0:                       # only respond when moving into surface
        vt = vel - vn * n              # tangential velocity (preserved dir)
        vt_mag = vt.norm()

        # normal impulse magnitude (change of normal velocity)
        jn = (1.0 + rest) * (-vn)      # > 0
        # Coulomb: friction impulse opposes tangential motion, |jt| <= mu*jn
        dvt = ti.min(fric * jn, vt_mag)

        new_vt = vt
        if vt_mag > 1e-12:
            new_vt = vt * (1.0 - dvt / vt_mag)   # subtract friction impulse

        new_vn = -rest * vn            # reflected normal velocity (>= 0)
        out = new_vt + new_vn * n
    return out


@ti.kernel
def physics_step_kernel(dt: ti.f32):
    g = gravity[None]
    rest = restitution[None]
    fric = friction_coeff[None]
    rad = sphere_radius[None]
    gz = ground_z[None]
    eps = 1e-4
    for i in range(n_spheres[None]):
        sphere_vel[i] += g * dt
        new_pos = sphere_pos[i] + sphere_vel[i] * dt

        # mesh SDF collision
        d = sdf_query(new_pos)
        if d < rad:
            n = sdf_gradient(new_pos, eps)
            new_pos += (rad - d) * n
            sphere_vel[i] = collide_response(sphere_vel[i], n, rest, fric)

        # ground plane
        if new_pos[2] < gz + rad:
            new_pos[2] = gz + rad
            n = ti.Vector([0.0, 0.0, 1.0])
            sphere_vel[i] = collide_response(sphere_vel[i], n, rest, fric)

        sphere_pos[i] = new_pos

# PLACEHOLDER_MAIN

# ===================================================================
# Mesh render buffers (filled in main)
# ===================================================================
mesh_verts = ti.Vector.field(3, ti.f32, shape=_MAX_TRIS * 3)
mesh_indices = ti.field(ti.i32, shape=_MAX_TRIS * 3)
mesh_color = ti.Vector.field(3, ti.f32, shape=_MAX_TRIS * 3)

# ground quad
ground_verts = ti.Vector.field(3, ti.f32, shape=4)
ground_indices = ti.field(ti.i32, shape=6)
ground_color = ti.Vector.field(3, ti.f32, shape=4)


def main():
    parser = argparse.ArgumentParser(
        description="GGUI: spheres tumbling down an SDF staircase")
    parser.add_argument("--obj", default="model/stair.obj")
    parser.add_argument("--grid-res", "-r", type=int, default=64)
    parser.add_argument("--num-spheres", "-n", type=int, default=80)
    parser.add_argument("--sphere-radius", type=float, default=0.08)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=2e-3)
    parser.add_argument("--gravity-y", type=float, default=-1.0,
                        help="-y gravity to nudge balls down the stairs")
    parser.add_argument("--gravity-z", type=float, default=-9.8)
    parser.add_argument("--restitution", type=float, default=0.15)
    parser.add_argument("--friction", type=float, default=0.15,
                        help="Coulomb friction (0 = frictionless slide; "
                             "~0.15 makes balls settle on the steps on-screen)")
    parser.add_argument("--init-vy", type=float, default=-1.0)
    parser.add_argument("--ground-z", type=float, default=None)
    parser.add_argument("--reset-after", type=float, default=8.0,
                        help="auto-reset spheres every N seconds (0=never)")
    args = parser.parse_args()

    # ---- load mesh ----
    print(f"Loading OBJ: {args.obj}")
    verts_np, faces_np = load_obj(args.obj)
    nf = len(faces_np)
    v = verts_np.astype(np.float32)
    f = faces_np.astype(np.int32)

    # upload triangles for SDF
    n_faces[None] = nf
    for fi in range(nf):
        a, b, c = v[f[fi, 0]], v[f[fi, 1]], v[f[fi, 2]]
        tri_v0[fi] = ti.math.vec3(*[float(x) for x in a])
        tri_v1[fi] = ti.math.vec3(*[float(x) for x in b])
        tri_v2[fi] = ti.math.vec3(*[float(x) for x in c])

    # ---- compute SDF ----
    r = args.grid_res
    R[None] = r
    vmin = verts_np.min(axis=0)
    vmax = verts_np.max(axis=0)
    grid_min[None] = ti.math.vec3(*[float(vmin[i] - 0.1) for i in range(3)])
    grid_max[None] = ti.math.vec3(*[float(vmax[i] + 0.1) for i in range(3)])
    print(f"Computing SDF ({r}^3)...")
    import time
    t0 = time.time()
    compute_sdf_kernel()
    print(f"  done in {time.time() - t0:.2f}s")

    # ---- build mesh render buffers (flat-shaded, per-triangle) ----
    flat_v = np.zeros((nf * 3, 3), dtype=np.float32)
    flat_i = np.arange(nf * 3, dtype=np.int32)
    flat_c = np.zeros((nf * 3, 3), dtype=np.float32)
    for fi in range(nf):
        for j in range(3):
            flat_v[fi * 3 + j] = v[f[fi, j]]
            flat_c[fi * 3 + j] = [0.80, 0.82, 0.86]
    mesh_verts.from_numpy(np.vstack([flat_v, np.zeros((_MAX_TRIS * 3 - nf * 3, 3), np.float32)]))
    mesh_indices.from_numpy(np.concatenate([flat_i, np.zeros(_MAX_TRIS * 3 - nf * 3, np.int32)]))
    mesh_color.from_numpy(np.vstack([flat_c, np.zeros((_MAX_TRIS * 3 - nf * 3, 3), np.float32)]))
    n_mesh_idx = nf * 3

    # ---- ground plane ----
    gz_floor = float(args.ground_z) if args.ground_z is not None else float(vmin[2])
    cy = float((vmin[1] + vmax[1]) * 0.5)
    ext = 4.0
    gverts = np.array([
        [-ext, cy - ext, gz_floor], [ext, cy - ext, gz_floor],
        [ext, cy + ext, gz_floor], [-ext, cy + ext, gz_floor],
    ], dtype=np.float32)
    ground_verts.from_numpy(gverts)
    ground_indices.from_numpy(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))
    ground_color.from_numpy(np.tile([0.55, 0.60, 0.66], (4, 1)).astype(np.float32))

    # ---- physics params ----
    sphere_radius[None] = args.sphere_radius
    gravity[None] = ti.math.vec3(0.0, float(args.gravity_y), float(args.gravity_z))
    restitution[None] = args.restitution
    friction_coeff[None] = args.friction
    ground_z[None] = gz_floor

    # spawn region: above the top of the staircase
    spawn_lo[None] = ti.math.vec3(-0.6, float(vmax[1]) - 0.5, float(vmax[2]) + 0.3)
    spawn_hi[None] = ti.math.vec3(0.6, float(vmax[1]) - 0.1, float(vmax[2]) + 0.7)
    spawn_vy[None] = float(args.init_vy)

    ns = min(args.num_spheres, _MAX_SPHERES)
    init_spheres_kernel(ns)

    # ---- scene center / camera target ----
    scene_center = np.array([0.0,
                             float((vmin[1] + vmax[1]) * 0.5),
                             float((vmin[2] + vmax[2]) * 0.5)], dtype=np.float32)

    # ===================================================================
    # GGUI window
    # ===================================================================
    window = ti.ui.Window("Spheres on SDF Staircase", (1100, 850), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.95, 0.96, 0.98))
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # camera orbit state (Z-up world)
    cam_dist = 7.5
    cam_azimuth = math.radians(145.0)   # matches the rotated offline view
    cam_elev = math.radians(20.0)

    last_mx, last_my = 0.0, 0.0
    dragging = False
    paused = False
    sim_time = 0.0

    print("=" * 56)
    print("Controls:  LMB drag = orbit   |   wheel/[/]= zoom")
    print("           R = reset   SPACE = pause   ESC = quit")
    print(f"friction={args.friction}  gravity=({args.gravity_y},{args.gravity_z})")
    print("=" * 56)

    while window.running:
        # ---- events ----
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ti.ui.SPACE:
                paused = not paused
            elif e.key == 'r':
                init_spheres_kernel(ns)
                sim_time = 0.0
            elif e.key == '[':
                cam_dist = min(20.0, cam_dist + 0.5)
            elif e.key == ']':
                cam_dist = max(2.0, cam_dist - 0.5)

        # ---- mouse drag orbit ----
        if window.is_pressed(ti.ui.LMB):
            mx, my = window.get_cursor_pos()
            if not dragging:
                dragging = True
                last_mx, last_my = mx, my
            else:
                dx = mx - last_mx
                dy = my - last_my
                cam_azimuth -= dx * 3.0
                cam_elev += dy * 3.0
                cam_elev = max(-math.pi / 2 + 0.05,
                               min(math.pi / 2 - 0.05, cam_elev))
                last_mx, last_my = mx, my
        else:
            dragging = False

        # ---- step physics ----
        if not paused:
            for _ in range(args.substeps):
                physics_step_kernel(args.dt)
            sim_time += args.substeps * args.dt
            if args.reset_after > 0 and sim_time > args.reset_after:
                init_spheres_kernel(ns)
                sim_time = 0.0

        # ---- camera (Z-up) ----
        cx = scene_center[0] + cam_dist * math.cos(cam_elev) * math.sin(cam_azimuth)
        cy_ = scene_center[1] + cam_dist * math.cos(cam_elev) * math.cos(cam_azimuth)
        cz = scene_center[2] + cam_dist * math.sin(cam_elev)
        camera.position(cx, cy_, cz)
        camera.lookat(*scene_center)
        camera.up(0, 0, 1)            # Z is up in PhysGaussian space
        camera.fov(50)
        scene.set_camera(camera)

        scene.ambient_light((0.6, 0.6, 0.65))
        scene.point_light(pos=(3, 3, 8), color=(1, 1, 1))
        scene.point_light(pos=(-4, -2, 5), color=(0.4, 0.45, 0.55))

        # ---- draw ground ----
        scene.mesh(ground_verts, indices=ground_indices,
                   per_vertex_color=ground_color, two_sided=True)

        # ---- draw staircase ----
        scene.mesh(mesh_verts, indices=mesh_indices,
                   per_vertex_color=mesh_color,
                   index_count=n_mesh_idx, two_sided=True)

        # ---- draw spheres ----
        scene.particles(sphere_pos, radius=args.sphere_radius,
                        per_vertex_color=sphere_color,
                        index_count=ns)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()


