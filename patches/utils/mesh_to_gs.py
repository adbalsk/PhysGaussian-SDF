"""
Bake a triangle mesh into STATIC 3D Gaussians for rendering alongside the
deforming doll Gaussians in the same rasterizer.

Why: rendering the stairs as Gaussians (instead of a separate mesh renderer)
makes occlusion between the doll and the stairs automatic -- everything goes
through one Diff-Gaussian-Rasterization pass.

The produced Gaussians live in SIM SPACE [0, grid_lim]^3 (same space the MPM
solver and the SDF use).  The caller maps them back to world space with the
SAME inverse transform applied to the doll before rasterizing.

Output tensors match the doll's field layout so they can be concatenated:
  pos      : (M, 3)         float32   sim-space positions
  cov3D    : (M, 6)         float32   upper-tri covariance (xx,xy,xz,yy,yz,zz)
  opacity  : (M, 1)         float32
  shs      : (M, K, 3)      float32   SH coeffs (K = (sh_degree+1)^2), DC only
"""

import math

import numpy as np
import torch

# SH DC basis constant (sh_utils.SH2RGB / RGB2SH use C0 = 0.28209479177387814)
_SH_C0 = 0.28209479177387814


def _sample_triangles(verts, faces, samples_per_area, min_per_face=1, seed=0,
                      sampling="stratified"):
    """
    Area-weighted sampling of points on the mesh surface.

    sampling="random"     : plain random barycentric (legacy; clumps + holes).
    sampling="stratified" : jittered low-discrepancy barycentric grid per face,
                            giving even coverage with far fewer clumps/holes.

    Returns (pts, normals):
      pts     : (P, 3) sampled positions
      normals : (P, 3) per-sample surface normal (flat per-face normal)
    """
    rng = np.random.default_rng(seed)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    # triangle areas + (un-normalised) face normals
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-20)

    counts = np.maximum(min_per_face,
                        np.round(areas * samples_per_area).astype(np.int64))
    total = int(counts.sum())

    pts = np.empty((total, 3), dtype=np.float32)
    nrm = np.empty((total, 3), dtype=np.float32)
    idx = 0
    for fi in range(len(faces)):
        c = int(counts[fi])
        if sampling == "stratified":
            # lay c samples on a jittered sqrt(c) x sqrt(c) grid in unit square,
            # then fold into the triangle (Turk's reflection) so the barycentric
            # coverage is even instead of random-clumped.
            side = max(1, int(math.ceil(math.sqrt(c))))
            gi, gj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
            u = (gi.ravel() + rng.random(side * side)) / side
            v = (gj.ravel() + rng.random(side * side)) / side
            sel = rng.permutation(side * side)[:c]
            u = u[sel]
            v = v[sel]
            # fold square -> triangle
            over = (u + v) > 1.0
            u[over] = 1.0 - u[over]
            v[over] = 1.0 - v[over]
            r1 = u
            r2 = v
            a = (1.0 - r1 - r2)[:, None]
            b = r1[:, None]
            cc = r2[:, None]
        else:
            r1 = np.sqrt(rng.random(c))
            r2 = rng.random(c)
            a = (1.0 - r1)[:, None]
            b = (r1 * (1.0 - r2))[:, None]
            cc = (r1 * r2)[:, None]
        pts[idx:idx + c] = a * v0[fi] + b * v1[fi] + cc * v2[fi]
        nrm[idx:idx + c] = face_normals[fi]
        idx += c
    return pts[:idx], nrm[:idx]


def _anisotropic_cov6(normals, scale_tangent, scale_normal):
    """
    Build per-point upper-tri covariance (M,6: xx,xy,xz,yy,yz,zz) for FLAT disk
    gaussians: large variance in the surface tangent plane, thin along normal.

    For each normal n, pick two orthonormal tangents t1,t2, then
        C = s_t^2 (t1 t1^T + t2 t2^T) + s_n^2 (n n^T).
    """
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-20)
    # robust tangent: cross with whichever axis is least aligned with n
    helper = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (len(n), 1))
    near_z = np.abs(n[:, 2]) > 0.9
    helper[near_z] = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(n, helper)
    t1 /= (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-20)
    t2 = np.cross(n, t1)
    t2 /= (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-20)

    st2 = scale_tangent * scale_tangent
    sn2 = scale_normal * scale_normal
    # C = st2*(t1 t1^T + t2 t2^T) + sn2*(n n^T)
    def outer(a, b):
        return a[:, :, None] * b[:, None, :]
    C = st2 * (outer(t1, t1) + outer(t2, t2)) + sn2 * outer(n, n)
    cov6 = np.stack([C[:, 0, 0], C[:, 0, 1], C[:, 0, 2],
                     C[:, 1, 1], C[:, 1, 2], C[:, 2, 2]], axis=1)
    return cov6.astype(np.float32)


def lambertian_shade(normals, base_color, lights=None, ambient=0.35):
    """
    Bake simple Lambertian shading into per-gaussian RGB (M,3), clamped [0,1].

    normals    : (M,3) world-space surface normals.
    base_color : (3,) albedo in [0,1].
    lights     : list of (dir_xyz, intensity, color_rgb); dir points FROM surface
                 TOWARD the light.  Default: one key + one fill.
    ambient    : constant ambient term added before lights.
    """
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-20)
    base = np.asarray(base_color, dtype=np.float32)
    if lights is None:
        lights = [
            (np.array([0.4, 0.5, 0.8]), 0.9, np.array([1.0, 1.0, 0.97])),
            (np.array([-0.6, -0.3, 0.5]), 0.35, np.array([0.7, 0.78, 1.0])),
        ]
    shade = np.full((len(n),), ambient, dtype=np.float32)
    rgb = base[None, :] * shade[:, None]
    for ldir, inten, lcol in lights:
        ld = np.asarray(ldir, dtype=np.float64)
        ld = ld / (np.linalg.norm(ld) + 1e-20)
        # two-sided: stairs face both ways; use abs so back faces aren't black
        ndl = np.abs(n @ ld)
        rgb = rgb + (base * np.asarray(lcol, dtype=np.float32))[None, :] * \
            (inten * ndl)[:, None]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def bake_mesh_to_gaussians(
    vertices: np.ndarray,
    faces: np.ndarray,
    sh_degree: int = 3,
    color=(0.55, 0.55, 0.58),
    samples_per_area: float = 4000.0,
    gaussian_scale: float = None,
    opacity: float = 0.995,
    device: str = "cuda",
    seed: int = 0,
    sampling: str = "stratified",
    anisotropic: bool = True,
    flatten_ratio: float = 0.15,
    lights=None,
    ambient: float = 0.35,
):
    """
    Bake the mesh into static Gaussians (in the mesh's own space).

    Parameters
    ----------
    vertices, faces  : mesh geometry (sim space recommended).
    sh_degree        : must match the doll model's sh_degree (so K matches).
    color            : RGB in [0,1] for the (constant) surface albedo.
    samples_per_area : sampling density (points per unit area).
    gaussian_scale   : in-plane stddev of each Gaussian; if None, auto from
                       mean sample spacing.
    opacity          : per-Gaussian opacity (high => opaque solid surface).
    sampling         : "stratified" (even, recommended) or "random" (legacy).
    anisotropic      : if True, gaussians are FLAT disks aligned to the surface
                       (thin along normal) -> crisp surfaces, no fuzzy blobs.
    flatten_ratio    : normal-axis stddev = flatten_ratio * gaussian_scale.
    lights           : optional light list for baked Lambertian shading; if
                       provided (or non-None), per-gaussian colors are shaded.
                       Pass [] / None to keep flat constant color.
    ambient          : ambient term used when shading.

    Returns dict of torch tensors on `device`:
      pos (M,3), cov3D (M,6), opacity (M,1), shs (M,K,3),
      normals (M,3), colors_flat (M,3), colors_lit (M,3)
    The two color arrays let a viewer toggle lighting at zero runtime cost.
    """
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int64)

    pts, normals = _sample_triangles(v, f, samples_per_area, seed=seed,
                                     sampling=sampling)
    M = len(pts)

    # auto gaussian scale: roughly the mean nearest-neighbor spacing so the
    # surface is covered without big holes.  Approximate via total area / M.
    if gaussian_scale is None:
        v0 = v[f[:, 0]]; v1 = v[f[:, 1]]; v2 = v[f[:, 2]]
        area = float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
        spacing = np.sqrt(area / max(M, 1))
        gaussian_scale = float(spacing * 0.7)

    if anisotropic:
        # flat disk: large in tangent plane, thin along normal
        cov6 = _anisotropic_cov6(normals,
                                 scale_tangent=gaussian_scale,
                                 scale_normal=gaussian_scale * flatten_ratio)
    else:
        s2 = gaussian_scale * gaussian_scale
        cov6 = np.zeros((M, 6), dtype=np.float32)
        cov6[:, 0] = s2; cov6[:, 3] = s2; cov6[:, 5] = s2

    # SH: DC term only so the surface is a flat constant color.
    # rasterizer reconstructs rgb = SH_C0 * dc + 0.5  => dc = (rgb-0.5)/SH_C0
    K = (sh_degree + 1) ** 2
    shs = np.zeros((M, K, 3), dtype=np.float32)
    rgb = np.asarray(color, dtype=np.float32)
    shs[:, 0, :] = (rgb - 0.5) / _SH_C0

    opa = np.full((M, 1), float(opacity), dtype=np.float32)

    # precompute both flat and lit per-gaussian colors (for runtime toggle)
    colors_flat = np.tile(rgb, (M, 1)).astype(np.float32)
    colors_lit = lambertian_shade(normals, color, lights=lights, ambient=ambient)

    print(f"[bake_mesh_to_gaussians] {M} gaussians, scale={gaussian_scale:.5f}, "
          f"K={K}, sampling={sampling}, aniso={anisotropic}, color={tuple(color)}")

    return {
        "pos": torch.from_numpy(pts).to(device),
        "cov3D": torch.from_numpy(cov6).to(device),
        "opacity": torch.from_numpy(opa).to(device),
        "shs": torch.from_numpy(shs).to(device),
        "normals": torch.from_numpy(normals).to(device),
        "colors_flat": torch.from_numpy(colors_flat).to(device),
        "colors_lit": torch.from_numpy(colors_lit).to(device),
    }
