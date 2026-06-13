"""
Mesh to Signed Distance Field (SDF) conversion utilities.

Provides:
  - load_obj: Parse triangulated OBJ to vertices + faces
  - MeshSDF: Grid-based signed distance field with trilinear query
  - Warp integration: SDFCollider struct + collision kernel

Usage:
    sdf = MeshSDF.from_obj("model/stair.obj", grid_res=64)
    sdf.save("model/stair_sdf.npz")

    # Query in numpy
    d = sdf.query_numpy(points_np)

    # Query in Warp kernels
    collider = sdf.to_warp(device="cuda:0")
    # ... then use apply_sdf_collision kernel
"""

import numpy as np
import os
from typing import Optional, Tuple
from dataclasses import dataclass


# =========================================================================
# Fast Taichi baker: SDF aligned to the MPM grid (sim space [0, grid_lim]^3)
# =========================================================================
#
# This is the canonical path used by the doll-on-stairs pipeline.  It bakes a
# signed distance field directly onto the SAME grid the MPM solver uses, so the
# collision kernel can index sdf[i, j, k] with NO runtime interpolation.
#
# Grid node (i, j, k) maps to sim-space position (i, j, k) * dx,
# where dx = grid_lim / n_grid (matching MPM's convention, domain [0, grid_lim]^3).
#
# Sign is determined by RAY-PARITY (Moller-Trumbore + majority vote over 3 axes),
# which is robust for thin-shell / non-watertight meshes (e.g. the stair model).


def bake_sdf_mpm_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_grid: int,
    grid_lim: float,
    device: str = "cuda",
    verbose: bool = True,
) -> np.ndarray:
    """
    Bake a signed distance field onto the MPM grid using Taichi (GPU).

    Parameters
    ----------
    vertices : (V, 3) float32 — mesh vertices, ALREADY in sim space [0, grid_lim]^3.
    faces    : (F, 3) int     — triangle vertex indices.
    n_grid   : int            — MPM grid resolution.
    grid_lim : float          — sim-space domain size; node (i,j,k) at (i,j,k)*dx.

    Returns
    -------
    sdf : (n_grid, n_grid, n_grid) float32 — signed distance (>0 outside).
    """
    import taichi as ti  # lazy import (module stays warp-only loadable)

    if not ti.is_logging_effective:  # cheap no-op guard; ti.init is idempotent-ish
        pass
    # ti.init is safe to call again; if the host already inited cuda this is a no-op
    try:
        ti.init(arch=ti.cuda)
    except Exception:
        pass

    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    nf = len(f)
    dx = grid_lim / n_grid

    tri0 = ti.Vector.field(3, ti.f32, shape=nf)
    tri1 = ti.Vector.field(3, ti.f32, shape=nf)
    tri2 = ti.Vector.field(3, ti.f32, shape=nf)
    sdf_field = ti.field(ti.f32, shape=(n_grid, n_grid, n_grid))

    tri0.from_numpy(v[f[:, 0]])
    tri1.from_numpy(v[f[:, 1]])
    tri2.from_numpy(v[f[:, 2]])

    @ti.func
    def _closest_on_edge(p, p0, p1):
        e = p1 - p0
        ee = e.dot(e)
        res = p0
        if ee >= 1e-15:
            t = (p - p0).dot(e) / ee
            t = ti.max(0.0, ti.min(1.0, t))
            res = p0 + t * e
        return res

    @ti.func
    def _pt_tri_d2(p, a, b, c):
        v0 = c - a
        v1 = b - a
        v2 = p - a
        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d11 = v1.dot(v1)
        denom = d00 * d11 - d01 * d01
        best = v2.dot(v2)
        need = 1
        if denom >= 1e-15:
            d02 = v2.dot(v0)
            d12 = v2.dot(v1)
            uu = (d11 * d02 - d01 * d12) / denom
            vv = (d00 * d12 - d01 * d02) / denom
            if uu >= 0.0 and vv >= 0.0 and uu + vv <= 1.0:
                q = a + uu * v0 + vv * v1
                best = (p - q).dot(p - q)
                need = 0
        if need:
            qab = _closest_on_edge(p, a, b)
            qbc = _closest_on_edge(p, b, c)
            qca = _closest_on_edge(p, c, a)
            best = (p - qab).dot(p - qab)
            d_bc = (p - qbc).dot(p - qbc)
            d_ca = (p - qca).dot(p - qca)
            if d_bc < best:
                best = d_bc
            if d_ca < best:
                best = d_ca
        return best

    @ti.func
    def _ray_hit(orig, dir, a, b, c):
        eps = 1e-9
        e1 = b - a
        e2 = c - a
        pv = dir.cross(e2)
        det = e1.dot(pv)
        hit = 0
        if det > eps or det < -eps:
            inv = 1.0 / det
            tv = orig - a
            u = tv.dot(pv) * inv
            if u >= 0.0 and u <= 1.0:
                qv = tv.cross(e1)
                vrt = dir.dot(qv) * inv
                if vrt >= 0.0 and u + vrt <= 1.0:
                    t = e2.dot(qv) * inv
                    if t > eps:
                        hit = 1
        return hit

    @ti.kernel
    def _bake(nfk: ti.i32, dxk: ti.f32):
        for i, j, k in ti.ndrange(n_grid, n_grid, n_grid):
            p = ti.Vector([float(i), float(j), float(k)]) * dxk
            best_d2 = 1e20
            cx = 0
            cy = 0
            cz = 0
            for fi in range(nfk):
                a = tri0[fi]
                b = tri1[fi]
                c = tri2[fi]
                d2 = _pt_tri_d2(p, a, b, c)
                if d2 < best_d2:
                    best_d2 = d2
                cx += _ray_hit(p, ti.Vector([1.0, 0.0, 0.0]), a, b, c)
                cy += _ray_hit(p, ti.Vector([0.0, 1.0, 0.0]), a, b, c)
                cz += _ray_hit(p, ti.Vector([0.0, 0.0, 1.0]), a, b, c)
            votes = (cx % 2) + (cy % 2) + (cz % 2)
            sd = best_d2 ** 0.5
            if votes >= 2:
                sd = -sd
            sdf_field[i, j, k] = sd

    import time
    t0 = time.time()
    _bake(nf, dx)
    out = sdf_field.to_numpy()
    if verbose:
        inside = (out < 0).sum()
        tot = n_grid ** 3
        print(f"[bake_sdf_mpm_grid] {n_grid}^3 grid, {nf} tris, dx={dx:.5f}, "
              f"{time.time()-t0:.2f}s")
        print(f"  inside {inside}/{tot} ({100*inside/tot:.2f}%)  "
              f"range [{out.min():.4f}, {out.max():.4f}]")
    return out.astype(np.float32)

import warp as wp


# =========================================================================
# OBJ Loading
# =========================================================================


def load_obj(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a triangulated OBJ file.

    Returns
    -------
    vertices : (V, 3) float32
    faces    : (F, 3) int64  (vertex indices, 0-based)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"OBJ file not found: {filepath}")

    vertices = []
    faces = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                idxs = []
                for p in parts[1:]:
                    idx = p.split("/")[0]
                    idxs.append(int(idx) - 1)  # OBJ indices are 1-based
                # triangulate polygons / quads
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])

    V = np.array(vertices, dtype=np.float32).reshape(-1, 3)
    F = np.array(faces, dtype=np.int64).reshape(-1, 3)

    print(f"[load_obj] {filepath}: {len(V)} vertices, {len(F)} faces")
    print(f"  bounds: X [{V[:, 0].min():.4f}, {V[:, 0].max():.4f}]")
    print(f"          Y [{V[:, 1].min():.4f}, {V[:, 1].max():.4f}]")
    print(f"          Z [{V[:, 2].min():.4f}, {V[:, 2].max():.4f}]")
    return V, F


# =========================================================================
# Triangle distance helpers
# =========================================================================


def _closest_point_on_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Return (closest_point, squared_distance, barycentric_weights)
    for the closest point on triangle (a,b,c) to point p.
    """
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot11 = float(np.dot(v1, v1))
    denom = dot00 * dot11 - dot01 * dot01

    if denom < 1e-15:
        # degenerate triangle — fall back to vertex a
        d2 = float(np.dot(v2, v2))
        return a, d2, np.array([1.0, 0.0, 0.0], dtype=np.float32)

    dot02 = float(np.dot(v2, v0))
    dot12 = float(np.dot(v2, v1))

    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    w = 1.0 - u - v

    if u >= 0.0 and v >= 0.0 and w >= 0.0:
        # inside triangle
        closest = u * a + v * b + w * c
        d2 = float(np.dot(p - closest, p - closest))
        return closest, d2, np.array([u, v, w], dtype=np.float32)

    # --- outside triangle: check 3 edges and 3 vertices ---
    best_d2 = float("inf")
    best_p = a.copy()

    def _check_edge(p0, p1):
        nonlocal best_d2, best_p
        e = p1 - p0
        ee = float(np.dot(e, e))
        if ee < 1e-15:
            return
        t = np.dot(p - p0, e) / ee
        t = float(np.clip(t, 0.0, 1.0))
        q = p0 + t * e
        d2 = float(np.dot(p - q, p - q))
        if d2 < best_d2:
            best_d2 = d2
            best_p = q.copy()

    # edges
    _check_edge(a, b)
    _check_edge(b, c)
    _check_edge(c, a)
    # vertices (already covered by edge endpoints but explicit for safety)
    for vtx in (a, b, c):
        d2 = float(np.dot(p - vtx, p - vtx))
        if d2 < best_d2:
            best_d2 = d2
            best_p = vtx.copy()

    return best_p, best_d2, None


# =========================================================================
# Grid-based SDF computation
# =========================================================================


def compute_voxel_sdf(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid_res: int = 64,
    padding: float = 0.1,
    margin_cells: int = 3,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a signed distance field on a uniform grid.

    The algorithm rasterises each triangle's bounding box (with margin_cells
    of padding) and computes exact point-to-triangle distance for every
    grid point inside.  Sign is determined by the closest triangle's normal.

    Parameters
    ----------
    vertices : (V, 3) float32 — mesh vertex positions.
    faces : (F, 3) int64 — triangle vertex indices.
    grid_res : int — number of grid intervals per axis (grid has res+1 nodes).
    padding : float — extra space around mesh bounding box.
    margin_cells : int — how many extra grid cells to examine around each
                         triangle's AABB.

    Returns
    -------
    sdf_grid  : (R, R, R) float32  where R = grid_res+1
    grid_min  : (3,) float32
    grid_max  : (3,) float32
    """
    vmin = vertices.min(axis=0).astype(np.float64) - padding
    vmax = vertices.max(axis=0).astype(np.float64) + padding
    dx = (vmax - vmin) / grid_res

    R = grid_res + 1
    # initialise with large positive value
    sdf2 = np.full((R, R, R), 1e20, dtype=np.float64)
    # which triangle gave the minimum (for sign)
    closest_fi = np.full((R, R, R), -1, dtype=np.int32)

    # precompute face normals (un-normalised length stored for speed)
    tri_v = vertices[faces]  # (F, 3, 3)
    e0 = tri_v[:, 1] - tri_v[:, 0]
    e1 = tri_v[:, 2] - tri_v[:, 0]
    cross = np.cross(e0, e1)
    norms = np.linalg.norm(cross, axis=1)
    normals = cross / (norms[:, None] + 1e-30)

    nF = len(faces)
    for fi in range(nF):
        a, b, c = tri_v[fi]
        n = normals[fi]

        # AABB of this triangle (world space, expanded by margin)
        bb_min = tri_v[fi].min(axis=0) - margin_cells * dx
        bb_max = tri_v[fi].max(axis=0) + margin_cells * dx

        i0 = max(0, int(np.floor((bb_min[0] - vmin[0]) / dx[0])))
        i1 = min(grid_res, int(np.ceil((bb_max[0] - vmin[0]) / dx[0])))
        j0 = max(0, int(np.floor((bb_min[1] - vmin[1]) / dx[1])))
        j1 = min(grid_res, int(np.ceil((bb_max[1] - vmin[1]) / dx[1])))
        k0 = max(0, int(np.floor((bb_min[2] - vmin[2]) / dx[2])))
        k1 = min(grid_res, int(np.ceil((bb_max[2] - vmin[2]) / dx[2])))

        for i in range(i0, i1 + 1):
            px = vmin[0] + i * dx[0]
            for j in range(j0, j1 + 1):
                py = vmin[1] + j * dx[1]
                for k in range(k0, k1 + 1):
                    pz = vmin[2] + k * dx[2]
                    p = np.array([px, py, pz])
                    _, d2, _ = _closest_point_on_triangle(p, a, b, c)
                    if d2 < sdf2[i, j, k]:
                        sdf2[i, j, k] = d2
                        closest_fi[i, j, k] = fi

        if verbose and (fi + 1) % 500 == 0:
            print(f"  [{fi+1}/{nF}] triangles processed ...")

    # ---- assign sign ----
    if verbose:
        print("  Assigning sign via closest triangle normals ...")
    sdf = np.sqrt(sdf2).astype(np.float32)

    for i in range(R):
        px = vmin[0] + i * dx[0]
        for j in range(R):
            py = vmin[1] + j * dx[1]
            for k in range(R):
                fi = closest_fi[i, j, k]
                if fi < 0:
                    continue  # keep positive (outside)
                pz = vmin[2] + k * dx[2]
                p = np.array([px, py, pz])
                a, b, c = tri_v[fi]
                q, _, _ = _closest_point_on_triangle(p, a, b, c)
                # normal direction  → inside if we're on the opposite side
                if np.dot(p - q, normals[fi]) < 0:
                    sdf[i, j, k] = -sdf[i, j, k]

    if verbose:
        inside = (sdf < 0).sum()
        total = R * R * R
        print(f"  SDF done: {inside}/{total} cells inside ({100*inside/total:.1f}%)")
        print(f"  SDF range: [{sdf.min():.6f}, {sdf.max():.6f}]")

    return sdf, vmin.astype(np.float32), vmax.astype(np.float32)


# =========================================================================
# MeshSDF  class  (numpy — CPU)
# =========================================================================


@dataclass
class MeshSDF:
    """Grid-based signed distance field built from a triangle mesh."""

    sdf_grid: np.ndarray       # (R, R, R) float32
    grid_min: np.ndarray       # (3,) float32
    grid_max: np.ndarray       # (3,) float32
    grid_res: int              # number of intervals (grid has res+1 nodes)

    def __post_init__(self):
        self._dx = (self.grid_max - self.grid_min) / self.grid_res
        self._inv_dx = 1.0 / (self._dx + 1e-30)

    # ---- Factory ----

    @classmethod
    def from_obj(
        cls,
        obj_path: str,
        grid_res: int = 64,
        padding: float = 0.1,
        margin_cells: int = 3,
    ) -> "MeshSDF":
        """Build a MeshSDF from an OBJ file."""
        vertices, faces = load_obj(obj_path)
        sdf_grid, gmin, gmax = compute_voxel_sdf(
            vertices, faces,
            grid_res=grid_res, padding=padding,
            margin_cells=margin_cells,
        )
        return cls(sdf_grid=sdf_grid, grid_min=gmin, grid_max=gmax, grid_res=grid_res)

    # ---- I/O ----

    def save(self, path: str):
        """Save to compressed numpy archive."""
        np.savez_compressed(
            path,
            sdf_grid=self.sdf_grid,
            grid_min=self.grid_min,
            grid_max=self.grid_max,
            grid_res=np.array(self.grid_res, dtype=np.int32),
        )
        print(f"[MeshSDF] saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MeshSDF":
        """Load from .npz saved by save()."""
        data = np.load(path)
        return cls(
            sdf_grid=data["sdf_grid"],
            grid_min=data["grid_min"],
            grid_max=data["grid_max"],
            grid_res=int(data["grid_res"]),
        )

    # ---- Query (numpy) ----

    def query_numpy(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate SDF at arbitrary 3D points via trilinear interpolation.

        Parameters
        ----------
        points : (..., 3) float32 — query positions.

        Returns
        -------
        values : (...) float32 — signed distances (positive = outside).
        """
        pts = np.asarray(points, dtype=np.float32)
        orig_shape = pts.shape[:-1]
        pts = pts.reshape(-1, 3)

        # continuous grid coordinates
        g = (pts - self.grid_min[None]) * self._inv_dx[None]  # (N, 3)
        gi = np.floor(g).astype(np.int32)
        gf = g - gi.astype(np.float32)

        # clamp to valid range
        R = self.grid_res
        gi = np.clip(gi, 0, R - 1)

        # gather 8 corners
        def _gather(i, j, k):
            return self.sdf_grid[
                np.clip(i, 0, R),
                np.clip(j, 0, R),
                np.clip(k, 0, R),
            ]

        c000 = _gather(gi[:, 0], gi[:, 1], gi[:, 2])
        c100 = _gather(gi[:, 0] + 1, gi[:, 1], gi[:, 2])
        c010 = _gather(gi[:, 0], gi[:, 1] + 1, gi[:, 2])
        c110 = _gather(gi[:, 0] + 1, gi[:, 1] + 1, gi[:, 2])
        c001 = _gather(gi[:, 0], gi[:, 1], gi[:, 2] + 1)
        c101 = _gather(gi[:, 0] + 1, gi[:, 1], gi[:, 2] + 1)
        c011 = _gather(gi[:, 0], gi[:, 1] + 1, gi[:, 2] + 1)
        c111 = _gather(gi[:, 0] + 1, gi[:, 1] + 1, gi[:, 2] + 1)

        fx, fy, fz = gf[:, 0], gf[:, 1], gf[:, 2]

        # trilinear interpolation
        c00 = c000 + (c100 - c000) * fx
        c10 = c010 + (c110 - c010) * fx
        c01 = c001 + (c101 - c001) * fx
        c11 = c011 + (c111 - c011) * fx
        c0 = c00 + (c10 - c00) * fy
        c1 = c01 + (c11 - c01) * fy
        return (c0 + (c1 - c0) * fz).reshape(orig_shape)

    def gradient_numpy(self, points: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Compute SDF gradient (normalised) via finite differences.

        Returns
        -------
        grad : (..., 3) float32
        """
        pts = np.asarray(points, dtype=np.float32)
        orig = pts.reshape(-1, 3)
        N = orig.shape[0]
        grad = np.empty_like(orig)
        for d in range(3):
            off = np.zeros((N, 3), dtype=np.float32)
            off[:, d] = eps
            fp = self.query_numpy(orig + off)
            fn = self.query_numpy(orig - off)
            grad[:, d] = (fp - fn) / (2.0 * eps)
        # normalise
        nrm = np.linalg.norm(grad, axis=1, keepdims=True)
        nrm = np.where(nrm > 1e-10, nrm, 1.0)
        grad = grad / nrm
        return grad.reshape(*pts.shape[:-1], 3)


# =========================================================================
# Warp integration  (GPU)
# =========================================================================


@wp.struct
class SDFCollider:
    """Warp-compatible collider backed by a grid SDF."""

    grid_min: wp.vec3
    grid_max: wp.vec3
    grid_res: int
    dx: wp.vec3
    inv_dx: wp.vec3
    sdf: wp.array(dtype=float, ndim=3)


def mesh_sdf_to_warp(sdf: MeshSDF, device: str = "cuda:0") -> SDFCollider:
    """Upload a MeshSDF to a Warp SDFCollider struct."""
    sdf_wp = wp.from_numpy(sdf.sdf_grid, dtype=float, device=device)
    collider = SDFCollider()
    collider.grid_min = wp.vec3(*sdf.grid_min)
    collider.grid_max = wp.vec3(*sdf.grid_max)
    collider.grid_res = sdf.grid_res
    collider.dx = wp.vec3(*sdf._dx)
    collider.inv_dx = wp.vec3(*sdf._inv_dx)
    collider.sdf = sdf_wp
    return collider


@wp.func
def sdf_query(collider: SDFCollider, pos: wp.vec3) -> float:
    """
    Trilinear interpolation of the SDF at *pos*.

    Returns signed distance (positive = outside, negative = inside).
    """
    gx = (pos[0] - collider.grid_min[0]) * collider.inv_dx[0]
    gy = (pos[1] - collider.grid_min[1]) * collider.inv_dx[1]
    gz = (pos[2] - collider.grid_min[2]) * collider.inv_dx[2]

    # integer (base) and fractional parts
    ix = wp.int(wp.floor(gx))
    iy = wp.int(wp.floor(gy))
    iz = wp.int(wp.floor(gz))

    fx = gx - wp.float(ix)
    fy = gy - wp.float(iy)
    fz = gz - wp.float(iz)

    # clamp to valid range
    R = collider.grid_res
    def _clamp(v: int) -> int:
        return wp.min(wp.max(v, 0), R)

    i000 = _clamp(ix)
    i001 = _clamp(ix)
    i010 = _clamp(ix)
    i011 = _clamp(ix)
    i100 = _clamp(ix + 1)
    i101 = _clamp(ix + 1)
    i110 = _clamp(ix + 1)
    i111 = _clamp(ix + 1)

    j000 = _clamp(iy)
    j001 = _clamp(iy + 1)
    j010 = _clamp(iy + 1)
    j011 = _clamp(iy + 1)
    j100 = _clamp(iy)
    j101 = _clamp(iy + 1)
    j110 = _clamp(iy + 1)
    j111 = _clamp(iy + 1)

    k000 = _clamp(iz)
    k001 = _clamp(iz)
    k010 = _clamp(iz)
    k011 = _clamp(iz)
    k100 = _clamp(iz)
    k101 = _clamp(iz)
    k110 = _clamp(iz)
    k111 = _clamp(iz + 1)

    c000 = collider.sdf[i000, j000, k000]
    c100 = collider.sdf[i100, j100, k100]
    c010 = collider.sdf[i010, j010, k010]
    c110 = collider.sdf[i110, j110, k110]
    c001 = collider.sdf[i001, j001, k001]
    c101 = collider.sdf[i101, j101, k101]
    c011 = collider.sdf[i011, j011, k011]
    c111 = collider.sdf[i111, j111, k111]

    c00 = c000 + (c100 - c000) * fx
    c10 = c010 + (c110 - c010) * fx
    c01 = c001 + (c101 - c001) * fx
    c11 = c011 + (c111 - c011) * fx

    c0 = c00 + (c10 - c00) * fy
    c1 = c01 + (c11 - c01) * fy
    return c0 + (c1 - c0) * fz


@wp.func
def sdf_gradient(
    collider: SDFCollider, pos: wp.vec3, eps: float
) -> wp.vec3:
    """Finite-difference gradient of the SDF (normalised)."""
    dx = wp.vec3(eps, 0.0, 0.0)
    dy = wp.vec3(0.0, eps, 0.0)
    dz = wp.vec3(0.0, 0.0, eps)

    gx = (sdf_query(collider, pos + dx) - sdf_query(collider, pos - dx)) / (2.0 * eps)
    gy = (sdf_query(collider, pos + dy) - sdf_query(collider, pos - dy)) / (2.0 * eps)
    gz = (sdf_query(collider, pos + dz) - sdf_query(collider, pos - dz)) / (2.0 * eps)

    n = wp.sqrt(gx * gx + gy * gy + gz * gz)
    n = wp.max(n, 1e-10)
    return wp.vec3(gx / n, gy / n, gz / n)


@wp.kernel
def apply_sdf_collision_kernel(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    collider: SDFCollider,
    dt: float,
    restitution: float,
    friction: float,
):
    """
    Simple SDF collision response for MPM particles.

    For each particle that has penetrated the SDF (sdf < 0), the kernel:
      1. Computes the SDF gradient = surface normal
      2. Pushes the particle to the surface along the normal
      3. Reflects the normal component of velocity (with restitution)
      4. Applies Coulomb friction to the tangential component
    """
    p = wp.tid()
    pos = particle_x[p]
    s = sdf_query(collider, pos)

    if s >= 0.0:
        return

    # surface normal pointing outward
    n = sdf_gradient(collider, pos, 1e-4)

    # push particle out to surface
    pos = pos - s * n
    particle_x[p] = pos

    # velocity decomposition
    vn = wp.dot(particle_v[p], n)          # normal component
    vt = particle_v[p] - vn * n            # tangential component

    # reflect normal with restitution
    if vn < 0.0:
        vn = -restitution * vn

    # friction clamp on tangential
    vn_mag = wp.max(wp.abs(vn), 1e-10)
    vt_mag = wp.length(vt)
    if vt_mag > friction * vn_mag:
        vt = (friction * vn_mag / vt_mag) * vt

    particle_v[p] = vt + vn * n


# =========================================================================
# CLI entry point  (python -m utils.mesh_sdf ...)
# =========================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert an OBJ mesh to SDF")
    parser.add_argument("obj_path", help="Path to input .obj file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .npz path (default: <input>_sdf_<res>.npz)")
    parser.add_argument("--grid-res", "-r", type=int, default=64,
                        help="Grid resolution (default: 64)")
    parser.add_argument("--padding", "-p", type=float, default=0.1,
                        help="Padding around mesh bounds (default: 0.1)")
    parser.add_argument("--margin", "-m", type=int, default=3,
                        help="Triangle bounding box margin in cells (default: 3)")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Print a quick slice check instead of visualising")
    args = parser.parse_args()

    obj_path = args.obj_path
    if args.output is None:
        base = os.path.splitext(obj_path)[0]
        args.output = f"{base}_sdf_r{args.grid_res}.npz"

    # Build SDF
    sdf = MeshSDF.from_obj(
        obj_path,
        grid_res=args.grid_res,
        padding=args.padding,
        margin_cells=args.margin,
    )
    sdf.save(args.output)

    # Quick validation
    test_pts = np.array([
        sdf.grid_min + 0.5 * (sdf.grid_max - sdf.grid_min),  # centre
        sdf.grid_min + 0.1 * (sdf.grid_max - sdf.grid_min),  # near corner
        sdf.grid_max - 0.1 * (sdf.grid_max - sdf.grid_min),  # far corner
    ], dtype=np.float32)
    vals = sdf.query_numpy(test_pts)
    print("\nValidation queries (3 test points):")
    for i, (p, v) in enumerate(zip(test_pts, vals)):
        print(f"  [{i}] pos {p}  →  sdf = {v:.4f}")

    # Check a few known-outside points
    far_out = sdf.grid_min - 0.5 * (sdf.grid_max - sdf.grid_min)
    val_out = sdf.query_numpy(far_out[None])[0]
    print(f"  Far outside point  →  sdf = {val_out:.4f}  (should be > 0)")


if __name__ == "__main__":
    main()
