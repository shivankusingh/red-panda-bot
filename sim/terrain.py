"""Lumpy terrain for the walking env (Tier 2: small bumps, rug wrinkles, cables).

Layout (Rudin et al. 2021 style tile grid, seen from above):

    +x (rows = difficulty levels, low bumps -> high bumps)
    ^   [lumpy][lumpy][lumpy][flat ][lumpy][lumpy][lumpy][flat ]   level 9 (up to 1.5 cm)
    |   ...
    |   [lumpy][lumpy][lumpy][flat ][lumpy][lumpy][lumpy][flat ]   level 0 (up to 0.15 cm)
    +--------------------------------------------------------------> +y (columns = types)

Each lump is a smooth cos^2 bell (zero slope at its edge); where lumps overlap the
higher one wins. A flat border surrounds the tiles so a robot never walks off the map.

Heights are stored the way MuJoCo's height field wants them: a (nrow, ncol) grid,
row r <-> y, column c <-> x, values normalized to 0..1 and scaled by max_z.
"""
import numpy as np


def make_heights(tc):
  """Builds the height grid (meters) for terrain_config `tc`.

  Returns (heights[nrow, ncol], half_x, half_y): the grid covers
  x in [-half_x, half_x], y in [-half_y, half_y], centered on the world origin.
  """
  res = tc.resolution
  size_x = tc.n_levels * tc.tile_size + 2 * tc.border   # levels run along x
  size_y = tc.n_cols * tc.tile_size + 2 * tc.border     # types run along y
  ncol = int(round(size_x / res)) + 1
  nrow = int(round(size_y / res)) + 1
  h = np.zeros((nrow, ncol), dtype=np.float32)
  rng = np.random.default_rng(tc.seed)
  tile_px = int(round(tc.tile_size / res))
  border_px = int(round(tc.border / res))
  r_min, r_max = (int(round(r / res)) for r in tc.lump_radius)

  for level in range(tc.n_levels):
    level_h = tc.max_height * (level + 1) / tc.n_levels
    for col in range(tc.n_cols):
      if col in tc.flat_cols:
        continue
      c0 = border_px + level * tile_px          # tile corner in grid cells (x)
      r0 = border_px + col * tile_px            # (y)
      tile = h[r0:r0 + tile_px, c0:c0 + tile_px]
      # How many lumps: enough that their discs cover `lump_cover` of the tile
      # (counting overlaps), with the average lump radius.
      mean_area = np.pi * ((r_min + r_max) / 2) ** 2
      n = int(tc.lump_cover * tile_px * tile_px / mean_area)
      for _ in range(n):
        rad = rng.integers(r_min, r_max + 1)
        # keep each lump inside its tile so tiles of different levels meet at 0
        cy, cx = rng.integers(rad, tile_px - rad, size=2)
        peak = level_h * rng.uniform(0.5, 1.0)
        yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
        d = np.sqrt(xx ** 2 + yy ** 2) / rad
        bell = np.where(d < 1.0, peak * np.cos(0.5 * np.pi * d) ** 2, 0.0)
        patch = tile[cy - rad:cy + rad + 1, cx - rad:cx + rad + 1]
        np.maximum(patch, bell, out=patch)
  return h, size_x / 2, size_y / 2


def tile_center(tc, level, col):
  """World (x, y) of a tile's center. Works on numpy or jax numbers."""
  x = -0.5 * (tc.n_levels * tc.tile_size) + (level + 0.5) * tc.tile_size
  y = -0.5 * (tc.n_cols * tc.tile_size) + (col + 0.5) * tc.tile_size
  return x, y


def ground_height(np_or_jp, data, nrow, ncol, half_x, half_y, max_z, x, y):
  """Terrain height under world point(s) (x, y), matching MuJoCo's surface exactly.

  MuJoCo splits each grid cell into two triangles along the diagonal from
  (c, r) to (c+1, r+1) and collides with those flat triangles, so this does the
  same interpolation (not bilinear). `data` is the flat, 0..1 normalized grid.
  """
  xp = np_or_jp
  dx = 2 * half_x / (ncol - 1)
  dy = 2 * half_y / (nrow - 1)
  fx = xp.clip((x + half_x) / dx, 0.0, ncol - 1.0001)
  fy = xp.clip((y + half_y) / dy, 0.0, nrow - 1.0001)
  c = xp.floor(fx).astype(int)
  r = xp.floor(fy).astype(int)
  u, v = fx - c, fy - r                       # position inside the cell, 0..1
  h00 = data[r * ncol + c]
  h10 = data[r * ncol + c + 1]                # one step in x
  h01 = data[(r + 1) * ncol + c]              # one step in y
  h11 = data[(r + 1) * ncol + c + 1]
  upper = h00 + u * (h11 - h01) + v * (h01 - h00)   # triangle (0,0) (1,1) (0,1)
  lower = h00 + u * (h10 - h00) + v * (h11 - h10)   # triangle (0,0) (1,0) (1,1)
  return xp.where(v > u, upper, lower) * max_z
