#!/usr/bin/env python3
"""Dzanga-Sangha elephant / poacher / acoustic-sensor simulation.

Python port of large_scale_sim_FINAL_ver_5_REAL.m.

Run desktop app:
    python dzanga_sim.py

Run a non-GUI smoke test:
    python dzanga_sim.py --headless --steps 500

The optional background image ``dzanga_sangha_updated_black.png`` may be placed
beside this script.  If it is absent, the app uses a dark green background.
"""
from __future__ import annotations

import argparse
import math
import random
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageTk


# -----------------------------------------------------------------------------
# Configuration (kept equivalent to the MATLAB model)
# -----------------------------------------------------------------------------
WIDTH = 600
HEIGHT = 600
SIDEBAR_W = 240
REF_WIDTH = 1000
REF_HEIGHT = 1000
SF_X = WIDTH / REF_WIDTH
SF_Y = HEIGHT / REF_HEIGHT

METERS_PER_PIXEL = 158.0
SPEED_ELEPHANT_MPS = 1.11
SPEED_POACHER_MPS = 0.55
SPEED_RANGER_MPS = 5.55
DETECTION_PROBABILITY = 0.90
POACH_PROBABILITY = 0.85
SCAN_INTERVAL_SIM = 1.0

DEFAULT_MIC_STRATEGY = 1
DEFAULT_NUM_MICS = 80
NUM_ELEPHANTS = 20
NUM_POACHERS = 10
POOL_SIZE = 2000

RAD_NOLA_M = 3040
RAD_SALO_M = 2280
RAD_BAYANGA_M = 3040
RAD_LIDJOMBO_M = 2280
RAD_MOSSIPA_M = 2280
BAI_RAD_M = 500
BAI_SENSE_M = 6080
MIC_RANGE_E_M = 3950
MIC_RANGE_P_M = 200
POACH_DIST_M = 100
DIST_REACHED_M = 1266
AVOID_BUFFER_M = 2533
DEEP_FOREST_M = 10000

STRATEGY_NAMES = {
    1: "Uniform Spread",
    2: "Targeted Fortress",
    3: "Perimeter Defense",
    4: "Combination Fortress & Perimeter",
}


def m2px(meters: float) -> float:
    return meters / METERS_PER_PIXEL


def norm2(dx: float, dy: float) -> float:
    return math.hypot(dx, dy)


def safe_unit(dx: float, dy: float) -> Tuple[float, float]:
    mag = math.hypot(dx, dy)
    if mag <= 1e-12:
        return 0.0, 0.0
    return dx / mag, dy / mag


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
@dataclass
class Zone:
    x: float
    y: float
    radius: float
    name: str


@dataclass
class Attractor:
    x: float
    y: float
    radius: float
    sensing_range: float


@dataclass
class Elephant:
    x: float
    y: float
    target_x: float
    target_y: float
    radius: float = 8 * SF_X
    is_poached: bool = False
    poach_time: float = -999.0
    is_threatened: bool = False
    encounter_rolled: bool = False
    state: str = "ROAMING"
    attractor_entry_time: float = 0.0
    last_visit_time: float = -999.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class Poacher:
    x: float
    y: float
    target_x: float
    target_y: float
    radius: float = 8 * SF_X
    vx: float = 0.0
    vy: float = 0.0
    is_caught: bool = False
    caught_time: float = -999.0
    is_targeted: bool = False
    ranger_x: float = -999.0
    ranger_y: float = -999.0
    base_x: float = -999.0
    base_y: float = -999.0


@dataclass
class Mic:
    x: float
    y: float
    active_e: bool = False
    active_p: bool = False
    threat: bool = False
    last_scan: float = 0.0
    missed: bool = False
    has_memory: bool = False
    elephant_memory: float = -999999.0


@dataclass
class StepEvents:
    poached_alert: bool = False
    caught_alert: bool = False


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def points_in_polygon(xs: np.ndarray, ys: np.ndarray,
                      poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    inside = np.zeros(np.broadcast(xs, ys).shape, dtype=bool)
    j = len(poly_x) - 1
    for i in range(len(poly_x)):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        crosses = ((yi > ys) != (yj > ys))
        denom = (yj - yi)
        if abs(denom) < 1e-15:
            x_at_y = np.full_like(xs, np.inf, dtype=float)
        else:
            x_at_y = (xj - xi) * (ys - yi) / denom + xi
        inside ^= crosses & (xs < x_at_y)
        j = i
    return inside


# -----------------------------------------------------------------------------
# Simulation model
# -----------------------------------------------------------------------------
class DzangaSimulation:
    def __init__(self, mic_strategy: int = DEFAULT_MIC_STRATEGY,
                 num_mics: int = DEFAULT_NUM_MICS,
                 seed: Optional[int] = None):
        if mic_strategy not in STRATEGY_NAMES:
            raise ValueError("mic_strategy must be 1, 2, 3, or 4")
        if num_mics < 1:
            raise ValueError("num_mics must be positive")

        self.mic_strategy = mic_strategy
        self.requested_num_mics = num_mics
        self.rng = random.Random(seed)
        self.current_sim_time = 0.0
        self.frame_count = 0

        ref_poly_x = np.array([
            50, 150, 300, 450, 600, 680, 750, 850, 980,
            980, 850, 720, 650, 620, 640, 620, 550, 500,
            450, 380, 350, 350, 320, 250, 200, 100, 50,
            20, 50
        ], dtype=float)
        ref_poly_y = np.array([
            150, 100, 60, 50, 20, 20, 20, 30, 40,
            120, 140, 140, 300, 400, 500, 650, 800, 900,
            1000, 850, 750, 650, 550, 520, 480, 400, 300,
            280, 150
        ], dtype=float)
        self.park_boundary_x = ref_poly_x * SF_X
        self.park_boundary_y = ref_poly_y * SF_Y

        self.repulsors = [
            Zone(320 * SF_X, 220 * SF_Y, m2px(RAD_NOLA_M), "Nola"),
            Zone(380 * SF_X, 430 * SF_Y, m2px(RAD_SALO_M), "Salo"),
            Zone(480 * SF_X, 600 * SF_Y, m2px(RAD_BAYANGA_M), "Bayanga"),
            Zone(380 * SF_X, 750 * SF_Y, m2px(RAD_LIDJOMBO_M), "Lidjombo"),
            Zone(190 * SF_X, 120 * SF_Y, m2px(RAD_MOSSIPA_M), "Mossipa"),
        ]
        self.attractor = Attractor(
            550 * SF_X,
            550 * SF_Y,
            m2px(BAI_RAD_M),
            m2px(BAI_SENSE_M),
        )
        self.mic_range_e = m2px(MIC_RANGE_E_M)
        self.mic_range_p = m2px(MIC_RANGE_P_M)

        self._build_park_grid()
        self.elephants = self._init_elephants(NUM_ELEPHANTS)
        self.poachers = self._init_poachers(NUM_POACHERS)
        self.pool_general, self.pool_elephant, self.pool_deep = self._make_position_pools()
        self.mics = self._place_mics(mic_strategy, num_mics)

    def _build_park_grid(self) -> None:
        gx, gy = np.meshgrid(np.arange(WIDTH + 1), np.arange(HEIGHT + 1))
        self.park_grid = points_in_polygon(
            gx, gy, self.park_boundary_x, self.park_boundary_y
        )

    def is_inside_park(self, x: float, y: float) -> bool:
        ix = min(max(int(round(x)), 0), WIDTH)
        iy = min(max(int(round(y)), 0), HEIGHT)
        return bool(self.park_grid[iy, ix])

    def _in_red_buffer(self, x: float, y: float, buffer_m: float = 1266) -> bool:
        buffer_px = m2px(buffer_m)
        return any(norm2(x - z.x, y - z.y) < z.radius + buffer_px for z in self.repulsors)

    def _random_elephant_position(self) -> Tuple[float, float]:
        for _ in range(300000):
            x = self.rng.random() * WIDTH
            y = self.rng.random() * HEIGHT
            if self.is_inside_park(x, y) and not self._in_red_buffer(x, y):
                return x, y
        raise RuntimeError("Could not generate valid elephant position")

    def _random_general_position(self) -> Tuple[float, float]:
        for _ in range(300000):
            x = self.rng.random() * WIDTH
            y = self.rng.random() * HEIGHT
            d_green = norm2(x - self.attractor.x, y - self.attractor.y)
            if self.is_inside_park(x, y) and d_green > self.attractor.radius + m2px(AVOID_BUFFER_M):
                return x, y
        raise RuntimeError("Could not generate valid general position")

    def _init_elephants(self, count: int) -> List[Elephant]:
        out: List[Elephant] = []
        for _ in range(count):
            x, y = self._random_elephant_position()
            tx, ty = self._random_elephant_position()
            out.append(Elephant(x=x, y=y, target_x=tx, target_y=ty))
        return out

    def _init_poachers(self, count: int) -> List[Poacher]:
        out: List[Poacher] = []
        for _ in range(count):
            valid = False
            sx = sy = 0.0
            for _attempt in range(300000):
                if self.rng.random() > 0.5:
                    z = self.rng.choice(self.repulsors)
                    sx = z.x + (self.rng.random() - 0.5) * m2px(6000)
                    sy = z.y + (self.rng.random() - 0.5) * m2px(6000)
                    d_green = norm2(sx - self.attractor.x, sy - self.attractor.y)
                    if self.is_inside_park(sx, sy) and d_green > self.attractor.radius + m2px(AVOID_BUFFER_M):
                        valid = True
                else:
                    idx = self.rng.randrange(len(self.park_boundary_x))
                    px = self.park_boundary_x[idx]
                    py = self.park_boundary_y[idx]
                    sx = px + (self.rng.random() - 0.5) * m2px(5000)
                    sy = py + (self.rng.random() - 0.5) * m2px(5000)
                    if not self.is_inside_park(sx, sy):
                        valid = True
                if valid:
                    break
            if not valid:
                raise RuntimeError("Could not generate valid poacher start")
            tx, ty = self._random_general_position()
            out.append(Poacher(x=sx, y=sy, target_x=tx, target_y=ty))
        return out

    def _make_position_pools(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        general: List[Tuple[float, float]] = []
        elephant: List[Tuple[float, float]] = []
        deep: List[Tuple[float, float]] = []

        attempts = 0
        while len(general) < POOL_SIZE and attempts < 200000:
            x = self.rng.random() * WIDTH
            y = self.rng.random() * HEIGHT
            d_green = norm2(x - self.attractor.x, y - self.attractor.y)
            if self.is_inside_park(x, y) and d_green > self.attractor.radius + m2px(AVOID_BUFFER_M):
                general.append((x, y))
            attempts += 1

        attempts = 0
        while len(elephant) < POOL_SIZE and attempts < 200000:
            x = self.rng.random() * WIDTH
            y = self.rng.random() * HEIGHT
            if self.is_inside_park(x, y) and not self._in_red_buffer(x, y):
                elephant.append((x, y))
            attempts += 1

        attempts = 0
        while len(deep) < POOL_SIZE and attempts < 200000:
            x = self.rng.random() * WIDTH
            y = self.rng.random() * HEIGHT
            if self.is_inside_park(x, y) and norm2(x - self.attractor.x, y - self.attractor.y) > m2px(DEEP_FOREST_M):
                deep.append((x, y))
            attempts += 1

        if not general or not elephant:
            raise RuntimeError("Unable to build valid position pools")
        if not deep:
            deep = elephant.copy()

        return np.asarray(general), np.asarray(elephant), np.asarray(deep)

    def _sample_pool(self, pool: np.ndarray) -> Tuple[float, float]:
        row = pool[self.rng.randrange(len(pool))]
        return float(row[0]), float(row[1])

    def _candidate_grid(self, step: int = 10) -> np.ndarray:
        vals = []
        for x in range(step, WIDTH + 1, step):
            for y in range(step, HEIGHT + 1, step):
                if self.is_inside_park(x, y):
                    vals.append((float(x), float(y)))
        if not vals:
            return np.empty((0, 2), dtype=float)
        return np.asarray(vals, dtype=float)

    @staticmethod
    def _mic_at(x: float, y: float) -> Mic:
        return Mic(x=float(x), y=float(y))

    def _fps_fill(self, existing: List[Mic], target_count: int,
                  candidates: np.ndarray) -> List[Mic]:
        if len(candidates) == 0:
            return existing
        if not existing:
            centroid = np.mean(candidates, axis=0)
            d = np.linalg.norm(candidates - centroid, axis=1)
            seed_idx = int(np.argmin(d))
            existing.append(self._mic_at(*candidates[seed_idx]))
            min_dists = np.linalg.norm(candidates - candidates[seed_idx], axis=1)
        else:
            min_dists = np.full(len(candidates), np.inf)
            for mic in existing:
                d = np.linalg.norm(candidates - np.array([mic.x, mic.y]), axis=1)
                min_dists = np.minimum(min_dists, d)

        while len(existing) < target_count:
            best_idx = int(np.argmax(min_dists))
            pt = candidates[best_idx]
            existing.append(self._mic_at(*pt))
            d_new = np.linalg.norm(candidates - pt, axis=1)
            min_dists = np.minimum(min_dists, d_new)
        return existing

    def _place_mics_uniform(self, num_mics: int) -> List[Mic]:
        return self._fps_fill([], num_mics, self._candidate_grid())

    def _place_mics_fortress(self, num_mics: int) -> List[Mic]:
        mics: List[Mic] = []
        budget_village = round(num_mics * 0.75)
        mics_per_village = budget_village // len(self.repulsors)
        if mics_per_village > 0:
            for z in self.repulsors:
                ring_radius = z.radius + self.mic_range_p
                for i in range(1, mics_per_village + 1):
                    angle = (i / mics_per_village) * 2 * math.pi
                    mx = z.x + ring_radius * math.cos(angle)
                    my = z.y + ring_radius * math.sin(angle)
                    if self.is_inside_park(mx, my) and len(mics) < num_mics:
                        mics.append(self._mic_at(mx, my))
        return self._fps_fill(mics, num_mics, self._candidate_grid())

    def _perimeter_points(self, num_points: int) -> List[Tuple[float, float]]:
        # Preserve the MATLAB S3 algorithm, which walks boundary segments using
        # the open polyline as provided in the source.
        x = self.park_boundary_x
        y = self.park_boundary_y
        total_length = sum(norm2(x[i + 1] - x[i], y[i + 1] - y[i]) for i in range(len(x) - 1))
        spacing = total_length / max(num_points, 1)
        curr_dist = 0.0
        segment = 0
        points: List[Tuple[float, float]] = []
        while len(points) < num_points and segment < len(x) - 1:
            p1x, p1y = x[segment], y[segment]
            p2x, p2y = x[segment + 1], y[segment + 1]
            dx, dy = p2x - p1x, p2y - p1y
            seg_len = norm2(dx, dy)
            while curr_dist + spacing <= seg_len and len(points) < num_points:
                curr_dist += spacing
                ratio = curr_dist / seg_len
                mx = p1x + ratio * dx
                my = p1y + ratio * dy
                ux, uy = dx / seg_len, dy / seg_len
                normal_x, normal_y = -uy, ux
                mx += normal_x * self.mic_range_p * 0.5
                my += normal_y * self.mic_range_p * 0.5
                points.append((mx, my))
            curr_dist -= seg_len
            segment += 1
        return points

    def _place_mics_perimeter(self, num_mics: int) -> List[Mic]:
        return [self._mic_at(x, y) for x, y in self._perimeter_points(num_mics)]

    def _place_mics_optimized(self, num_mics: int) -> List[Mic]:
        mics: List[Mic] = []
        half_mics = num_mics // 2
        mics_per_village = half_mics // len(self.repulsors)
        if mics_per_village > 0:
            for z in self.repulsors:
                ring_radius = z.radius + self.mic_range_p
                for i in range(1, mics_per_village + 1):
                    angle = (i / mics_per_village) * 2 * math.pi
                    mx = z.x + ring_radius * math.cos(angle)
                    my = z.y + ring_radius * math.sin(angle)
                    if self.is_inside_park(mx, my) and len(mics) < half_mics:
                        mics.append(self._mic_at(mx, my))

        # MATLAB phase 2 continues count toward num_mics using spacing based on half_mics.
        perim = self._perimeter_points(max(half_mics, 1))
        for x, y in perim:
            if len(mics) >= num_mics:
                break
            mics.append(self._mic_at(x, y))

        if len(mics) < num_mics:
            mics = self._fps_fill(mics, num_mics, self._candidate_grid())
        return mics

    def _place_mics(self, strategy: int, num_mics: int) -> List[Mic]:
        if strategy == 1:
            return self._place_mics_uniform(num_mics)
        if strategy == 2:
            return self._place_mics_fortress(num_mics)
        if strategy == 3:
            return self._place_mics_perimeter(num_mics)
        return self._place_mics_optimized(num_mics)

    def step(self, real_dt: float, time_multiplier: float) -> StepEvents:
        if real_dt <= 0 or time_multiplier <= 0:
            return StepEvents()

        self.frame_count += 1
        sim_dt = real_dt * time_multiplier
        self.current_sim_time += sim_dt

        step_e = SPEED_ELEPHANT_MPS * sim_dt / METERS_PER_PIXEL
        step_p = SPEED_POACHER_MPS * sim_dt / METERS_PER_PIXEL
        step_r = SPEED_RANGER_MPS * sim_dt / METERS_PER_PIXEL

        self._move_poachers(step_p)
        self._move_elephants(step_e)
        self._scan_mics()
        self._move_rangers(step_r)
        return self._poaching_check()

    def _move_poachers(self, step_p: float) -> None:
        for p in self.poachers:
            if p.is_caught:
                continue
            dx = p.target_x - p.x
            dy = p.target_y - p.y
            dist = norm2(dx, dy)
            if dist < m2px(DIST_REACHED_M):
                p.target_x, p.target_y = self._sample_pool(self.pool_general)
                dx = p.target_x - p.x
                dy = p.target_y - p.y
                dist = norm2(dx, dy)

            des_vx, des_vy = safe_unit(dx, dy)
            d_green = norm2(p.x - self.attractor.x, p.y - self.attractor.y)
            if 1e-12 < d_green < self.attractor.radius + m2px(3000):
                des_vx += ((p.x - self.attractor.x) / d_green) * 4.0
                des_vy += ((p.y - self.attractor.y) / d_green) * 4.0

            des_vx, des_vy = safe_unit(des_vx, des_vy)
            p.vx = p.vx * 0.95 + des_vx * 0.05
            p.vy = p.vy * 0.95 + des_vy * 0.05
            p.vx, p.vy = safe_unit(p.vx, p.vy)

            was_inside = self.is_inside_park(p.x, p.y)
            nx = p.x + p.vx * step_p
            ny = p.y + p.vy * step_p
            is_inside_now = self.is_inside_park(nx, ny)
            if was_inside and not is_inside_now:
                p.vx = -p.vx
                p.vy = -p.vy
                p.target_x, p.target_y = self._sample_pool(self.pool_general)
            else:
                p.x, p.y = nx, ny

    def _move_elephants(self, step_e: float) -> None:
        for e in self.elephants:
            if e.is_poached:
                continue
            dx = e.target_x - e.x
            dy = e.target_y - e.y
            dist = norm2(dx, dy)
            d_bai = norm2(e.x - self.attractor.x, e.y - self.attractor.y)

            if e.state == "ROAMING":
                if self.current_sim_time - e.last_visit_time > 60 and d_bai < self.attractor.sensing_range:
                    e.state = "SEEKING"
            elif e.state == "SEEKING":
                if d_bai < self.attractor.radius:
                    e.state = "WAITING"
                    e.attractor_entry_time = self.current_sim_time
            elif e.state == "WAITING":
                if self.current_sim_time - e.attractor_entry_time > 10:
                    e.state = "ROAMING"
                    e.last_visit_time = self.current_sim_time
                    e.target_x, e.target_y = self._sample_pool(self.pool_deep)

            if dist < m2px(AVOID_BUFFER_M) and e.state != "WAITING":
                e.target_x, e.target_y = self._sample_pool(self.pool_elephant)
                dx = e.target_x - e.x
                dy = e.target_y - e.y
                dist = norm2(dx, dy)

            if e.state == "WAITING":
                des_vx = des_vy = 0.0
            else:
                des_vx, des_vy = safe_unit(e.target_x - e.x, e.target_y - e.y)

            if e.state == "SEEKING" and d_bai > 1e-12:
                des_vx += ((self.attractor.x - e.x) / d_bai) * 0.8
                des_vy += ((self.attractor.y - e.y) / d_bai) * 0.8

            for z in self.repulsors:
                d_rep = norm2(e.x - z.x, e.y - z.y)
                if 1e-12 < d_rep < z.radius + m2px(AVOID_BUFFER_M):
                    des_vx += ((e.x - z.x) / d_rep) * 4.0
                    des_vy += ((e.y - z.y) / d_rep) * 4.0

            des_vx, des_vy = safe_unit(des_vx, des_vy)
            e.vx = e.vx * 0.95 + des_vx * 0.05
            e.vy = e.vy * 0.95 + des_vy * 0.05
            e.vx, e.vy = safe_unit(e.vx, e.vy)

            nx = e.x + e.vx * step_e
            ny = e.y + e.vy * step_e
            if self.is_inside_park(nx, ny):
                e.x, e.y = nx, ny
            else:
                e.vx = -e.vx
                e.vy = -e.vy
                e.target_x, e.target_y = self._sample_pool(self.pool_elephant)

    def _scan_mics(self) -> None:
        for mic in self.mics:
            if self.current_sim_time - mic.last_scan < SCAN_INTERVAL_SIM:
                continue
            mic.last_scan = self.current_sim_time
            mic.active_e = False
            mic.active_p = False
            mic.threat = False
            mic.missed = False
            e_in_range = False
            p_in_range = False

            for e in self.elephants:
                if e.is_poached:
                    continue
                if norm2(e.x - mic.x, e.y - mic.y) < self.mic_range_e:
                    e_in_range = True
                    if self.rng.random() <= DETECTION_PROBABILITY:
                        mic.active_e = True
                        mic.elephant_memory = self.current_sim_time

            for p in self.poachers:
                if p.is_caught:
                    continue
                if norm2(p.x - mic.x, p.y - mic.y) < self.mic_range_p:
                    p_in_range = True
                    if self.rng.random() <= DETECTION_PROBABILITY:
                        mic.active_p = True

            if (e_in_range and not mic.active_e) or (p_in_range and not mic.active_p):
                mic.missed = True

            mic.has_memory = self.current_sim_time - mic.elephant_memory <= 14400

            if mic.active_p and (mic.active_e or mic.has_memory):
                for p in self.poachers:
                    if p.is_caught or p.is_targeted:
                        continue
                    if norm2(p.x - mic.x, p.y - mic.y) < self.mic_range_p:
                        mic.threat = True
                        p.is_targeted = True
                        closest = min(
                            self.repulsors,
                            key=lambda z: norm2(p.x - z.x, p.y - z.y),
                        )
                        p.base_x = closest.x
                        p.base_y = closest.y
                        p.ranger_x = closest.x
                        p.ranger_y = closest.y

    def _move_rangers(self, step_r: float) -> None:
        for p in self.poachers:
            if not p.is_targeted or p.is_caught:
                continue
            dx = p.x - p.ranger_x
            dy = p.y - p.ranger_y
            dist = norm2(dx, dy)
            if dist < m2px(500):
                p.is_caught = True
                p.caught_time = self.current_sim_time
            else:
                ux, uy = safe_unit(dx, dy)
                p.ranger_x += ux * step_r
                p.ranger_y += uy * step_r

    def _poaching_check(self) -> StepEvents:
        events = StepEvents()
        for p in self.poachers:
            if p.is_caught and self.current_sim_time - p.caught_time < 3.0:
                events.caught_alert = True

        for e in self.elephants:
            if e.is_poached:
                if self.current_sim_time - e.poach_time < 3.0:
                    events.poached_alert = True
                continue

            e.is_threatened = False
            elephant_in_range = False
            for p in self.poachers:
                if p.is_caught:
                    continue
                d_ep = norm2(e.x - p.x, e.y - p.y)
                if d_ep < m2px(POACH_DIST_M):
                    elephant_in_range = True
                    d_safe = norm2(e.x - self.attractor.x, e.y - self.attractor.y)
                    if d_safe > self.attractor.radius:
                        e.is_threatened = True
                        if not e.encounter_rolled:
                            e.encounter_rolled = True
                            if self.rng.random() <= POACH_PROBABILITY:
                                e.is_poached = True
                                e.poach_time = self.current_sim_time
                                events.poached_alert = True
                            else:
                                flee_dx = e.x - p.x
                                flee_dy = e.y - p.y
                                flee_dx, flee_dy = safe_unit(flee_dx, flee_dy)
                                flee_dist_px = m2px(AVOID_BUFFER_M) * 1.2
                                e.target_x = e.x + flee_dx * flee_dist_px
                                e.target_y = e.y + flee_dy * flee_dist_px
                                p.target_x, p.target_y = self._sample_pool(self.pool_general)

            if not elephant_in_range:
                e.encounter_rolled = False
        return events

    @property
    def poached_count(self) -> int:
        return sum(e.is_poached for e in self.elephants)

    @property
    def caught_count(self) -> int:
        return sum(p.is_caught for p in self.poachers)

    def report_text(self) -> str:
        sim_days = int(self.current_sim_time // 86400)
        sim_hours = int((self.current_sim_time % 86400) // 3600)
        return (
            f"STRATEGY: {STRATEGY_NAMES[self.mic_strategy]}\n"
            "--------------\n"
            "TIME ELAPSED\n"
            "--------------\n"
            f"{sim_days} Days, {sim_hours:02d} Hrs\n\n"
            "ELEPHANTS\n"
            "--------------\n"
            f"Total: {len(self.elephants)}\n"
            f"Safe/Active: {len(self.elephants) - self.poached_count}\n"
            f"Poached: {self.poached_count}\n\n"
            "POACHERS\n"
            "--------------\n"
            f"Total: {len(self.poachers)}\n"
            f"Active: {len(self.poachers) - self.caught_count}\n"
            f"Neutralized: {self.caught_count}"
        )


# -----------------------------------------------------------------------------
# Tkinter desktop application
# -----------------------------------------------------------------------------
class DzangaApp:
    BG = "#262626"
    MAP_BG = "#19331f"

    def __init__(self, root: tk.Tk, strategy: int, num_mics: int,
                 seed: Optional[int] = None):
        self.root = root
        self.root.title("Dzanga Simulation Dashboard - Python")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        self.seed = seed
        self.sim = DzangaSimulation(strategy, num_mics, seed=seed)
        self.paused = False
        self.last_real_time = time.perf_counter()
        self.status_until = 0.0
        self.status_text = ""
        self.status_color = "white"
        self.background_photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._redraw_static()
        self._create_dynamic_items()
        self._render_dynamic()
        self._tick()

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root, bg=self.BG)
        outer.pack(padx=12, pady=10)

        self.canvas = tk.Canvas(
            outer, width=WIDTH, height=HEIGHT,
            bg=self.MAP_BG, highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="n")

        side = tk.Frame(outer, width=SIDEBAR_W, height=HEIGHT, bg="black")
        side.grid(row=0, column=1, rowspan=2, padx=(10, 0), sticky="ns")
        side.grid_propagate(False)

        tk.Label(side, text="LIVE MISSION REPORT", bg="black", fg="white",
                 font=("TkDefaultFont", 11, "bold")).pack(pady=(8, 4))
        self.report = tk.Label(
            side, text=self.sim.report_text(), justify="left", anchor="nw",
            bg="black", fg="white", font=("Courier New", 10)
        )
        self.report.pack(fill="x", padx=10)

        ttk.Separator(side, orient="horizontal").pack(fill="x", padx=8, pady=8)
        tk.Label(side, text="LEGEND", bg="black", fg="white",
                 font=("TkDefaultFont", 11, "bold")).pack()
        legend = [
            ("●", "#1976ff", "Safe Elephant"),
            ("●", "yellow", "Threatened Elephant"),
            ("×", "red", "Poached Elephant"),
            ("●", "magenta", "Active Poacher"),
            ("■", "white", "Active Ranger"),
            ("★", "#00cc44", "Neutralized Poacher"),
            ("■", "#1976ff", "Mic (Idle)"),
            ("■", "white", "Mic (Detected)"),
            ("■", "#e66600", "Mic (Memory)"),
            ("■", "#cc0000", "Mic (Missed)"),
        ]
        for sym, color, text in legend:
            row = tk.Frame(side, bg="black")
            row.pack(fill="x", padx=12, pady=1)
            tk.Label(row, text=sym, width=2, bg="black", fg=color,
                     font=("TkDefaultFont", 11, "bold")).pack(side="left")
            tk.Label(row, text=text, bg="black", fg="white",
                     font=("TkDefaultFont", 9, "bold")).pack(side="left")

        controls = tk.Frame(self.root, bg=self.BG)
        controls.pack(fill="x", padx=12, pady=(0, 10))

        tk.Label(controls, text="Time Lapse", bg=self.BG, fg="white",
                 font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.multiplier = tk.DoubleVar(value=1000.0)
        self.slider = tk.Scale(
            controls, from_=1, to=5000, resolution=1,
            orient="horizontal", variable=self.multiplier,
            length=300, bg=self.BG, fg="white", troughcolor="#555555",
            highlightthickness=0
        )
        self.slider.grid(row=1, column=0, padx=(0, 10))
        self.mult_label = tk.Label(controls, text="1000x", width=8,
                                  bg=self.BG, fg="white",
                                  font=("TkDefaultFont", 10, "bold"))
        self.mult_label.grid(row=1, column=1)

        self.pause_button = tk.Button(
            controls, text="PAUSE ||", command=self.toggle_pause,
            bg="#cc3333", fg="white", width=10,
            font=("TkDefaultFont", 9, "bold")
        )
        self.pause_button.grid(row=1, column=2, padx=8)

        tk.Label(controls, text="Mic Strategy", bg=self.BG, fg="white").grid(row=0, column=3)
        self.strategy_var = tk.StringVar(value=str(self.sim.mic_strategy))
        strategy_box = ttk.Combobox(
            controls, width=3, state="readonly", textvariable=self.strategy_var,
            values=("1", "2", "3", "4")
        )
        strategy_box.grid(row=1, column=3, padx=(10, 3))

        tk.Label(controls, text="Mics", bg=self.BG, fg="white").grid(row=0, column=4)
        self.mics_var = tk.StringVar(value=str(self.sim.requested_num_mics))
        tk.Spinbox(controls, from_=1, to=500, width=5, textvariable=self.mics_var).grid(row=1, column=4, padx=3)

        tk.Button(controls, text="RESET", command=self.reset_simulation,
                  bg="#444444", fg="white", width=8).grid(row=1, column=5, padx=(8, 0))

    def _load_background(self) -> None:
        image_path = Path(__file__).with_name("dzanga_sangha_updated_black.png")
        if not image_path.exists():
            return
        try:
            # MATLAB uses image(..., 'XData', [0 WIDTH], 'YData', [0 HEIGHT]),
            # which maps the complete source image exactly onto the 600x600 axes.
            # Do the same here.  Tk PhotoImage.subsample() only supports integer
            # factors (e.g. 1000px -> 500px), which caused the old background
            # to sit in the upper-left and no longer align with the overlays.
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                fitted = source.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(fitted)

            self.background_photo = photo
            self.canvas.create_image(
                WIDTH / 2, HEIGHT / 2, image=photo, anchor="center", tags="static"
            )
        except (OSError, tk.TclError):
            self.background_photo = None

    def _redraw_static(self) -> None:
        self.canvas.delete("all")
        self._load_background()

        # Boundary
        coords = []
        for x, y in zip(self.sim.park_boundary_x, self.sim.park_boundary_y):
            coords.extend([x, y])
        coords.extend([self.sim.park_boundary_x[0], self.sim.park_boundary_y[0]])
        self.canvas.create_line(*coords, fill="white", dash=(5, 4), width=1, tags="static")

        # Repulsion zones
        for z in self.sim.repulsors:
            self.canvas.create_oval(z.x - z.radius, z.y - z.radius,
                                    z.x + z.radius, z.y + z.radius,
                                    fill="#662222", outline="", stipple="gray50", tags="static")
            self.canvas.create_text(z.x, z.y, text=z.name, fill="white",
                                    font=("TkDefaultFont", 9, "bold"), tags="static")

        a = self.sim.attractor
        self.canvas.create_oval(a.x - a.radius, a.y - a.radius,
                                a.x + a.radius, a.y + a.radius,
                                fill="#176b2b", outline="#00ff55", width=2,
                                stipple="gray50", tags="static")

        self.canvas.create_text(WIDTH - 15, 560, text="Dzanga-Sangha National Park",
                                anchor="e", fill="white",
                                font=("TkDefaultFont", 10, "bold"), tags="static")
        self.canvas.create_text(WIDTH - 15, 580, text="Area monitored: 4900 sq. Km",
                                anchor="e", fill="#b3d9ff",
                                font=("TkDefaultFont", 10), tags="static")

        # Mic ranges and fixed sensor squares.
        for mic in self.sim.mics:
            re = self.sim.mic_range_e
            rp = self.sim.mic_range_p
            self.canvas.create_oval(mic.x - re, mic.y - re, mic.x + re, mic.y + re,
                                    outline="#2bd9e8", dash=(4, 3), width=1, tags="static")
            self.canvas.create_oval(mic.x - rp, mic.y - rp, mic.x + rp, mic.y + rp,
                                    outline="#ffd04d", width=1, tags="static")

        self.status_id = self.canvas.create_text(
            WIDTH / 2, 50, text="", fill="white",
            font=("TkDefaultFont", 20, "bold"), tags="dynamic"
        )

    def _create_dynamic_items(self) -> None:
        self.mic_ids = []
        for mic in self.sim.mics:
            self.mic_ids.append(self.canvas.create_rectangle(
                mic.x - 3, mic.y - 3, mic.x + 3, mic.y + 3,
                fill="#1976ff", outline="white", tags="dynamic"
            ))

        self.elephant_ids = []
        for e in self.sim.elephants:
            r = max(3, e.radius)
            self.elephant_ids.append(self.canvas.create_oval(
                e.x - r, e.y - r, e.x + r, e.y + r,
                fill="#1976ff", outline="white", tags="dynamic"
            ))

        self.poacher_ids = []
        self.ranger_ids = []
        self.threat_line_ids = []
        for p in self.sim.poachers:
            r = 4
            self.poacher_ids.append(self.canvas.create_oval(
                p.x - r, p.y - r, p.x + r, p.y + r,
                fill="magenta", outline="white", tags="dynamic"
            ))
            self.ranger_ids.append(self.canvas.create_rectangle(
                -20, -20, -10, -10, fill="white", outline="white", tags="dynamic"
            ))
            self.threat_line_ids.append(self.canvas.create_line(
                0, 0, 0, 0, fill="red", dash=(7, 4), width=2,
                state="hidden", tags="dynamic"
            ))

    def reset_simulation(self) -> None:
        try:
            strategy = int(self.strategy_var.get())
            num_mics = int(self.mics_var.get())
        except ValueError:
            return
        num_mics = min(max(num_mics, 1), 500)
        self.sim = DzangaSimulation(strategy, num_mics, seed=self.seed)
        self.last_real_time = time.perf_counter()
        self.paused = False
        self.pause_button.configure(text="PAUSE ||", bg="#cc3333")
        self._redraw_static()
        self._create_dynamic_items()
        self._render_dynamic()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.last_real_time = time.perf_counter()
        if self.paused:
            self.pause_button.configure(text="PLAY >", bg="#33aa55")
        else:
            self.pause_button.configure(text="PAUSE ||", bg="#cc3333")

    def _render_dynamic(self) -> None:
        for item, mic in zip(self.mic_ids, self.sim.mics):
            color = "#1976ff"
            if mic.has_memory:
                color = "#e66600"
            elif mic.active_p or mic.active_e:
                color = "white"
            if mic.missed:
                color = "#cc0000"
            self.canvas.itemconfigure(item, fill=color)

        for item, e in zip(self.elephant_ids, self.sim.elephants):
            r = max(3, e.radius)
            self.canvas.coords(item, e.x - r, e.y - r, e.x + r, e.y + r)
            if e.is_poached:
                # Tk Canvas has no oval 'x' marker, so show a red filled marker plus X overlay styling.
                self.canvas.itemconfigure(item, fill="red", outline="red")
            elif e.is_threatened:
                self.canvas.itemconfigure(item, fill="yellow", outline="white")
            else:
                self.canvas.itemconfigure(item, fill="#1976ff", outline="white")

        for poacher_item, ranger_item, line_item, p in zip(
                self.poacher_ids, self.ranger_ids, self.threat_line_ids, self.sim.poachers):
            if p.is_caught:
                r = 7
                self.canvas.coords(poacher_item, p.x - r, p.y - r, p.x + r, p.y + r)
                self.canvas.itemconfigure(poacher_item, fill="#00cc44", outline="#00cc44")
                self.canvas.coords(ranger_item, -20, -20, -10, -10)
                self.canvas.itemconfigure(line_item, state="hidden")
            else:
                r = 4
                self.canvas.coords(poacher_item, p.x - r, p.y - r, p.x + r, p.y + r)
                self.canvas.itemconfigure(poacher_item, fill="magenta", outline="white")
                if p.is_targeted:
                    rr = 4
                    self.canvas.coords(ranger_item, p.ranger_x - rr, p.ranger_y - rr,
                                       p.ranger_x + rr, p.ranger_y + rr)
                    self.canvas.coords(line_item, p.base_x, p.base_y,
                                       p.ranger_x, p.ranger_y, p.x, p.y)
                    self.canvas.itemconfigure(line_item, state="normal")
                else:
                    self.canvas.coords(ranger_item, -20, -20, -10, -10)
                    self.canvas.itemconfigure(line_item, state="hidden")

        self.report.configure(text=self.sim.report_text())
        self.mult_label.configure(text=f"{self.multiplier.get():.0f}x")
        if time.perf_counter() < self.status_until:
            self.canvas.itemconfigure(self.status_id, text=self.status_text, fill=self.status_color)
        else:
            self.canvas.itemconfigure(self.status_id, text="")

    def _tick(self) -> None:
        now = time.perf_counter()
        real_dt = min(now - self.last_real_time, 0.1)
        self.last_real_time = now
        if not self.paused:
            events = self.sim.step(real_dt, self.multiplier.get())
            if events.poached_alert:
                self.status_text = "POACHED!"
                self.status_color = "#ff4444"
                self.status_until = now + 0.8
            elif events.caught_alert:
                self.status_text = "THREAT NEUTRALIZED!"
                self.status_color = "#55ff55"
                self.status_until = now + 0.8
        self._render_dynamic()
        self.root.after(16, self._tick)


# -----------------------------------------------------------------------------
# CLI / headless test
# -----------------------------------------------------------------------------
def run_headless(strategy: int, num_mics: int, steps: int, seed: Optional[int]) -> None:
    sim = DzangaSimulation(strategy, num_mics, seed=seed)
    # Fixed 60 FPS wall-clock timestep with default 1000x acceleration.
    for _ in range(steps):
        sim.step(1.0 / 60.0, 1000.0)
    print("Headless simulation completed successfully.")
    print(f"Strategy: {STRATEGY_NAMES[strategy]}")
    print(f"Mics placed: {len(sim.mics)}")
    print(f"Simulated time: {sim.current_sim_time:.1f} s")
    print(f"Elephants poached: {sim.poached_count}/{len(sim.elephants)}")
    print(f"Poachers neutralized: {sim.caught_count}/{len(sim.poachers)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dzanga-Sangha simulation Python port")
    parser.add_argument("--strategy", type=int, choices=(1, 2, 3, 4), default=DEFAULT_MIC_STRATEGY)
    parser.add_argument("--mics", type=int, default=DEFAULT_NUM_MICS)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional deterministic random seed")
    parser.add_argument("--headless", action="store_true",
                        help="Run without Tk GUI (useful for testing)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Headless simulation steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.headless:
        run_headless(args.strategy, args.mics, args.steps, args.seed)
        return
    root = tk.Tk()
    DzangaApp(root, args.strategy, args.mics, seed=args.seed)
    root.mainloop()


if __name__ == "__main__":
    main()
