"""
traffic_aggregator.py
---------------------
Cuenta vehículos únicos por tipo durante cada ventana de tiempo
y calcula la media de apariciones por minuto para cada tipo.

Lógica de conteo:
  - Cada track se cuenta UNA SOLA VEZ por ventana, la primera vez que aparece
    visible (estado distinto de LOST), independientemente de si está en
    movimiento o parado.
  - Al cerrar la ventana se guarda:
      · count_motorcycle / count_car / count_heavy  (únicos vistos)
      · mean_per_min_*  (promedio de vehículos visibles por minuto,
                         calculado frame a frame)
"""

from __future__ import annotations

import time
from typing import Dict, List

from data_structures import Track, WindowFeatures


class TrafficAggregator:

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = window_seconds
        self._window_start: float = time.time()

        # --- Unique vehicle IDs seen this window --------------------------
        self._seen_ids_motorcycle: set = set()
        self._seen_ids_car:        set = set()
        self._seen_ids_heavy:      set = set()

        # --- Per-frame visible counts (para calcular la media por minuto) -
        self._frame_counts_motorcycle: List[int] = []
        self._frame_counts_car:        List[int] = []
        self._frame_counts_heavy:      List[int] = []

        # --- FPS muestras -------------------------------------------------
        self._fps_samples: List[float] = []

        self._last_update_time: float = time.time()

    # ------------------------------------------------------------------
    def update(
        self,
        active_tracks: Dict[int, Track],
        timestamp: float,
        fps: float,
    ) -> None:
        """Llamar una vez por frame. active_tracks: tracks no-LOST del tracker."""
        self._fps_samples.append(fps)

        vis_motorcycle = 0
        vis_car        = 0
        vis_heavy      = 0

        for tid, tr in active_tracks.items():
            # Contar como único si aún no lo hemos registrado esta ventana
            if not tr.is_counted_in_current_window:
                tr.is_counted_in_current_window = True
                if tr.project_class == "light_vehicle":
                    self._seen_ids_motorcycle.add(tid)
                elif tr.project_class == "medium_vehicle":
                    self._seen_ids_car.add(tid)
                elif tr.project_class == "heavy_vehicle":
                    self._seen_ids_heavy.add(tid)

            # Contabilizar visibilidad de este frame
            if tr.project_class == "light_vehicle":
                vis_motorcycle += 1
            elif tr.project_class == "medium_vehicle":
                vis_car += 1
            elif tr.project_class == "heavy_vehicle":
                vis_heavy += 1

        self._frame_counts_motorcycle.append(vis_motorcycle)
        self._frame_counts_car.append(vis_car)
        self._frame_counts_heavy.append(vis_heavy)

        self._last_update_time = timestamp

    # ------------------------------------------------------------------
    def get_live_snapshot(self, active_tracks: Dict[int, Track]) -> dict:
        """Snapshot rápido para mostrar en pantalla / terminal."""
        motorcycle = sum(1 for tr in active_tracks.values()
                         if tr.project_class == "light_vehicle")
        car        = sum(1 for tr in active_tracks.values()
                         if tr.project_class == "medium_vehicle")
        heavy      = sum(1 for tr in active_tracks.values()
                         if tr.project_class == "heavy_vehicle")
        return {
            "motorcycle": motorcycle,
            "car":        car,
            "heavy":      heavy,
            "total":      motorcycle + car + heavy,
        }

    # ------------------------------------------------------------------
    def close_window(
        self,
        active_tracks: Dict[int, Track],
        window_end: float,
    ) -> WindowFeatures:
        """
        Cierra la ventana actual y devuelve un WindowFeatures con los
        conteos y medias por minuto. Resetea los acumuladores internos.
        """
        actual_seconds = window_end - self._window_start
        if actual_seconds <= 0:
            actual_seconds = self.window_seconds

        # --- Conteos únicos -----------------------------------------------
        count_motorcycle = len(self._seen_ids_motorcycle)
        count_car        = len(self._seen_ids_car)
        count_heavy      = len(self._seen_ids_heavy)

        # --- Media de vehículos visibles por minuto -----------------------
        # Promedio frame-a-frame de cuántos eran visibles, escalado a /min.
        # Ejemplo: si de media había 2 coches visibles en cada frame y la
        # ventana duró 60 s → mean_per_min_car = 2.0
        # Si duró 30 s → mean_per_min_car = 2.0 igualmente (ya normalizado)
        n_frames = len(self._frame_counts_motorcycle)
        scale    = 60.0 / actual_seconds  # factor para normalizar a /minuto

        if n_frames > 0:
            mean_per_min_motorcycle = (sum(self._frame_counts_motorcycle) / n_frames) * scale
            mean_per_min_car        = (sum(self._frame_counts_car)        / n_frames) * scale
            mean_per_min_heavy      = (sum(self._frame_counts_heavy)      / n_frames) * scale
        else:
            mean_per_min_motorcycle = mean_per_min_car = mean_per_min_heavy = 0.0

        # --- FPS ----------------------------------------------------------
        fps_mean = (
            sum(self._fps_samples) / len(self._fps_samples)
            if self._fps_samples else 0.0
        )

        wf = WindowFeatures(
            window_start=self._window_start,
            window_end=window_end,
            window_seconds=round(actual_seconds, 1),
            count_motorcycle=count_motorcycle,
            count_car=count_car,
            count_heavy=count_heavy,
            mean_per_min_motorcycle=mean_per_min_motorcycle,
            mean_per_min_car=mean_per_min_car,
            mean_per_min_heavy=mean_per_min_heavy,
            fps_mean=fps_mean,
        )

        self._reset_window(window_end, active_tracks)
        return wf

    # ------------------------------------------------------------------
    def _reset_window(self, new_start: float, active_tracks: Dict[int, Track]) -> None:
        self._window_start            = new_start
        self._seen_ids_motorcycle     = set()
        self._seen_ids_car            = set()
        self._seen_ids_heavy          = set()
        self._frame_counts_motorcycle = []
        self._frame_counts_car        = []
        self._frame_counts_heavy      = []
        self._fps_samples             = []

        for tr in active_tracks.values():
            tr.is_counted_in_current_window = False

    # ------------------------------------------------------------------
    @property
    def window_start(self) -> float:
        return self._window_start

    def seconds_since_window_start(self, now: float) -> float:
        return now - self._window_start