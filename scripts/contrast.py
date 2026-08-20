"""Parametric contrast control for learned z-score maps.

The z-score panels were displayed with a hard-coded ``|z| <= 4`` clip, which
risks flattening a real signal: a faint underdrawing sitting at ``|z| ~ 1``
occupies a quarter of the colour ramp and is invisible next to a handful of
saturated pixels. This module makes that clip a parameter, chosen either as a
fixed bound or from the map's own distribution, with an optional gamma
compression that lifts weak-but-present detail without touching what was
computed.

Everything here is applied **after** the z-score maps exist, so different
contrast settings are compared by re-plotting, never by re-predicting.

Typical use — one model::

    scale = ZScale(mode=ZScaleMode.PERCENTILE, percentile=99.5, gamma=0.6)
    fig = plot_zscore(ir_real, mu, sigma, z_scale=scale)

and across models, where a shared limit is what makes the panels comparable::

    scaled, vrange = scale.apply_many(z_by_architecture)
    fig = plot_signal_comparison(ir_real, scaled, vrange=vrange)
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

_EPS = 1e-8


class ZScaleMode(str, Enum):
    """How the display limit of a z-score map is chosen.

    Attributes
    ----------
    FIXED : str
        Use ``ZScale.vmax`` as-is. Comparable across images and runs, since it
        does not depend on the data.
    PERCENTILE : str
        Use a percentile of ``|z|``, so the limit adapts to how anomalous the
        image actually is. Better contrast on a quiet image; not comparable
        across images unless the limit is computed once and shared.
    """

    FIXED = "fixed"
    PERCENTILE = "percentile"


@dataclass(frozen=True)
class ScaledZScore:
    """A z-score map prepared for display, with the range it should be shown in.

    Attributes
    ----------
    values : np.ndarray
        Map to hand to ``imshow``. Raw z-scores when ``gamma == 1``; gamma
        compressed to ``[-1, 1]`` otherwise.
    vmin, vmax : float
        Colour-scale bounds for ``values``.
    limit : float
        The ``|z|`` that maps to the edge of the colour scale, in z units,
        whatever ``values`` is expressed in.
    label : str
        Human-readable description of the transform, for a panel title or
        colorbar — so a figure always states the contrast it was rendered at.
    """

    values: np.ndarray
    vmin: float
    vmax: float
    limit: float
    label: str


@dataclass(frozen=True)
class ZScale:
    """Contrast settings for rendering a z-score map.

    The default is ``FIXED`` at ``4.0`` with no gamma, i.e. the plain
    ``|z| <= 4`` clip in z units with an interpretable colorbar.

    Attributes
    ----------
    mode : ZScaleMode
        How the display limit is chosen.
    vmax : float
        The limit itself in ``FIXED`` mode; also the fallback in
        ``PERCENTILE`` mode when the map is degenerate (all-zero z).
    percentile : float
        Percentile of ``|z|`` used as the limit in ``PERCENTILE`` mode.
    gamma : float
        Exponent applied to ``|z| / limit``. ``1.0`` leaves the map in z units.
        Below ``1.0`` expands weak values and compresses strong ones — the
        setting for faint underdrawings; above ``1.0`` does the opposite,
        isolating only the strongest anomalies.

    Raises
    ------
    ValueError
        If ``vmax`` or ``gamma`` is not strictly positive, or ``percentile``
        falls outside ``(0, 100]``.
    """

    mode: ZScaleMode = ZScaleMode.FIXED
    vmax: float = 4.0
    percentile: float = 99.0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.vmax <= 0:
            raise ValueError(f"vmax must be > 0, got {self.vmax}.")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}.")
        if not 0 < self.percentile <= 100:
            raise ValueError(f"percentile must be in (0, 100], got {self.percentile}.")

    def limit(self, *z_maps: np.ndarray) -> float:
        """Compute the ``|z|`` that should map to the edge of the colour scale.

        Parameters
        ----------
        *z_maps : np.ndarray
            One or more z-score maps. In ``PERCENTILE`` mode they are pooled,
            yielding a single limit shared by all of them; ignored in
            ``FIXED`` mode.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If ``PERCENTILE`` mode is used without any map to measure.
        """
        if self.mode is ZScaleMode.FIXED:
            return float(self.vmax)
        if not z_maps:
            raise ValueError(
                "PERCENTILE mode needs at least one z-score map to measure."
            )
        pooled = np.concatenate([np.abs(z).ravel() for z in z_maps])
        limit = float(np.percentile(pooled, self.percentile))
        return limit if limit > _EPS else float(self.vmax)

    def apply(self, z: np.ndarray, limit: float | None = None) -> ScaledZScore:
        """Prepare one z-score map for display.

        Parameters
        ----------
        z : np.ndarray
            Signed z-score map, shape ``(H, W)``.
        limit : float or None
            Display limit to use. Pass a shared limit (from :meth:`limit`) when
            several maps must stay visually comparable; ``None`` derives it
            from this map alone.

        Returns
        -------
        ScaledZScore
        """
        bound = self.limit(z) if limit is None else float(limit)

        if self.gamma == 1.0:
            return ScaledZScore(
                values=z,
                vmin=-bound,
                vmax=bound,
                limit=bound,
                label=self._label(bound),
            )

        compressed = np.sign(z) * np.minimum(np.abs(z) / bound, 1.0) ** self.gamma
        return ScaledZScore(
            values=compressed,
            vmin=-1.0,
            vmax=1.0,
            limit=bound,
            label=self._label(bound),
        )

    def apply_many(
        self, z_maps: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], tuple[float, float]]:
        """Scale several z-score maps against one shared limit.

        The cross-model case: each architecture's z-score rendered on the same
        colour scale, so brighter really does mean more anomalous rather than
        merely differently normalised.

        Parameters
        ----------
        z_maps : dict[str, np.ndarray]
            Maps a model/architecture name to its z-score map.

        Returns
        -------
        tuple[dict[str, np.ndarray], tuple[float, float]]
            The scaled maps under the same keys, and the shared
            ``(vmin, vmax)`` to plot them with — the argument pair
            ``scripts.visualization_nll.plot_signal_comparison`` expects.

        Raises
        ------
        ValueError
            If ``z_maps`` is empty.
        """
        if not z_maps:
            raise ValueError("z_maps must contain at least one map.")
        shared = self.limit(*z_maps.values())
        scaled = {name: self.apply(z, limit=shared) for name, z in z_maps.items()}
        first = next(iter(scaled.values()))
        return (
            {name: s.values for name, s in scaled.items()},
            (first.vmin, first.vmax),
        )

    def _label(self, bound: float) -> str:
        source = "" if self.mode is ZScaleMode.FIXED else f" (p{self.percentile:g})"
        gamma = "" if self.gamma == 1.0 else f", gamma={self.gamma:g}"
        return f"|z| <= {bound:.2f}{source}{gamma}"
