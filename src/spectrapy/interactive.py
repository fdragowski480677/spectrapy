"""Interactive explorers built on matplotlib's own widgets (no extra dependencies).

Run them from a script or a notebook with an interactive backend (``%matplotlib widget`` or
``qt``)::

    import spectrapy as sp
    ex = sp.interactive.blackbody_explorer()
    ex.show()

Keep a reference to the returned :class:`Explorer` - matplotlib widgets stop responding when
they are garbage-collected. ``ex.set(T=3000)`` drives the explorer from code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from .colorimetry import color, xyz
from .colorspaces import rgb_to_hex
from .illuminants import blackbody
from .spectrum import wavelengths as default_wavelengths
from .temperature import cct
from .thinfilm import AIR, WATER, Stack, as_material, constant


@dataclass
class Explorer:
    """Handle to an interactive figure: the figure, its sliders and its update function."""

    fig: object
    sliders: dict = field(default_factory=dict)
    update: object = None

    def set(self, **values) -> Explorer:
        """Move sliders programmatically (each change triggers a redraw)."""
        for name, v in values.items():
            self.sliders[name].set_val(v)
        return self

    def show(self) -> None:
        plt.show()


def _swatch_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def blackbody_explorer(T: float = 5772.0, T_range=(1000.0, 20000.0)) -> Explorer:
    """Planck spectrum (normalised), its colour and CCT for a temperature chosen with a slider."""
    from .plot import spectrum

    wl = default_wavelengths()
    fig = plt.figure(figsize=(8.0, 4.6))
    ax = fig.add_axes([0.08, 0.30, 0.62, 0.62])
    sw = _swatch_axes(fig, [0.75, 0.42, 0.21, 0.50])
    label = fig.text(0.855, 0.33, "", ha="center", va="center", fontsize=9)
    s_ax = fig.add_axes([0.12, 0.10, 0.76, 0.05])
    slider = Slider(s_ax, "T (K)", np.log10(T_range[0]), np.log10(T_range[1]), valinit=np.log10(T))
    slider.valtext.set_text(f"{T:,.0f} K")

    def update(val):
        t = 10.0 ** val
        slider.valtext.set_text(f"{t:,.0f} K")
        s = blackbody(t, wavelengths=wl, normalize="max")
        ax.cla()
        spectrum(s, ax=ax, swatch=False, ylabel="Relative spectral radiance")
        ax.set_ylim(0, 1.08)
        c = color(s)
        sw.set_facecolor(tuple(c))
        T_c, duv = cct(xyz(s))
        label.set_text(f"{rgb_to_hex(c)}\nCCT {float(T_c):,.0f} K")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(slider.val)
    return Explorer(fig, {"T": _LogProxy(slider)}, update)


class _LogProxy:
    """Lets users set the log-scaled temperature slider in kelvin."""

    def __init__(self, slider):
        self.slider = slider

    def set_val(self, value):
        self.slider.set_val(np.log10(value))

    @property
    def val(self):
        return 10.0 ** self.slider.val


def thinfilm_explorer(material=WATER, thickness: float = 400.0, angle: float = 0.0, ambient=AIR, substrate=AIR,
                      illuminant="D65", max_thickness: float = 1500.0, index_slider: bool = True) -> Explorer:
    """Reflectance and colour of a single film; sliders for thickness, angle of incidence and (for a
    constant index) the refractive index. A strip shows the colour against thickness."""
    from .plot import color_strip, spectrum

    mat = as_material(material)
    wl = default_wavelengths(380, 780, 2)
    fig = plt.figure(figsize=(8.0, 5.6))
    ax = fig.add_axes([0.11, 0.47, 0.59, 0.46])
    sw = _swatch_axes(fig, [0.75, 0.56, 0.21, 0.37])
    label = fig.text(0.855, 0.50, "", ha="center", va="center", fontsize=9)
    strip = fig.add_axes([0.11, 0.31, 0.59, 0.07])
    axes_s = [fig.add_axes([0.14, 0.19 - 0.055 * i, 0.72, 0.035]) for i in range(3 if index_slider else 2)]
    sliders = {"thickness": Slider(axes_s[0], "d (nm)", 0.0, max_thickness, valinit=thickness),
               "angle": Slider(axes_s[1], "θ (°)", 0.0, 85.0, valinit=angle)}
    n0 = float(np.real(mat(np.array([550.0]))[0]))
    if index_slider:
        sliders["n"] = Slider(axes_s[2], "n", 1.0, 2.6, valinit=n0)

    def current_material():
        if index_slider and abs(sliders["n"].val - n0) > 1e-9:
            return constant(sliders["n"].val, f"n = {sliders['n'].val:.3f}")
        return mat

    def redraw_strip(m):
        d = np.linspace(0.0, max_thickness, 400)
        st = Stack([(m, 0.0)], ambient, substrate)
        R = st.scan(0, d, wavelengths=wl, angle=sliders["angle"].val)
        strip.cla()
        color_strip(d, color(R, illuminant=illuminant, brightness="global"), ax=strip)
        strip.set_xlabel("thickness (nm)", fontsize=8)

    def update(_=None):
        m = current_material()
        st = Stack([(m, sliders["thickness"].val)], ambient, substrate)
        R = st.reflectance(wl, angle=sliders["angle"].val)
        ax.cla()
        spectrum(R, ax=ax, swatch=False, ylabel="Reflectance", illuminant=illuminant)
        ax.set_title(f"{m.name}, {sliders['thickness'].val:.0f} nm, {sliders['angle'].val:.0f}°", fontsize=10)
        c = color(R, illuminant=illuminant, brightness="max")
        sw.set_facecolor(tuple(c))
        label.set_text(f"{rgb_to_hex(c)}\nR̄ = {float(np.mean(R.values)):.3f}")
        redraw_strip(m)
        strip.axvline(sliders["thickness"].val, color="white", lw=1.5)
        strip.axvline(sliders["thickness"].val, color="black", lw=0.6)
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)
    update()
    return Explorer(fig, sliders, update)


__all__ = ["Explorer", "blackbody_explorer", "thinfilm_explorer"]
