# -*- coding: utf-8 -*-
"""
3D → 2D puntenwolk selectie (lasso) voor raster (Xi, Yi, Zi) zonder her-interpolatie.
- 3D venster: roteer/zoom, druk L -> opent 2D scatter (alleen punten).
- In 2D venster: teken lasso, druk Enter of D om te bevestigen, Q/Esc om te annuleren.
- Geselecteerde grid-nodes worden op NaN gezet in Zi en direct hertekend.
- U in 3D venster = undo (stap terug).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from mpl_toolkits.mplot3d import proj3d
from scipy.ndimage import binary_dilation


class Lasso2DSelector:
    """
    Zuivere 2D-scatter + lasso (geen lijnen).
    show() retourneert een boolean-mask over de aangeleverde punten (pts2d).
    Keys: Enter/D = bevestigen, U = reset selectie, Q/Esc = annuleren.
    """
    def __init__(self, pts2d, title="2D selectie (Lasso)"):
        self.pts2d = np.asarray(pts2d, float)
        self.sel_mask = np.zeros(self.pts2d.shape[0], dtype=bool)  # wat de lasso selecteerde

        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)  # voorkom clash met default
        self.ax.set_title(title + " — Lasso tekenen; Enter/D=bevestig, U=reset, Q=annuleer")
        self.ax.set_xlabel("screen x (px)"); self.ax.set_ylabel("screen y (px)")
        self.ax.invert_yaxis()  # schermcoördinaat: y naar beneden
        self.ax.grid(True, alpha=0.2)

        # scatter: enkel puntjes, geen lijnen
        self.scat = self.ax.scatter(self.pts2d[:, 0], self.pts2d[:, 1], s=6, alpha=0.9)
        self._recolor()

        # key events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # lasso (zichtbare gele lijn)
        try:
            self.lasso = LassoSelector(self.ax, onselect=self._on_lasso, useblit=False,
                                       lineprops=dict(color='yellow', linewidth=1.8, alpha=0.95))
        except TypeError:
            self.lasso = LassoSelector(self.ax, onselect=self._on_lasso, useblit=False,
                                       line_props=dict(color='yellow', linewidth=1.8, alpha=0.95))

    def _on_lasso(self, verts):
        path = Path(verts)
        self.sel_mask = path.contains_points(self.pts2d)
        self._recolor()

    def _recolor(self):
        # blauw = niet geselecteerd, rood = geselecteerd
        c = np.full((self.pts2d.shape[0], 4), [0.2, 0.4, 1.0, 0.9])
        if np.any(self.sel_mask):
            c[self.sel_mask] = [1.0, 0.2, 0.2, 0.95]
        self.scat.set_color(c)
        self.fig.canvas.draw_idle()

    def _on_key(self, ev):
        k = (ev.key or '').lower()
        if k in ('enter', 'return', 'd'):
            plt.close(self.fig)
        elif k == 'u':
            self.sel_mask[:] = False
            self._recolor()
        elif k in ('q', 'escape'):
            self.sel_mask[:] = False
            plt.close(self.fig)

    def show(self):
        plt.show(block=True)
        return self.sel_mask


class RasterEditor3D:
    """
    3D-rasterbewerker voor (Xi, Yi, Zi).
    Werkwijze:
      - Roteer/zoom 3D.
      - Druk L -> opent 2D-scatter (geprojecteerde grid-nodes, zonder lijnen).
      - Lasso -> Enter/D -> geselecteerde nodes worden op NaN gezet in Zi.
      - U = undo; Q = sluit.
    """
    def __init__(self, Xi, Yi, Zi, zexag=1.0, dilate_cells=0, visible_only=True):
        self.Xi = np.asarray(Xi, float)
        self.Yi = np.asarray(Yi, float)
        self.Zi = np.array(Zi, float)  # eigen kop
        self.zexag = float(zexag)
        self.dilate_cells = int(max(0, dilate_cells))
        self.visible_only = bool(visible_only)

        self._undo_stack = []

        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._draw()

    # ---------- rendering ----------
    def _draw(self, subtitle=None):
        self.ax.clear()
        depth = -self.Zi
        vmin = np.nanmin(depth)
        vmax = np.nanmax(depth)
        norm = plt.Normalize(vmin, vmax)
        cmap = 'turbo' if 'turbo' in plt.colormaps() else 'rainbow'
        colors = plt.cm.get_cmap(cmap)(norm(depth))
        facecolors = colors.copy()
        facecolors[np.isnan(depth)] = [0, 0, 0, 0]

        self.ax.plot_surface(self.Xi, self.Yi, depth,
                             facecolors=facecolors, rstride=1, cstride=1,
                             linewidth=0, antialiased=True)
        cbar = self.fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=self.ax, shrink=0.5)
        cbar.set_label("Diepte (m)")

        self.ax.set_xlabel("Easting (m)")
        self.ax.set_ylabel("Northing (m)")
        self.ax.set_zlabel("Diepte (m)")
        title = "3D Rasterbewerker — L: 2D-lasso  U: undo  Q: quit"
        if subtitle:
            title += f"  [{subtitle}]"
        self.ax.set_title(title)
        try:
            self.ax.set_box_aspect([1, 1, self.zexag])
        except Exception:
            pass
        self.fig.canvas.draw_idle()

    # ---------- helpers ----------
    def _project_nodes_to_pixels(self):
        """Projecteer grid-nodes naar 2D schermpixels (vorm (N,2))."""
        xs, ys, zs = proj3d.proj_transform(self.Xi.ravel(),
                                           self.Yi.ravel(),
                                           (-self.Zi).ravel(),
                                           self.ax.get_proj())
        pts2d = np.vstack([xs, ys]).T
        return self.ax.transData.transform(pts2d)  # pixels

    def _front_facing_mask_nodes(self):
        """Benader zichtbaarheidsmasker per node (front-facing)."""
        elev = np.deg2rad(self.ax.elev)
        azim = np.deg2rad(self.ax.azim)
        v = np.array([np.cos(elev)*np.sin(azim),
                      -np.cos(elev)*np.cos(azim),
                      np.sin(elev)], float)

        Zp = -self.Zi
        dXi_i = np.gradient(self.Xi, axis=0); dYi_i = np.gradient(self.Yi, axis=0); dZi_i = np.gradient(Zp, axis=0)
        dXi_j = np.gradient(self.Xi, axis=1); dYi_j = np.gradient(self.Yi, axis=1); dZi_j = np.gradient(Zp, axis=1)

        nx = dYi_i * dZi_j - dZi_i * dYi_j
        ny = dZi_i * dXi_j - dXi_i * dZi_j
        nz = dXi_i * dYi_j - dYi_i * dXi_j
        dot = nx*v[0] + ny*v[1] + nz*v[2]
        front = (dot < 0)

        node = np.zeros(self.Zi.shape, bool)
        node[:-1, :-1] |= front[:-1, :-1]
        node[1:,  :-1] |= front[:-1, :-1]
        node[:-1, 1: ] |= front[:-1, :-1]
        node[1:,  1: ] |= front[:-1, :-1]
        return node.ravel()

    # ---------- interactie ----------
    def _on_key(self, ev):
        k = (ev.key or '').lower()
        if k == 'l':
            self._run_2d_lasso()
        elif k == 'u':
            self._undo()
        elif k == 'q':
            plt.close(self.fig)

    def _run_2d_lasso(self):
        # toolbar-modus uit (anders slikt pan/zoom events)
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and hasattr(tb, "mode"):
            tb.mode = ''

        # snapshot projectie (na jouw oriëntatie)
        pix = self._project_nodes_to_pixels()

        # optioneel enkel zichtbare nodes (front-facing)
        if self.visible_only:
            vis = self._front_facing_mask_nodes()
        else:
            vis = np.ones(self.Xi.size, bool)

        idx_all = np.arange(self.Xi.size)
        idx_vis = idx_all[vis]
        pts_vis = pix[vis]

        # 2D lasso venster (enkel punten)
        sel = Lasso2DSelector(pts_vis, title="Geprojecteerde puntenwolk (screen-space)").show()

        if sel is None or not np.any(sel):
            self._draw()
            return

        # map terug naar grid
        sel_flat = np.zeros(self.Xi.size, bool)
        sel_flat[idx_vis[sel]] = True
        sel_grid = sel_flat.reshape(self.Xi.shape)

        # optionele dilatie in grid-topologie (neemt een randje mee)
        if self.dilate_cells > 0:
            sel_grid = binary_dilation(sel_grid, iterations=self.dilate_cells)

        # undo & toepassen: zet geselecteerde nodes op NaN
        self._undo_stack.append(self.Zi.copy())
        self.Zi[sel_grid] = np.nan

        self._draw(subtitle=f"verwijderd: {int(sel.sum())} nodes")

    def _undo(self):
        if not self._undo_stack:
            return
        self.Zi = self._undo_stack.pop()
        self._draw()

    def show(self):
        plt.show(block=True)
        return self.Zi


# ---- DEMO ----
if __name__ == "__main__":
    # Demo met synthetisch raster (parabool + wat NaN)
    x = np.linspace(0, 10, 120)
    y = np.linspace(0, 10, 100)
    Xi, Yi = np.meshgrid(x, y)
    Zi = 2.0 + 0.05*(Xi-5)**2 + 0.03*(Yi-5)**2
    Zi[20:25, 40:45] += 2.0  # "spike" eilandje

    editor = RasterEditor3D(Xi, Yi, Zi, zexag=2.0, dilate_cells=1, visible_only=True)
    Zi_new = editor.show()

    # Optioneel: toon resultaat
    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xi, Yi, -Zi_new, cmap='viridis')
    ax.set_title("Resultaat na selectie (NaN-gaten zichtbaar)")
    plt.show()
