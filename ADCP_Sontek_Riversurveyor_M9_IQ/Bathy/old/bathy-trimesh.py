# -*- coding: utf-8 -*-
"""
Trimesh Bathy Viewer GUI (PySide6 + PyVistaQt) — met hoogte-overdrijving (Z×)
-------------------------------------------------------------------------------
- Leest .DAE (Collada) via trimesh (ook OBJ/PLY/STL)
- Interactief: rotaties (rx, ry, rz), translatie (tx, ty, tz), uniforme schaal, Z-overdrijving
- Visualisatie: vlak/smooth shading, wireframe, kleur op hoogte (Z/X/Y), uniforme kleur
- Mesh-processing: decimate (% reductie), smoothing (iteraties), normals herberekenen
- Camera-presets: top, iso, reset
- Export: OBJ/PLY/DAE
- Screenshot

Install:
    pip install trimesh pyvista pyvistaqt PySide6 numpy scipy

Run:
    python trimesh_bathy_viewer_gui.py
"""

import os
import numpy as np
import trimesh
from typing import Optional

from PySide6 import QtCore, QtWidgets, QtGui
from pyvistaqt import QtInteractor
import pyvista as pv


# --------------------------
# Converters & utilities
# --------------------------

def trimesh_to_single_mesh(tm):
    """Zorgt dat we één Trimesh terugkrijgen (merge geometries bij Scene)."""
    if isinstance(tm, trimesh.Scene):
        combined = None
        for name, geom in tm.geometry.items():
            g = geom.copy()
            if name in tm.graph.nodes_geometry:
                tf = tm.graph.get(frame_to=name)
                if tf is not None:
                    g.apply_transform(tf)
            combined = g if combined is None else trimesh.util.concatenate([combined, g])
        if combined is None:
            raise ValueError("Lege Scene (geen geometrieën).")
        return combined
    elif isinstance(tm, trimesh.Trimesh):
        return tm
    else:
        raise TypeError("Onbekend trimesh-type.")


def trimesh_to_pvpoly(tm: trimesh.Trimesh) -> pv.PolyData:
    """Zet Trimesh naar PyVista PolyData (triangles)."""
    vertices = np.asarray(tm.vertices, dtype=np.float64)
    faces = np.asarray(tm.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        tm = tm.triangulate()
        faces = np.asarray(tm.faces, dtype=np.int64)
        vertices = np.asarray(tm.vertices, dtype=np.float64)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    mesh = pv.PolyData(vertices, faces_pv)
    return mesh


def compute_elevation_scalars(mesh: pv.PolyData, axis='z'):
    """Voeg 'Elevation' scalars toe op X/Y/Z-coördinaat."""
    ax = axis.lower()
    if ax not in ('x', 'y', 'z'):
        ax = 'z'
    cols = {'x': 0, 'y': 1, 'z': 2}
    vals = mesh.points[:, cols[ax]].copy()
    mesh['Elevation'] = vals
    return mesh


# --------------------------
# Hoofd-venster
# --------------------------

class Viewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trimesh Bathy Viewer (PySide6 + PyVistaQt) — Z-overdrijving")
        self.resize(1200, 800)

        # Plot widget
        self.plotter = QtInteractor(self, auto_update=False)
        self.setCentralWidget(self.plotter)

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Klaar")

        # Data
        self.tm_original: Optional[trimesh.Trimesh] = None
        self.mesh_base: Optional[pv.PolyData] = None
        self.mesh_current: Optional[pv.PolyData] = None
        self.actor = None
        self.last_dir = os.getcwd()

        # UI
        self._build_toolbar()
        self._build_dock_controls()

        # Plotter settings
        pv.set_plot_theme("document")
        self.plotter.enable_anti_aliasing()
        self.plotter.add_axes()
        self.plotter.show_bounds(grid='front', location='outer', all_edges=True)
        self._reset_camera()

    # ---------- UI ----------

    def _build_toolbar(self):
        tb = self.addToolBar("Bestand")
        tb.setMovable(False)

        open_act = QtGui.QAction("Open .dae/.obj/.ply/.stl", self)
        open_act.triggered.connect(self.open_file)
        tb.addAction(open_act)

        export_menu = QtWidgets.QMenu("Export", self)
        act_exp_obj = export_menu.addAction("Export OBJ")
        act_exp_ply = export_menu.addAction("Export PLY")
        act_exp_dae = export_menu.addAction("Export DAE")
        act_exp_obj.triggered.connect(lambda: self.export_mesh("obj"))
        act_exp_ply.triggered.connect(lambda: self.export_mesh("ply"))
        act_exp_dae.triggered.connect(lambda: self.export_mesh("dae"))

        export_btn = QtWidgets.QToolButton()
        export_btn.setText("Export")
        export_btn.setMenu(export_menu)
        export_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        tb.addWidget(export_btn)

        shot_act = QtGui.QAction("Screenshot", self)
        shot_act.triggered.connect(self.save_screenshot)
        tb.addAction(shot_act)

        tb.addSeparator()

        reset_act = QtGui.QAction("Reset mesh", self)
        reset_act.triggered.connect(self.reset_mesh)
        tb.addAction(reset_act)

        cam_menu = QtWidgets.QMenu("Camera", self)
        act_top = cam_menu.addAction("Top (Z-down view)")
        act_iso = cam_menu.addAction("Isometrisch")
        act_reset = cam_menu.addAction("Reset camera")
        act_top.triggered.connect(self.camera_top)
        act_iso.triggered.connect(self.camera_iso)
        act_reset.triggered.connect(self._reset_camera)

        cam_btn = QtWidgets.QToolButton()
        cam_btn.setText("Camera")
        cam_btn.setMenu(cam_menu)
        cam_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        tb.addWidget(cam_btn)

    def _build_dock_controls(self):
        dock = QtWidgets.QDockWidget("Bediening", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        w = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(w)

        # Transforms
        self.spin_rx = self._spin(-180, 180, 0.0, 1.0, "°")
        self.spin_ry = self._spin(-180, 180, 0.0, 1.0, "°")
        self.spin_rz = self._spin(-180, 180, 0.0, 1.0, "°")
        self.spin_tx = self._dspin(-1e6, 1e6, 0.0, 0.1, "m")
        self.spin_ty = self._dspin(-1e6, 1e6, 0.0, 0.1, "m")
        self.spin_tz = self._dspin(-1e6, 1e6, 0.0, 0.1, "m")
        self.spin_scale = self._dspin(1e-6, 1e6, 1.0, 0.01, "×")
        # NIEUW: Z-overdrijving
        self.spin_zexag = self._dspin(0.01, 100.0, 1.0, 0.01, "×")

        layout.addRow(self._label("Rot X (°)"), self.spin_rx)
        layout.addRow(self._label("Rot Y (°)"), self.spin_ry)
        layout.addRow(self._label("Rot Z (°)"), self.spin_rz)
        layout.addRow(self._label("Trans X (m)"), self.spin_tx)
        layout.addRow(self._label("Trans Y (m)"), self.spin_ty)
        layout.addRow(self._label("Trans Z (m)"), self.spin_tz)
        layout.addRow(self._label("Schaal (uniform)"), self.spin_scale)
        layout.addRow(self._label("Hoogte-overdrijving (Z×)"), self.spin_zexag)

        btn_apply = QtWidgets.QPushButton("Toepassen ▶")
        btn_apply.clicked.connect(self.apply_transform)
        layout.addRow(btn_apply)

        layout.addRow(self._separator())

        # Visuals
        self.chk_wire = QtWidgets.QCheckBox("Wireframe")
        self.chk_smooth = QtWidgets.QCheckBox("Smooth shading")
        self.chk_normals = QtWidgets.QCheckBox("Recompute normals bij update")
        self.chk_normals.setChecked(True)

        self.cmb_color = QtWidgets.QComboBox()
        self.cmb_color.addItems([
            "Materiaal/Default",
            "Kleur op hoogte (Z)",
            "Kleur op X",
            "Kleur op Y",
            "Eigen kleur (uniform)"
        ])
        self.btn_color = QtWidgets.QPushButton("Kies kleur…")
        self.btn_color.clicked.connect(self.pick_color)
        self.uniform_color = (0.8, 0.8, 0.8)  # RGB

        layout.addRow(self.chk_wire)
        layout.addRow(self.chk_smooth)
        layout.addRow(self.chk_normals)
        layout.addRow(self._label("Kleurmodus"), self.cmb_color)
        layout.addRow(self.btn_color)

        btn_update_style = QtWidgets.QPushButton("Stijl bijwerken")
        btn_update_style.clicked.connect(self.update_style)
        layout.addRow(btn_update_style)

        layout.addRow(self._separator())

        # Processing
        self.spin_dec_pct = self._dspin(0.0, 99.9, 0.0, 1.0, "%")
        self.spin_smooth_iter = self._spin(0, 500, 0, 5, "iter")
        self.spin_smooth_relax = self._dspin(0.0, 1.0, 0.01, 0.01, "relax")

        layout.addRow(self._label("Decimate (% reductie)"), self.spin_dec_pct)
        layout.addRow(self._label("Smooth (iteraties)"), self.spin_smooth_iter)
        layout.addRow(self._label("Smooth (relaxatie)"), self.spin_smooth_relax)

        btn_apply_proc = QtWidgets.QPushButton("Toepassen op mesh")
        btn_apply_proc.clicked.connect(self.apply_processing)
        layout.addRow(btn_apply_proc)

        layout.addRow(self._separator())

        # Reset/Info
        btn_reset = QtWidgets.QPushButton("Volledig resetten (naar load)")
        btn_reset.clicked.connect(self.reset_mesh)
        layout.addRow(btn_reset)

        self.lbl_info = QtWidgets.QLabel("Geen mesh geladen.")
        self.lbl_info.setWordWrap(True)
        layout.addRow(self.lbl_info)

        w.setLayout(layout)
        dock.setWidget(w)

        # Live updates voor stijl
        self.chk_wire.stateChanged.connect(self.update_style)
        self.chk_smooth.stateChanged.connect(self.update_style)
        self.cmb_color.currentIndexChanged.connect(self.update_style)

    def _label(self, text):
        return QtWidgets.QLabel(text)

    def _spin(self, mn, mx, val, step, suffix=None):
        s = QtWidgets.QSpinBox()
        s.setRange(int(mn), int(mx))
        s.setValue(int(val))
        s.setSingleStep(int(step))
        if suffix:
            s.setSuffix(" " + suffix)
        return s

    def _dspin(self, mn, mx, val, step, suffix=None, decimals=4):
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(float(mn), float(mx))
        s.setValue(float(val))
        s.setSingleStep(float(step))
        if suffix:
            s.setSuffix(" " + suffix)
        return s

    def _separator(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    # ---------- File I/O ----------

    def open_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open 3D mesh", self.last_dir,
            "3D bestanden (*.dae *.obj *.ply *.stl);;Alle bestanden (*.*)"
        )
        if not fn:
            return
        self.last_dir = os.path.dirname(fn)
        self.load_mesh(fn)

    def load_mesh(self, path: str):
        self.status.showMessage(f"Laden: {path}")
        try:
            tm = trimesh.load(path, force='mesh', skip_materials=False)
            tm = trimesh_to_single_mesh(tm)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Fout", f"Kon mesh niet laden:\n{e}")
            self.status.showMessage("Laden mislukt")
            return

        self.tm_original = tm
        pvmesh = trimesh_to_pvpoly(tm)

        self.mesh_base = pvmesh.copy(deep=True)
        self.mesh_current = pvmesh.copy(deep=True)

        self._refresh_plot(recenter=True)
        self._update_info()
        self.status.showMessage("Mesh geladen")

    def export_mesh(self, fmt: str):
        if self.mesh_current is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Export als .{fmt}", self.last_dir, f"*.{fmt}"
        )
        if not fn:
            return
        if not fn.lower().endswith(f".{fmt}"):
            fn += f".{fmt}"
        try:
            tm = self._pv_to_trimesh(self.mesh_current)
            tm.export(fn)
            self.status.showMessage(f"Export OK: {fn}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export-fout", str(e))

    def save_screenshot(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Screenshot opslaan", self.last_dir, "PNG (*.png)"
        )
        if not fn:
            return
        if not fn.lower().endswith(".png"):
            fn += ".png"
        try:
            self.plotter.screenshot(fn)
            self.status.showMessage(f"Screenshot opgeslagen: {fn}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Screenshot-fout", str(e))

    # ---------- Plot & camera ----------

    def _refresh_plot(self, recenter=False):
        self.plotter.clear()
        if self.mesh_current is None:
            self.plotter.render()
            return

        mesh = self.mesh_current
        mode = self.cmb_color.currentText()
        clim = None
        scalars = None
        color = None

        if mode.startswith("Kleur op hoogte (Z)"):
            mesh = compute_elevation_scalars(mesh.copy(deep=True), 'z')
            scalars = 'Elevation'
        elif mode.endswith("(X)"):
            mesh = compute_elevation_scalars(mesh.copy(deep=True), 'x')
            scalars = 'Elevation'
        elif mode.endswith("(Y)"):
            mesh = compute_elevation_scalars(mesh.copy(deep=True), 'y')
            scalars = 'Elevation'
        elif mode.startswith("Eigen kleur"):
            color = self.uniform_color
        else:
            color = (0.8, 0.8, 0.8)

        style = 'wireframe' if self.chk_wire.isChecked() else 'surface'
        smooth_shading = self.chk_smooth.isChecked()

        self.actor = self.plotter.add_mesh(
            mesh, style=style, scalars=scalars, color=color,
            cmap="viridis", clim=clim, show_edges=False,
            smooth_shading=smooth_shading, name="mesh"
        )
        self.plotter.add_axes()
        self.plotter.show_bounds(grid='front', location='outer', all_edges=True)
        if recenter:
            self.plotter.reset_camera()
        self.plotter.render()

    def _reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.camera_position = 'iso'
        self.plotter.render()

    def camera_top(self):
        self.plotter.view_z()
        self.plotter.camera.ParallelProjectionOn()
        self.plotter.reset_camera()
        self.plotter.render()

    def camera_iso(self):
        self.plotter.camera_position = 'iso'
        self.plotter.camera.ParallelProjectionOff()
        self.plotter.reset_camera()
        self.plotter.render()

    # ---------- Mesh operations ----------

    def reset_mesh(self):
        if self.mesh_base is None:
            return
        self.mesh_current = self.mesh_base.copy(deep=True)
        # Reset UI
        for s in (self.spin_rx, self.spin_ry, self.spin_rz):
            s.setValue(0)
        for s in (self.spin_tx, self.spin_ty, self.spin_tz):
            s.setValue(0.0)
        self.spin_scale.setValue(1.0)
        self.spin_zexag.setValue(1.0)  # Z-overdrijving reset
        self.spin_dec_pct.setValue(0.0)
        self.spin_smooth_iter.setValue(0)
        self.spin_smooth_relax.setValue(0.01)
        self.chk_wire.setChecked(False)
        self.chk_smooth.setChecked(False)
        self.cmb_color.setCurrentIndex(0)
        self._refresh_plot(recenter=True)
        self._update_info()

    def apply_transform(self):
        if self.mesh_current is None:
            return
        rx = np.deg2rad(self.spin_rx.value())
        ry = np.deg2rad(self.spin_ry.value())
        rz = np.deg2rad(self.spin_rz.value())
        tx = self.spin_tx.value()
        ty = self.spin_ty.value()
        tz = self.spin_tz.value()
        s = self.spin_scale.value()
        zex = self.spin_zexag.value()  # NIEUW: Z-overdrijving

        mesh = self.mesh_current.copy(deep=True)

        # Anisotrope schaal: X/Y met s, Z met s*zex
        if (s != 1.0) or (zex != 1.0):
            pts = mesh.points.copy()
            pts[:, 0] *= float(s)
            pts[:, 1] *= float(s)
            pts[:, 2] *= float(s) * float(zex)
            mesh.points = pts

        # Rotaties (Z-Y-X)
        def rot_x(a):
            c, s_ = np.cos(a), np.sin(a)
            return np.array([[1,0,0],[0,c,-s_],[0,s_,c]])
        def rot_y(a):
            c, s_ = np.cos(a), np.sin(a)
            return np.array([[c,0,s_],[0,1,0],[-s_,0,c]])
        def rot_z(a):
            c, s_ = np.cos(a), np.sin(a)
            return np.array([[c,-s_,0],[s_,c,0],[0,0,1]])

        R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
        mesh.points = mesh.points @ R.T

        # Translatie
        mesh.points += np.array([tx, ty, tz], dtype=float)

        # Normals (optioneel)
        if self.chk_normals.isChecked():
            try:
                mesh = mesh.compute_normals(auto_orient_normals=True, inplace=False)
            except Exception:
                pass

        self.mesh_current = mesh
        self._refresh_plot(recenter=False)
        self._update_info()

    def apply_processing(self):
        if self.mesh_current is None:
            return
        mesh = self.mesh_current.copy(deep=True)

        # Decimate
        red_pct = float(self.spin_dec_pct.value())
        if red_pct > 0.0:
            target_reduction = max(0.0, min(0.999, red_pct / 100.0))
            try:
                mesh = mesh.decimate_pro(target_reduction=target_reduction)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Decimate", f"Decimate faalde: {e}")

        # Smooth
        iters = int(self.spin_smooth_iter.value())
        relax = float(self.spin_smooth_relax.value())
        if iters > 0:
            try:
                mesh = mesh.smooth(n_iter=iters, relaxation_factor=relax, inplace=False)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Smooth", f"Smooth faalde: {e}")

        # Normals
        if self.chk_normals.isChecked():
            try:
                mesh = mesh.compute_normals(auto_orient_normals=True, inplace=False)
            except Exception:
                pass

        self.mesh_current = mesh
        self._refresh_plot(recenter=False)
        self._update_info()

    def update_style(self):
        self._refresh_plot(recenter=False)

    def pick_color(self):
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(204, 204, 204), self, "Kies uniforme kleur")
        if col.isValid():
            self.uniform_color = (col.red() / 255.0, col.green() / 255.0, col.blue() / 255.0)
            if self.cmb_color.currentText().startswith("Eigen"):
                self._refresh_plot(recenter=False)

    # ---------- Helpers ----------

    def _pv_to_trimesh(self, mesh: pv.PolyData) -> trimesh.Trimesh:
        pts = np.asarray(mesh.points, dtype=np.float64)
        faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        tm = trimesh.Trimesh(vertices=pts, faces=faces, process=False)
        return tm

    def _update_info(self):
        if self.mesh_current is None:
            self.lbl_info.setText("Geen mesh geladen.")
            return
        n_pts = self.mesh_current.n_points
        n_cells = self.mesh_current.n_cells
        b = np.array(self.mesh_current.bounds).ravel()
        bbox = f"X[{b[0]:.3f}, {b[1]:.3f}]  Y[{b[2]:.3f}, {b[3]:.3f}]  Z[{b[4]:.3f}, {b[5]:.3f}]"
        self.lbl_info.setText(f"Punten: {n_pts:,} — Driehoeken: {n_cells:,}\n{bbox}")


# --------------------------
# main
# --------------------------

def main():
    app = QtWidgets.QApplication([])
    win = Viewer()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
