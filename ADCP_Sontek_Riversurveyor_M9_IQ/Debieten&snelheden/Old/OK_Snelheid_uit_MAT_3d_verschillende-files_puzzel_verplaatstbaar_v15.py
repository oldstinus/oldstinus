#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADCP Multi-Profile 3D GUI – v15-OK
Integratie van v7b-OK + Bathymetry functionaliteit (grid interpolatie, semi-transparant, export naar MAT/DAE/HTML)
"""

import os, math
import numpy as np
import scipy.io
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline
import xml.etree.ElementTree as ET

# ---- helpers voor mat ----
def _todict(matobj):
    if not hasattr(matobj, "_fieldnames"):
        return matobj
    d = {}
    for field in matobj._fieldnames:
        d[field] = _todict(getattr(matobj, field))
    return d

def _tolist(ndarray):
    return [_todict(x) if isinstance(x, scipy.io.matlab.mat_struct) else x for x in ndarray]

def _check_keys(d):
    for key in list(d.keys()):
        if isinstance(d[key], scipy.io.matlab.mat_struct):
            d[key] = _todict(d[key])
        elif isinstance(d[key], np.ndarray):
            d[key] = _tolist(d[key])
    return d

def loadmat(fname):
    data = scipy.io.loadmat(fname, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

# ---- extract ADCP arrays ----
def extract_3d_arrays(data_dict):
    summary = data_dict["Summary"]
    system  = data_dict["System"]
    btrack  = data_dict["BottomTrack"]
    wtrack  = data_dict["WaterTrack"]
    track   = np.array(summary["Track"])
    cstart  = np.array(system["Cell_Start"]).squeeze()
    csize   = np.array(system["Cell_Size"]).squeeze()
    bt_depth= np.array(btrack["BT_Depth"]).squeeze()
    vel     = np.array(wtrack["Velocity"])  # (NC,4,NS)
    u,v = vel[:,0,:], vel[:,1,:]
    speed = np.sqrt(u**2+v**2)

    NC,NS = speed.shape
    X,Y,Z,S = [],[],[],[]
    for i in range(NS):
        z_raw = cstart[i] + (np.arange(NC)+0.5)*csize[i]
        valid = z_raw < bt_depth[i]
        z_raw = z_raw[valid]; s_raw = speed[:,i][valid]
        if len(z_raw)==0: continue
        X.append([track[i,0]]*len(z_raw))
        Y.append([track[i,1]]*len(z_raw))
        Z.append(z_raw); S.append(s_raw)
    return np.concatenate(X), np.concatenate(Y), np.concatenate(Z), np.concatenate(S)

# ---- hoofdklasse GUI ----
class App:
    def __init__(self, master):
        self.root=master
        self.root.title("ADCP Multi-Profile 3D – v15-OK")
        self.files=[]
        self.profiles=[]
        self.norm=Normalize(0,1)
        self.bathy=None

        # figuur
        self.fig=Figure(figsize=(8,6),dpi=100)
        self.ax3d=self.fig.add_subplot(111,projection="3d")
        self.canvas=FigureCanvasTkAgg(self.fig,master=self.root)
        self.canvas.get_tk_widget().pack(fill="both",expand=True)
        NavigationToolbar2Tk(self.canvas,self.root).update()

        # controls
        frame=ttk.Frame(master); frame.pack(fill="x")
        ttk.Button(frame,text="Selecteer .mat",command=self.select_files).pack(side="left")
        ttk.Button(frame,text="Laad & plot",command=self.load_and_plot).pack(side="left")
        ttk.Button(frame,text="Genereer bathymetry",command=self.make_bathymetry).pack(side="left",padx=10)
        ttk.Button(frame,text="Export .mat",command=self.export_mat).pack(side="left")
        ttk.Button(frame,text="Export HTML",command=self.export_html).pack(side="left")

        self.bathy_res=tk.DoubleVar(value=1.0)
        self.bathy_method=tk.StringVar(value="linear")
        self.bathy_export=tk.BooleanVar(value=True)

        bathy=ttk.LabelFrame(master,text="Bathymetry opties")
        bathy.pack(fill="x")
        ttk.Label(bathy,text="Grid spacing (m):").pack(side="left")
        ttk.Entry(bathy,textvariable=self.bathy_res,width=6).pack(side="left")
        ttk.Label(bathy,text="Methode:").pack(side="left",padx=(10,0))
        ttk.Combobox(bathy,textvariable=self.bathy_method,values=["linear","cubic","nearest"],width=8).pack(side="left")
        ttk.Checkbutton(bathy,text="Bathymetry meenemen in export",variable=self.bathy_export).pack(side="left",padx=10)

    def select_files(self):
        f=filedialog.askopenfilenames(filetypes=[("MAT","*.mat")])
        if f: self.files=list(f)

    def load_and_plot(self):
        self.profiles=[]
        for f in self.files:
            d=loadmat(f)
            X,Y,Z,S=extract_3d_arrays(d)
            self.profiles.append(dict(file=f,X=X,Y=Y,Z=Z,S=S,xoffset=0,yoffset=0))
        if not self.profiles: return
        allS=np.concatenate([p["S"] for p in self.profiles])
        self.norm=Normalize(vmin=np.nanmin(allS),vmax=np.nanmax(allS))
        self.redraw()

    def redraw(self):
        self.ax3d.clear()
        for p in self.profiles:
            X=p["X"]+p["xoffset"]
            Y=p["Y"]+p["yoffset"]
            self.ax3d.scatter(X,Y,-p["Z"],c=p["S"],cmap="viridis",norm=self.norm,s=2)
        if self.bathy is not None:
            Xg,Yg,Zg=self.bathy
            self.ax3d.plot_surface(Xg,Yg,-Zg,cmap="terrain",alpha=0.5)
        self.ax3d.set_xlabel("X"); self.ax3d.set_ylabel("Y"); self.ax3d.set_zlabel("Diepte")
        self.canvas.draw_idle()

    def make_bathymetry(self):
        if not self.profiles: return
        pts=[]; vals=[]
        for p in self.profiles:
            X=p["X"]+p["xoffset"]; Y=p["Y"]+p["yoffset"]; Z=p["Z"]
            for x,y,z in zip(X,Y,Z):
                pts.append([x,y]); vals.append(z)
        pts=np.array(pts); vals=np.array(vals)
        dx=float(self.bathy_res.get())
        xi=np.arange(np.nanmin(pts[:,0]),np.nanmax(pts[:,0]),dx)
        yi=np.arange(np.nanmin(pts[:,1]),np.nanmax(pts[:,1]),dx)
        Xg,Yg=np.meshgrid(xi,yi)
        Zg=griddata(pts,vals,(Xg,Yg),method=self.bathy_method.get())
        self.bathy=(Xg,Yg,Zg)
        self.redraw()

    def export_mat(self):
        if not self.profiles: return
        out=filedialog.asksaveasfilename(defaultextension=".mat")
        if not out: return
        data={"profiles":self.profiles}
        if self.bathy is not None and self.bathy_export.get():
            Xg,Yg,Zg=self.bathy
            data["Bathymetry"]={"Xgrid":Xg,"Ygrid":Yg,"Zgrid":Zg,
                                "options":{"dx":float(self.bathy_res.get()),
                                           "method":self.bathy_method.get()}} 
        savemat(out,data)
        messagebox.showinfo("Export",f"MAT opgeslagen:\n{out}")

    def export_html(self):
        if not self.profiles: return
        out=filedialog.asksaveasfilename(defaultextension=".html")
        if not out: return
        traces=[]
        for p in self.profiles:
            X=p["X"]+p["xoffset"]; Y=p["Y"]+p["yoffset"]
            traces.append(go.Scatter3d(x=X,y=Y,z=-p["Z"],
                                       mode="markers",
                                       marker=dict(size=2,color=p["S"],
                                                   colorscale="Viridis",
                                                   cmin=self.norm.vmin,cmax=self.norm.vmax),
                                       name=os.path.basename(p["file"])))
        if self.bathy is not None and self.bathy_export.get():
            Xg,Yg,Zg=self.bathy
            traces.append(go.Surface(x=Xg,y=Yg,z=-Zg,colorscale="terrain",opacity=0.5,
                                     name="Bathymetry",showscale=False))
        fig=go.Figure(data=traces)
        plot_offline(fig,filename=out,auto_open=False)
        messagebox.showinfo("Export",f"HTML opgeslagen:\n{out}")

def main():
    root=tk.Tk()
    App(root)
    root.mainloop()

if __name__=="__main__":
    main()
