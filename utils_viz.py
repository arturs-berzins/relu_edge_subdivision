import torch

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_verts_and_edges(vs, edges, e_labels=None, v_labels=None, show=True, verts=True, edge_colors='k', bbox=None, ax=None):
    d = len(vs.T)
    vs = vs.cpu().detach()
    if edges is not None:
        edges = edges.cpu().detach()

    if ax is None:
        fig = plt.figure()
        fig.tight_layout()
    if d==1:
        if ax is None: ax = fig.add_subplot(111)
        if verts:
            ax.scatter(vs, [0]*len(vs), c='k', marker='.')
        if edges is not None:
            lc = LineCollection(torch.concat([vs[edges], torch.zeros(len(edges),2,1)], 2), colors=edge_colors)
            ax.add_collection(lc)
        ax.axis('equal')
    elif d==2:
        if ax is None: ax = fig.add_subplot(111)
        if verts:
            ax.scatter(*vs.T, c='k', marker='.')
        if edges is not None:
            lc = LineCollection(vs[edges], colors=edge_colors)
            ax.add_collection(lc)
        ax.axis('equal')
        if bbox is not None:
            ax.set_xlim(*bbox)
            ax.set_ylim(*bbox)
    elif d==3:
        if ax is None: ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
        if verts:
            ax.scatter(*vs.T, c='k', marker='.')
        if edges is not None:
            lc = Line3DCollection(vs[edges], colors=edge_colors)
            ax.add_collection3d(lc)
        ax.set_box_aspect([1,1,1])
        if bbox is not None: ## TODO: should be an option to set auto, where we compute from verts, maybe plot them invisbly
            ax.set_xlim(*bbox)
            ax.set_ylim(*bbox)
            ax.set_zlim(*bbox)
    else: return None
    ## Edge labels
    if e_labels is not None and edges is not None:
        for edge, label in zip(edges, e_labels):
            pos = vs[edge].mean(0)
            ax.text(*pos, label, ha='center', va='center', color='blue', fontsize=8, zorder=10)
    ## Vertex labels
    if v_labels is not None:
        for pos, label in zip(vs, v_labels):
            ax.text(*pos, label, ha='center', va='center', color='green', fontsize=8, zorder=10)
    if not show: return ax
    plt.show()