"""Some utils for building (k+1)-cells from the k-cells, as well as building triangle meshes of these for plotting."""

import numpy as np
import torch
from itertools import compress

def build_parents(e_sv, d):
    '''
    NOTE: IMPORTANT
    Build all k+1 cells of k skeleton.
    Not part of marching.

    A k-cell has d-k zeros (k=0 vertex, 1 edge, 2 face ...).
    To get all parent cells, perturb each (interior) 0 toward +/-.
    This gives 2*(d-k) parents.
    Now we need to invert this map: for each (k+1)-cell, find all k-cells it points to.
    '''
    k = d-(e_sv[0]==0).sum()
    # assert torch.all((e_sv==0).sum(1)==d-k), "Invalid sign-vectors"

    faces_of_edges = e_sv.repeat(2*(d-k),1,1)
    zero_idxs = ((e_sv==0).nonzero()[:,1]).reshape(-1,d-k) ## These are the indices of the d-1 constraints per edge
    rnge = torch.arange(len(e_sv))
    for i in range(d-k):
        faces_of_edges[2*i+0,rnge,zero_idxs[:,i]] =  1
        faces_of_edges[2*i+1,rnge,zero_idxs[:,i]] = -1
    

    ### Find unique faces and their edges ###
    b = faces_of_edges.flatten(end_dim=1)
    ## On cuda unique is faster: find unique faces and the indices telling where each face (perturbed edge) from b is in unique faces. 
    f_sv, inv = b.unique(dim=0, return_inverse=True)

    edges_of_faces = [[] for _ in f_sv]
    for i, inv_i in enumerate(inv): # inv_i is index of edge i (x4 perturbed) in unique faces
        edges_of_faces[inv_i].append(i%len(e_sv))
    
    ## Remove the faces outside the domain. These are faces with just a single edge
    nof_edges_of_face = torch.tensor(list(map(len, edges_of_faces)))
    f_sv = f_sv[nof_edges_of_face>1]
    edges_of_faces = list(compress(edges_of_faces, nof_edges_of_face>1))
    ## TODO: we can just perturb toward the interior. Sorting out the modulo should not be that hard?

    return f_sv, edges_of_faces


def combine_meshes(vertss, indicess):
    '''
    Combines several meshes in one.
    vertss: iterable of verts, each [N_i,3]
    indicess: iterable of indices, each [N_i,3]
    '''
    verts_combined = torch.empty([0,3], dtype=torch.float32)
    indices_combined = []
    for verts, indices in zip(vertss, indicess):
        # idx_updated = np.add(indices, len(verts_combined)-1).tolist()
        indices_combined.extend( np.add(indices, len(verts_combined)).tolist() )
        verts_combined = torch.vstack([verts_combined, verts])
    return verts_combined, indices_combined


"""This class just calls the small library and stores things as internal variables."""

class Complex():
    """
    TODO: think through the interface.
    We have two use-cases: plotting where we need the meshes and optimization where we need the speed.
    TODO: building meshes requires to detach gradients.
    NOTE: we can differentiate vs.shape[1] and d, e.g. if we want to plot in extra dimension
    """
    def __init__(self, vs, edges, v_sv, e_sv, vtransform=None, d=3,
                 do_build_faces=True,
                 do_build_cells=True,
                 do_build_mesh_helpers=False):
        assert d<=3, "d>3 has not been implemented yet"
        if vtransform is not None: vs = vtransform(vs)
        self.vs = vs
        self.edges = edges
        self.v_sv = v_sv
        self.e_sv = e_sv
        self.d = d
        ## TODO: maybe have a B attribute?
        ## TODO: store in a dict like cells[k], cell_sv[k] not with explicit naming like this 
        if d>=2 and do_build_faces:
            self.f_sv, self.faces = build_parents(self.e_sv, self.d)
        if d>=3 and do_build_cells:
            self.c_sv, self.cells = build_parents(self.f_sv, self.d)
        if do_build_mesh_helpers: ## TODO: this interface is not well thought trough
            self.build_mesh_helpers()
   
    def build_mesh_helpers(self):
        self.verts_face_twice = [self.vs[self.edges[list(eis)]].reshape(-1, self.vs.shape[1]) for eis in self.faces]
        self.centroids = torch.vstack([verts.mean(0) for verts in self.verts_face_twice])
    
    def get_face_mesh(self, idx):
        '''idx is index of cell'''
        ## Build triangles wrt centroid
        verts_mesh = torch.vstack([self.verts_face_twice[idx], self.centroids[idx]])
        ic = len(verts_mesh)-1
        indices = [[2*i,2*i+1,ic] for i in range(len(self.verts_face_twice[idx])//2)]
        indices = indices + [list(reversed(inds)) for inds in indices]
        return verts_mesh, indices
    
    def get_all_face_mesh(self, i):
        '''
        Get a mesh of all the faces of a folded hyperplane given by the idx.
        E.g. i=-1 will give the mesh of the folded hyperplane corresponding to the last neuron.
        '''
        face_idxs = torch.where(self.f_sv[:,i]==0)[0]
        vertss, indicess = zip(*[self.get_face_mesh(face_idx) for face_idx in face_idxs])
        return combine_meshes(vertss, indicess)

    def get_cell_mesh(self, idx):
        '''i is index of cell'''
        face_idxs = self.cells[idx]
        vertss, indicess = zip(*[self.get_face_mesh(face_idx) for face_idx in face_idxs])
        return combine_meshes(vertss, indicess)