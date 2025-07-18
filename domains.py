import torch
import numpy as np
import itertools
from utils import bits_to_int

def get_hypercube(d: int, bbox: tuple):
    '''
    Builds a centered hypercube in d-dimensions.
    TODO: add center, scale etc
    d: dimension
    bbox: pair of lower, upper values
    '''

    ### Vertices
    vs_bits = torch.tensor(np.stack(np.meshgrid(*[[False,True]]*d)), dtype=bool).flatten(start_dim=1).T
    ## Sort, so the int representing the bit corresponds to its index. Important for edges
    vs_bits = vs_bits[bits_to_int(vs_bits, d, 1).sort().indices]

    ### Edges
    ## Repeat vertex bits d times
    dits = vs_bits.repeat(d,1,1)
    ## For each dimension, flip a single bit 
    for i in range(d): ## TODO: maybe some vector operation with eye or arange?
        dits[i,:,d-i-1] = ~dits[i,:,d-i-1]
    ## Build edges by converting bits to integers and joining with arange, which is the original edge.
    ee = torch.stack([torch.arange(2**d).repeat(d,1).T, bits_to_int(dits, d, 2).T]).flatten(start_dim=1).T
    ## Filter duplicate edges
    edges = ee[ee[:,0]<ee[:,1]]
    
    ## Vertex coordinates by transforming unit hypercube vertices
    vs = (vs_bits*(bbox[1]-bbox[0]) + bbox[0]).float() ## NOTE: here we can easily use different ranges for each dimension
    ## Vertex sign-vectors
    v_sv = torch.hstack([vs_bits, ~vs_bits]).to(dtype=torch.int8)

    return vs, edges, v_sv ## NOTE: ideally edges would be uint32, but torch does not support this

 
def get_simplex(center: torch.Tensor, scale: float):
    '''
    Builds a unit simplex in d-dimensions.
    Returns:
      vertices: Tensor of shape (num_vertices, d)
      vs_bits: Tensor of booleans (num_vertices, d) showing the 0/1 code of each vertex
      edges: Tensor of shape (num_edges, 2) with pairs of indices of vertices
    '''
    D = center.shape[0]

    # Step 1: Construct regular simplex in D-dimensional hyperplane in R^{D+1}
    e = torch.eye(D + 1)                     # (D+1, D+1)
    mean = torch.mean(e, dim=0, keepdim=True)
    u = e - mean                             # Centered simplex in R^{D+1}
    
    # Step 2 Version 2: Construct orthonormal basis for hyperplane sum=0
    normal = torch.ones(D + 1) / torch.sqrt(torch.tensor(D + 1, dtype=torch.float))
    U, S, Vh = torch.linalg.svd(torch.eye(D + 1) - normal[:, None] @ normal[None, :])
    basis = U[:, :-1]
    u = u @ basis

    # Step 3: Normalize pairwise distances, then scale
    pairwise_dist = torch.norm(u[0] - u[1])
    u = scale * u / pairwise_dist

    # Step 4: Translate to desired center
    vs = u + center                   # (D+1, D)

    # Step 5: Generate edges (unordered pairs of vertex indices)
    edges = torch.tensor(list(itertools.combinations(range(D + 1), 2)), dtype=torch.long)

    # Step 6: Compute sign vectors: 0 on diagonal, 1 elsewhere
    v_sv = torch.eye(D + 1, dtype=torch.int)

    return vs, edges, v_sv


def get_simplex_rightangle(center: torch.Tensor, side_length: float):
    '''
    Builds a right-angle simplex in d-dimensions.
    Returns:
      vertices: Tensor of shape (num_vertices, d)
      vs_bits: Tensor of booleans (num_vertices, d) showing the 0/1 code of each vertex
      edges: Tensor of shape (num_edges, 2) with pairs of indices of vertices
    '''
    
    dimension = len(center)
    vs = torch.eye(dimension)
    # add a row of zeros
    vs = torch.cat([vs, torch.zeros(1, dimension)], dim=0) # we are addiing the center vertex with all the right angles
    
    vs = vs * side_length
    vs += center 
    vs -= (1/(dimension+1)) # we are looking at the center of mass of the simplex
    
    edges = torch.tensor(list(itertools.combinations(range(dimension+1), 2)))
    assert "Edges should be of shape (dimension*(dimension+1)//2, 2)" , edges.shape == (dimension*(dimension+1)//2,2)
    
    # Number of hyperplanes - D+1 choose 3
    # hyperplanes = (dimension+1)*(dimension)*(dimension-1)//6
    
    # Each vertex has d zeros
    v_sv = torch.eye(dimension+1, dimension+1) # the number of hyperplanes is always dimension+1 
    # every vertex touches every hyperplane except one
    
    return vs, edges, v_sv