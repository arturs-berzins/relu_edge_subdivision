import torch
import numpy as np


alphabet = np.array(['0','+','-'])

def validate_tb(tb, p):
    return (tb[0].sum(1)!=p).nonzero()

def tb_to_012(tb):
    '''Convert from three booleans to a single integer TODO: LEGACY'''
    ## Range to adress the three possible states 0,+,- 
    rnge_3 = torch.tensor([0,1,2], dtype=torch.uint8, device=tb.device)[:,None,None] ## TODO: preallocate const
    return (tb*rnge_3).sum(0, dtype=torch.uint8) ## Can we do this with indexing instead of sum?

def tb_to_sv(tb):
    '''Convert from three booleans to a sign-vector TODO: LEGACY'''
    rnge = torch.tensor([0,1,-1], dtype=torch.int8, device=tb.device)
    rnge = rnge.view(3,*([1]*(len(tb.shape)-1)))
    return (tb*rnge).sum(0, dtype=torch.int8)

def sv_to_tb(sv):
    '''Convert from three booleans to a sign-vector TODO: LEGACY'''
    tb = torch.zeros(3,*sv.shape, dtype=bool, device=sv.device)
    tb[0,sv==0] = 1
    tb[1,sv==1] = 1
    tb[2,sv==-1] = 1
    return tb

# def tri_to_str(tribit, warn=True):
#     label = ''.join(alphabet[tribit.nonzero()[:,1]])
#     if warn and len(label)>len(tribit): label += '!'
#     return label

# def get_labels(tbs, B=0):
#     '''Generate list of labels. B skips first B sign-vector entries, mainly those of bbox which are always +.'''
#     return [tri_to_str(tbs[:,i,B:].T) for i in range(tbs.shape[1])]

def get_labels(svs, B=0):
    '''Generate a list of labels from sign-vectors. Skip the first B signs.'''
    return [''.join(row) for row in alphabet[svs]]

def bits_to_int(bits, d, dim):
    return (bits*(2**torch.arange(d-1,-1,-1))).sum(dim)

def get_unit_hypercube(d):
    '''
    Builds a unit hypercube in d-dimensions.
    vs_bits contains coordinates as booleans.
    edges contains pairs of integers referencing vertices.
    '''
    ### Hypercube ###

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

    return vs_bits, edges ## NOTE: ideally edges would be uint32, but torch does not support this