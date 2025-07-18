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
    return [''.join(row) for row in alphabet[svs[:,B:]]]

def bits_to_int(bits, d, dim):
    return (bits*(2**torch.arange(d-1,-1,-1))).sum(dim)

def lin_interp(x1, x2, y1, y2):
    return y1 - x1*(y2-y1)/(x2-x1)

def get_e_sv(v_sv, edges):
    """Build edge sign-vectors from vertex sign-vectors."""
    return v_sv[edges].sum(1, dtype=torch.int8).sign()