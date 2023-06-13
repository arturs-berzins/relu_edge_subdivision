import torch
from torch import nn
from torch.nn.functional import relu


class Net(nn.Module):
    def __init__(self, ks):
        super(Net, self).__init__()
        ## Feature-sizes
        self.ks = ks
        ## Fully-connected layers
        self.fcs = nn.ModuleList([nn.Linear(in_features, out_features)
            for in_features, out_features in zip(self.ks[:-1],self.ks[1:])])
        ## Depth
        self.D = len(self.fcs)
        ## Placeholder for intermediate values
        self.fs = [None]*(self.D+1) ## TODO: maybe dict instead?
        self.ls = list(range(1,self.D+1))
        self.Ks = [sum(self.ks[1:l]) for l in self.ls] + [sum(self.ks[1:])] ## similar to neuron_idx
    def forward(self, x):
        self.fs[1] = self.fcs[0](x)
        for i in range(2,self.D+1):
            self.fs[i] = self.fcs[i-1](relu(self.fs[i-1]))
        return self.fs[-1]
    def eval_block(self, x, device='cpu'):
        '''Like forward but preallocates and slices the tensor for storing values.'''
        Ks = self.Ks
        fs = torch.empty(len(x), Ks[-1], device=device)
        fs[:,Ks[0]:Ks[1]] = self.fcs[0](x)
        for i in range(1,self.D):
            fs[:,Ks[i]:Ks[i+1]] = self.fcs[i](relu(fs[:,Ks[i-1]:Ks[i]]))
        return fs

### Some predefined models
from collections import OrderedDict
class NetBunny(Net):
    def __init__(self, dim=3, depth=3, width=32):
        super().__init__(ks=[dim,*[width]*depth,1])
        try:
            self.load_state_dict(torch.load(f'models/{dim},{depth}x{width},relu,bunny', map_location='cpu'))
        except:
            print(f"The requested bunny model {dim=}, {depth=}, {width=} is not available under models/")


if __name__ == "__main__":
    f = NetBunny(dim=3, depth=3, width=16)
    f = NetBunny(dim=3, depth=8, width=32)
    f = NetBunny(dim=2, depth=4, width=32)
    f = NetBunny(dim=2, depth=2, width=50)
    f = NetBunny(dim=2, depth=2, width=100)

    # xs = torch.rand([10000,3]) - .5
    # vs = f(xs)[:,0]
    
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # mask = vs<0 ## interior points
    # # mask = torch.abs(vs)<1e-3 ## boundary points
    # ax.scatter(*xs[mask].T, c='k')
    # ax.set_box_aspect([1,1,1])
    # # ax.set_xlabel('x')
    # # ax.set_ylabel('y')
    # # ax.set_zlabel('z')
    # plt.show()

    # # torch.manual_seed(11)
    # # f = Net(ks=[2,4,4,1])