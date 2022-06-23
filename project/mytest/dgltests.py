import dgl
import dgl.function as fn
import torch


g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
#g = dgl.to_bidirected(g)
g.ndata['x'] = torch.ones(5, 2)
print(g.ndata['x'])
print(type(g.ndata['x']))
# Specify edges using (Tensor, Tensor).

# print(g.edges())
# g.send_and_recv(g.edges(), fn.copy_u('x', 'm'), fn.sum('m', 'h'))

# print(g.ndata['h'])



# Specify edges using IDs.
# g.send_and_recv([0, 2, 3], fn.copy_u('x', 'm'), fn.sum('m', 'h'))
# g.ndata['h']

