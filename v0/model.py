from gh import *


class RGCNConv(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 ntypes: set[NodeType], 
                 etypes: set[EdgeType],
                 activation: Callable = nn.PReLU()):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ntypes = ntypes 
        self.etypes = etypes 
        
        # 论文公式(2)：W_0
        self.self_fc_dict = nn.ModuleDict({
            ntype: nn.Linear(in_dim, out_dim)
            for ntype in ntypes
        })

        # 论文公式(2)：W_r
        self.relation_fc_dict = nn.ModuleDict({
            '__'.join(etype): nn.Linear(in_dim, out_dim)
            for etype in etypes
        })
        
        self.activation = activation 

    def forward(self,
                hg: dgl.DGLHeteroGraph,  
                feat_dict: dict[NodeType, FloatTensor]) -> dict[NodeType, FloatTensor]:
        with hg.local_scope():
            h_neigh_list_dict: dict[NodeType, list[FloatTensor]] = defaultdict(list)
            
            for etype in self.etypes:
                _etype = '__'.join(etype)
                src_ntype, _, dest_ntype = etype 
                subgraph = hg[etype]
                
                # 论文公式(2)：W_r @ h_j
                feat_src = feat_dict[src_ntype]
                h_src = self.relation_fc_dict[_etype](feat_src)

                subgraph.srcdata['h_src'] = h_src 
                
                # 论文公式(2)：Mean( W_r @ h_j ) 
                subgraph.update_all(
                    message_func = dglfn.copy_u('h_src', 'msg'),
                    reduce_func = dglfn.mean('msg', 'h_neigh'), 
                )
                
                h_neigh = subgraph.dstdata.pop('h_neigh')
                
                h_neigh_list_dict[dest_ntype].append(h_neigh)
                
            h_neigh_dict: dict[NodeType, FloatTensor] = dict()     
                
            # 论文公式(2)：Sum( Mean( W_r @ h_j ) ) 
            for ntype, h_neigh_list in h_neigh_list_dict.items():
                h_neigh = torch.stack(h_neigh_list).sum(0)
                h_neigh_dict[ntype] = h_neigh 
            
            h_self_dict: dict[NodeType, FloatTensor] = dict()     
                
            # 论文公式(2)：W_0 @ h_i 
            for ntype, feat in feat_dict.items():
                h_self_dict[ntype] = self.self_fc_dict[ntype](feat)
            
            out_dict: dict[NodeType, FloatTensor] = dict()   
                
            # 论文公式(2)：Act( Sum(...) + W_0 @ h_i )
            for ntype in self.ntypes:
                out_dict[ntype] = self.activation(h_neigh_dict[ntype] + h_self_dict[ntype]) 

            return out_dict


class RGCN(nn.Module):
    def __init__(self,
                 in_dim_dict: dict[NodeType, int],
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int,
                 ntypes: set[NodeType], 
                 etypes: set[EdgeType],
                 activation: Callable = nn.PReLU()):
        super().__init__()

        self.in_fc_dict = nn.ModuleDict({
            ntype: nn.Linear(in_dim_dict[ntype], hidden_dim) 
            for ntype in ntypes
        })
        
        self.conv_list = nn.ModuleList([
            RGCNConv(
                in_dim = hidden_dim,
                out_dim = hidden_dim,
                ntypes = ntypes, 
                etypes = etypes,
                activation = activation, 
            )
            for _ in range(num_layers)
        ])
        
        self.out_fc_dict = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, out_dim) 
            for ntype in ntypes
        })
        
    def forward(self,
                hg: dgl.DGLHeteroGraph,
                feat_dict: dict[NodeType, FloatTensor]) -> dict[NodeType, FloatTensor]:
        h_dict = {
            ntype: self.in_fc_dict[ntype](feat)
            for ntype, feat in feat_dict.items() 
        }
        
        for conv in self.conv_list:
            h_dict = conv(hg=hg, feat_dict=h_dict)

        out_dict = {
            ntype: self.out_fc_dict[ntype](h)
            for ntype, h in h_dict.items() 
        }
            
        return out_dict 
