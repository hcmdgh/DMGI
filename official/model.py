from gh import * 


class DMGI(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 num_nodes: int, 
                 num_metapaths: int,
                 dropout: float = 0.5,
                 loss_alpha: float = 0.001):
        super().__init__()
        
        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.num_nodes = num_nodes
        self.loss_alpha = loss_alpha

        self.gcn_list = nn.ModuleList([
            dglnn.GraphConv(in_dim, out_dim) 
            for _ in range(num_metapaths)
        ])
        
        self.bfc = torch.nn.Bilinear(out_dim, out_dim, 1)

        self.Z = Parameter(torch.zeros(num_nodes, out_dim))
        
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Z)

    def forward(self, 
                metapath_g_list: list[dgl.DGLGraph],
                feat: FloatTensor) -> FloatScalarTensor:
        assert feat.shape == (self.num_nodes, self.in_dim)
        
        disc_pred_list = []
        disc_target_list = []
        pos_h_list = []
        neg_h_list = []
        
        assert len(metapath_g_list) == len(self.gcn_list)
        
        for g, gcn in zip(metapath_g_list, self.gcn_list):
            pos_h = self.dropout(feat)
            pos_h = gcn(graph=g, feat=pos_h)
            pos_h = torch.relu(pos_h)
            assert pos_h.shape == (self.num_nodes, self.out_dim)
            pos_h_list.append(pos_h)

            neg_h = self.dropout(feat)
            perm = np.random.permutation(self.num_nodes)
            neg_h = neg_h[perm]
            neg_h = gcn(graph=g, feat=neg_h)
            neg_h = torch.relu(neg_h)
            assert neg_h.shape == (self.num_nodes, self.out_dim)
            neg_h_list.append(neg_h)

            summary = pos_h.mean(0, keepdim=True)

            # [BEGIN] 判别器判别
            summary = summary.expand_as(pos_h)
            
            pred = self.bfc(pos_h, summary).view(-1)
            target = torch.ones_like(pred)
            assert pred.shape == target.shape == (self.num_nodes,)
            disc_pred_list.append(pred)
            disc_target_list.append(target)
            
            pred = self.bfc(neg_h, summary).view(-1)
            target = torch.zeros_like(pred)
            assert pred.shape == target.shape == (self.num_nodes,)
            disc_pred_list.append(pred)
            disc_target_list.append(target)
            # [END]
        
        disc_pred = torch.cat(disc_pred_list)
        disc_target = torch.cat(disc_target_list)
        loss = F.binary_cross_entropy_with_logits(input=disc_pred, target=disc_target)
        
        pos_h_mean = torch.stack(pos_h_list).mean(dim=0)
        neg_h_mean = torch.stack(neg_h_list).mean(dim=0)
        pos_reg_loss = (self.Z - pos_h_mean).pow(2).sum()
        neg_reg_loss = (self.Z - neg_h_mean).pow(2).sum()

        loss += self.loss_alpha * (pos_reg_loss - neg_reg_loss)

        return loss
