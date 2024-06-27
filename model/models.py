from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
from hyperbolic import expmap0, project,logmap0,expmap1,logmap1
from tqdm import tqdm
from euclidean import givens_rotations, givens_reflection
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import math
from geoopt import ManifoldParameter
from manifolds import Lorentz

EPS = 1e-5
temperature=0.2
max_norm=0.5
max_scale=2
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _,_ = self.forward(these_queries)
                    
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())] 
                        filter_out += [queries[b_begin + i, 2].item()]
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    # # Calculate the ranking of each query sample, that is, the number of scores greater than or equal to the target score, and add the results to ranks
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks

def lorentz_linear(x, weight, scale, bias=None):
    x = x @ weight.transpose(-2, -1)
    # time = x.narrow(-1, 0, 1).sigmoid() * scale + 1.1
    time = x.narrow(-1, 0, 1).sigmoid() * scale + 1.1
    if bias is not None:
        x = x + bias
    x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
    x_narrow = x_narrow / ((x_narrow * x_narrow).sum(dim=-1, keepdim=True) / (time * time - 1)).sqrt()
    x = torch.cat([time, x_narrow], dim=-1)
    return x


class MRME_KGC(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(MRME_KGC, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.init_size = 0.001
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()
        self.act = nn.Softmax(dim=1)
        self.num_heads = 2
        self.multihead_attn = nn.MultiheadAttention(embed_dim=rank, num_heads=2)
        self.temperature = 0.2
        self.manifold = Lorentz(max_norm=max_norm)
        self.scale = nn.Parameter(torch.ones(()) * max_scale, requires_grad=False)
        self.manifold = Lorentz(max_norm=max_norm)
        self.emb_entity = ManifoldParameter(self.manifold.random_normal((sizes[0], rank), std=1. / math.sqrt(rank)),
                                            manifold=self.manifold)

        self.relation_transform = nn.Parameter(torch.empty(sizes[0], rank, rank))
        nn.init.kaiming_uniform_(self.relation_transform)

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings1 = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings1[0].weight.data *= init_size
        self.embeddings1[1].weight.data *= init_size
        self.multi_c = 1;
        self.data_type = torch.float32

        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init1 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init2 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
            c_init1 = torch.ones((1, 1), dtype=self.data_type)
            c_init2 = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)
        self.c1 = nn.Parameter(c_init1, requires_grad=True)
        self.c2 = nn.Parameter(c_init2, requires_grad=True)

    def e_step(self, ent_embeddings, rel_embeddings):
        user_embeddings = ent_embeddings.detach().cpu().numpy()

        item_embeddings = rel_embeddings.detach().cpu().numpy()

        eStep_ent_embedding_centroids, eStep_ent_embedding_2cluster = self.run_kmeans(user_embeddings)
        eStep_rel_embedding_centroids, eStep_rel_embedding_2cluster = self.run_kmeans(item_embeddings)
        # eStep_ent_embedding_centroids, eStep_ent_embedding_2cluster = self.run_hierarchical_clustering(user_embeddings)
        # eStep_rel_embedding_centroids, eStep_rel_embedding_2cluster = self.run_hierarchical_clustering(item_embeddings)

        return eStep_ent_embedding_centroids, eStep_ent_embedding_2cluster, eStep_rel_embedding_centroids, eStep_rel_embedding_2cluster

    def run_hierarchical_clustering(self, x):
        """Run hierarchical clustering algorithm to get k clusters of the input tensor x
        """
        num_clusters = 1
        clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(x)
        # Compute "centroids" as the mean of all points in each cluster
        cluster_cents = np.array([x[clustering.labels_ == i].mean(axis=0) for i in range(num_clusters)])
        centroids = normalize(cluster_cents, axis=1)
        # Convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()

        node2cluster = torch.LongTensor(clustering.labels_).cuda()
        return centroids, node2cluster

    def run_kmeans(self, x):
        num_clusters = 3
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x)
        cluster_cents = kmeans.cluster_centers_
        centroids = normalize(cluster_cents, axis=1)
        centroids = torch.Tensor(centroids).cuda()

        node2cluster = torch.LongTensor(kmeans.labels_).cuda()
        return centroids, node2cluster

    def ProtoNCE_loss(self, ent_embeddings, rel_embeddings):
        ssl_temp = 0.3
        proto_reg = 8e-8
        norm_ent_embeddings = ent_embeddings
        ent_centroids, ent_2cluster, rel_centroids, rel_2cluster = self.e_step(ent_embeddings, rel_embeddings)  # [B,]
        ent_2centroids = ent_centroids[ent_2cluster]  # [B, e]
        pos_score_user = torch.mul(norm_ent_embeddings, ent_2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / ssl_temp)
        ttl_score_user = torch.matmul(norm_ent_embeddings, ent_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()
        # ------------------------    -------------------------------------------------
        norm_rel_embeddings = rel_embeddings
        rel_2centroids = rel_centroids[rel_2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_rel_embeddings, rel_2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / ssl_temp)
        ttl_score_item = torch.matmul(norm_rel_embeddings, rel_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        # proto_nce_loss = proto_reg * (proto_nce_loss_user)
        return proto_nce_loss

    def cal_cl_loss(self, x1, x2):
        norm1 = x1.norm(dim=-1)
        norm2 = x2.norm(dim=-1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / (torch.einsum('i,j->ij', norm1, norm2) + EPS)
        sim_matrix = torch.exp(sim_matrix / self.temperature)
        pos_sim = sim_matrix.diag()
        loss_1 = pos_sim / (sim_matrix.sum(dim=-2) - pos_sim + EPS)
        loss_2 = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim + EPS)

        loss_1 = -torch.log(loss_1).mean()
        loss_2 = -torch.log(loss_2).mean()
        loss = (loss_1 + loss_2) / 2.
        return loss

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lo_lhs = self.emb_entity[x]
        lo_rel = self.relation_transform[x]
        # print(lo_lhs.shape)
        rel1 = self.embeddings1[0](x[:, 1])
        rel2 = self.embeddings1[1](x[:, 1])
        entities = self.embeddings[0].weight
        entity1 = entities[:, :self.rank]
        entity2 = entities[:, self.rank:]
        lhs_t = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel1 = rel1[:, :self.rank]
        rel2 = rel2[:, :self.rank]
        lhs = lhs_t[0]
        c1 = F.softplus(self.c1[x[:, 1]])
        head1 = expmap0(lhs, c1)
        rel11 = expmap0(rel1, c1)
        lhs = head1
        res_c1 = logmap0(givens_reflection(rel2, lhs), c1)
        translation1 = lhs_t[1] * rel[1]
        c2 = F.softplus(self.c2[x[:, 1]])
        head2 = expmap1(lhs, c2)
        rel12 = expmap1(rel1, c2)
        lhss = head2
        res_c2 = logmap1(givens_rotations(rel2, lhss), c2)
        translation2 = lhs_t[1] * rel[0]
        c = F.softplus(self.c[x[:, 1]])
        head = lhs
        rot_q = givens_reflection(rel2, head).view((-1, 1, self.rank))

        # print("11",res_c1.shape)
        # lo_h = lorentz_linear(lo_lhs.unsqueeze(1), lo_rel, self.scale)
        # print(lo_h.shape)
        # print("lo_lhs.unsequeeze",lo_lhs.unsqueeze(1).shape)
        # print("lo_rel",lo_rel.shape)
        lo_h = lorentz_linear(lo_lhs.unsqueeze(1), lo_rel, self.scale).squeeze(1)
        lo_h = lo_h.mean(dim=(1, 2))
        # dropout = nn.Dropout(p=0.1)
        # lo_h = lo_h.sum(dim=(1,2))
        # lo_h = dropout(lo_h)
        # print(lo_h.shape)
        # lo_h = torch.squeeze(lo_h,dim=1)
        # print(lo_h.shape)

        cands = torch.cat(            [res_c1.view(-1, 1, self.rank), res_c2.view(-1, 1, self.rank), rot_q, lo_h.view(-1, 1, self.rank)], dim=1)
        context_vec = self.context_vec(x[:, 1]).view((-1, 1, self.rank))#(batch_size,1,rank)
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)#(batch_size,3,1)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)#(batch-size,rank)

        return (
                      (att_q * rel[0] - translation1) @ entity1.t() + (att_q * rel[1] + translation2) @ entity2.t()   
               ), [
                   (torch.sqrt(lhs_t[0] ** 2 + lhs_t[1] ** 2),
                    torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
               ] ,self.ProtoNCE_loss(res_c1,res_c2)



