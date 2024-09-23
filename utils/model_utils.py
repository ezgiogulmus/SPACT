import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPACT(nn.Module):
    def __init__(self, path_input_dim, omic_input_dim=None, embedding_dim=256, fusion=None, drop_out=0.25, heads=4, dim_head=16, n_classes=4, nb_cluster_groups=2, ff=2, activation="gelu", mlp_skip=True, mlp_depth=4, pooled_clusters=False, slide_aggregation="early", **kwargs):
        super(SPACT, self).__init__()
        
        self.slide_aggregation = slide_aggregation
        self.fusion = fusion if fusion is not None and nb_cluster_groups > 1 else "concat"
        self.nb_cluster_groups = nb_cluster_groups
        
        if omic_input_dim is not None:
            if not isinstance(omic_input_dim, list):
                omic_input_dim = [omic_input_dim]
            self.omic_encoders = nn.ModuleList([SNN(omic_input_dim=omic_input_dim[i], n_classes=embedding_dim, mlp_depth=mlp_depth, feature_extractor=True) for i in range(len(omic_input_dim))])
        else:
            self.omic_encoders = None

        self.gates = nn.ModuleList([GatedAttention(dim=path_input_dim) for _ in range(self.nb_cluster_groups)]) if not pooled_clusters else None
        self.clusters_ff = nn.ModuleList([FeedForward(dim=path_input_dim, out_dim=embedding_dim, activation=activation, mlp_skip=mlp_skip, drop_out=drop_out, hidden_dim=embedding_dim*4) for _ in range(nb_cluster_groups)])
        self.mha_fusion =  nn.ModuleList([MultiHeadAttention(dim=embedding_dim, heads=heads, dim_head=dim_head, drop_out=drop_out, cross_attention=True if self.omic_encoders is not None else False) for _ in range(nb_cluster_groups)]) 
        self.fused_clusters_ff = nn.ModuleList([FeedForward(dim=embedding_dim, activation=activation, mlp_skip=mlp_skip, drop_out=drop_out, hidden_dim=embedding_dim*4) for _ in range(nb_cluster_groups)]) if ff != 1 else None
        
        self.final_gate = GatedAttention(dim=embedding_dim)
        if slide_aggregation == "mid":
            self.slide_gate = GatedAttention(dim=embedding_dim)
        if self.fusion == "bilinear":
            self.final_fuser = BilinearFusion(skip=1, use_bilinear=1, dim1=embedding_dim, dim2=embedding_dim, scale_dim1=16, scale_dim2=16, gate1=1, gate2=1, mmhid=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

        initialize_weights(self)

    def forward(self, img_clusters, omics=None, return_weights=False):
        
        if return_weights:
            attn_weights = {}
            
        # Encode omics data
        if not isinstance(omics, list):
            omics = [omics]
        encoded_omics = torch.stack([self.omic_encoders[i](omics[i]) for i in range(len(omics))], dim=1) if self.omic_encoders is not None else None

        if not isinstance(img_clusters, list):
            img_clusters = [[img_clusters]]
        slide_outputs, slide_logits = [], []
        for slide_id, slide_clusters in enumerate(img_clusters):
            attended_features = []
            if return_weights:
                attn_weights[slide_id] = {
                "cluster_gates": [],
                "mha": []
            }
            for i in range(self.nb_cluster_groups):
                # Attention Gate
                x = slide_clusters[i]
                # print(x.shape, "1")
                if len(x.shape) == 2:
                    x = x.unsqueeze(0)
                # print(x.shape, "2")
                if self.gates is not None:
                    x = self.gates[i](x, return_weights=return_weights)
                    if return_weights:
                        attn_weights[slide_id]["cluster_gates"].append(x[1])
                        x = x[0]
                # print(x.shape, "3")
                # Feed Forward
                x = self.clusters_ff[i](x.transpose(0, 1))
                # print(x.shape, "4")
                # MHA Fusion
                fused_x = self.mha_fusion[i](x, encoded_omics, return_weights=return_weights)
                if return_weights:
                    attn_weights[slide_id]["mha"].append(fused_x[1])
                    fused_x = fused_x[0]
                # print(fused_x.shape, "5")
                attended_features.append(fused_x)
                
            # Merge cluster outputs
            attended_features = [i.squeeze(0) for i in attended_features]
            if self.fusion == "concat":
                attended_features = torch.cat(attended_features, dim=0)
            elif self.fusion == "bilinear":
                # print([i.shape for i in attended_features])
                attended_features = self.final_fuser(attended_features)
                # print(attended_features.shape)

            # Attention Gate
            output = self.final_gate(attended_features.unsqueeze(0), return_weights=return_weights) if self.final_gate is not None else attended_features.unsqueeze(0)
            
            if return_weights:
                attn_weights[slide_id]["final_gate"] = output[1]
                output = output[0]
            slide_outputs.append(output)
            if self.slide_aggregation == "late":
                output = self.classifier(output.squeeze(0))
            slide_logits.append(output)
        
        if self.slide_aggregation != "late":
            if len(slide_outputs) == 1:
                slide_outputs = slide_outputs[0]
            elif self.slide_aggregation == "mid":
                slide_outputs = self.slide_gate(torch.cat(slide_outputs, dim=1), return_weights=return_weights)
                if return_weights:
                    attn_weights["slide_gate"] = slide_outputs[1]
                    slide_outputs = slide_outputs[0]

            # Classifier Head
            slide_outputs = slide_outputs.squeeze(0)
            logits = self.classifier(slide_outputs)
        else:
            logits = torch.mean(torch.stack(slide_logits, dim=0), dim=0)
        
        hazards = torch.sigmoid(logits)
        if return_weights:
            return hazards, attn_weights
        return hazards
    

class SNN_Block(nn.Module):
    def __init__(self, dim, out_dim, drop_out=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.ELU(),
            nn.AlphaDropout(p=drop_out, inplace=False)
        )
    def forward(self, x):
        return self.net(x)


class SNN(nn.Module):
    def __init__(self, omic_input_dim, n_classes, mlp_depth, feature_extractor=False, **kwargs):
        super(SNN, self).__init__()
        self.feature_extractor = feature_extractor
        hidden = [omic_input_dim]
        for _ in range(mlp_depth-1):
            hidden.append(256)
        if feature_extractor:
            hidden.append(n_classes)
        self.fc_omic = nn.Sequential(*[
            SNN_Block(dim=hidden[i], out_dim=hidden[i+1], drop_out=0.25) for i in range(len(hidden)-1)
        ])
        if not feature_extractor:
            self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)

    def forward(self, x):
        feats = self.fc_omic(x)
        if self.feature_extractor:
            return feats
        logits = self.classifier(feats)
        hazards = torch.sigmoid(logits)
        return hazards
    
def init_max_weights(module):    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()
                
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, drop_out=0., cross_attention=True):
        super().__init__()
        self.cross_attention = cross_attention
        self.heads = heads
        self.dim_head = dim_head
        self.scale = 1 / math.sqrt(self.dim_head)
        inner_dim = dim_head * heads

        if cross_attention:
            self.omics_norm = nn.LayerNorm(dim)
        self.img_norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(drop_out)
        )
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop_out)
        
    def forward(self, img, omics=None, return_weights=False):
        # Normalize both vectors
        if len(img.shape) == 2:
            img = img.unsqueeze(0) # 1, P, 512
        img = self.img_norm(img)
        if self.cross_attention:
            if len(omics.shape) == 2:
                omics = omics.unsqueeze(0) # 1, O, 512
            omics = self.omics_norm(omics)
        else:
            omics = img

        # Q, K, V Embedding
        q = self.to_q(omics)
        k, v = self.to_kv(img).chunk(2, dim=-1)        
        q, k, v = map(lambda t: t.view(img.shape[0], -1, self.heads, self.dim_head).transpose(1, 2), [q, k, v]) # B, H, N, D
        
        # Scaled dot product
        attn_output, attn_weight = self._scaled_dot_product(q, k, v)
        attn_output = attn_output.transpose(1, 2).reshape(img.shape[0], -1, self.heads * self.dim_head)
        
        if return_weights:
            return self.to_out(attn_output), attn_weight
        return self.to_out(attn_output)

    def _scaled_dot_product(self, query, key, value):
        # K, V: 1, 4, N, D | Q: 1, 4, O, D
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weight = self.attend(attn_weight)
        attn_weight = self.dropout(attn_weight)
        # A: 1, 4, O, N (softmax at dim=-1)
        out = torch.matmul(attn_weight, value)
        # 1, 4, O, D
        return out, attn_weight
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=128, out_dim=None, drop_out = 0., mlp_skip=True, activation="gelu"):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU() if activation == "relu" else nn.LeakyReLU()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            act_fn,
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(drop_out)
        )
        self.residual = nn.Identity() if out_dim == dim else nn.Linear(dim, out_dim) if mlp_skip else None
        
    def forward(self, x):
        x_skip = x
        x = self.net(x)
        if self.residual is not None:
            return x + self.residual(x_skip)
        return x
    
    
class GatedAttention(nn.Module):
    def __init__(self, dim, drop_out=0.):
        super(GatedAttention, self).__init__()
        self.attention_fc = nn.Linear(dim, 1)
        self.gate_fc = nn.Linear(dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, return_weights=False):
        attention_scores = torch.tanh(self.attention_fc(x))
        gate_values = torch.sigmoid(self.gate_fc(x))
        gated_attention_scores = attention_scores * gate_values
        attention_weights = self.softmax(gated_attention_scores)
        attention_weights = self.dropout(attention_weights)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), x)
        if return_weights:
            return context_vector, attention_weights
        return context_vector
    
    
class BilinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling

    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, v):
        out = self._fuse(v[0], v[1])
        if len(v) > 2:
            for i in range(2, len(v)):
                out = self._fuse(out, v[i])
        return out
    
    def _fuse(self, vec1, vec2):
        device = vec1.device
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)
        ### Fusion
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=device)), 1)

        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out
                
class MIL(nn.Module):
    def __init__(self, drop_out = 0., n_classes = 2, path_input_dim=1024, **kwargs):
        super(MIL, self).__init__()
        assert n_classes > 2
        
        fc = [nn.Linear(path_input_dim, 512), nn.ReLU(), nn.Dropout(drop_out)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(512, n_classes)
        self.n_classes = n_classes
        initialize_weights(self)
    
    def forward(self, x):    
        h = self.fc(x)
        logits = self.classifiers(h)
        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]
        hazards = torch.sigmoid(top_instance)
        return hazards

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# if __name__ == "__main__":
    # import numpy as np
    # np.random.seed(0)
    # torch.random.manual_seed(0)
    # torch.use_deterministic_algorithms(True)

    # model_dict = {
    #     "path_input_dim": 512,
    #     "omic_input_dim": [100, 200, 300],
    #     "embedding_dim": 256,
    #     "fusion": "concat",
    #     "drop_out": 0.25,
    #     "n_classes": 4,
    #     "heads": 4,
    #     "dim_head": 16,
    #     "mlp_skip": True,
    #     "mlp_type": "small",
    #     "activation": "relu",
    #     "drop_out": .25
    # }
    # model = SPACT(**model_dict)
    # print(model)
    # print("Number of parameters: {:.2f}M" .format(sum(p.numel() for p in model.parameters())*1e-6))

    # omics = [torch.randn(1, n) for n in model_dict["omic_input_dim"]]
    # img_clusters = [torch.randn(480, 10, model_dict["path_input_dim"]), torch.randn(94, 50, model_dict["path_input_dim"]), torch.randn(46, 100, model_dict["path_input_dim"])]
    # output = model(img_clusters, omics)
    # print(output)