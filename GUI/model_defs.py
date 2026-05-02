import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class SE2D(nn.Module):
    """Squeeze-and-Excitation 2D"""
    def __init__(self, c, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c//r, 1)
        self.fc2 = nn.Conv2d(c//r, c, 1)
    
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class DSRes2D(nn.Module):
    """Depthwise Separable Residual Block"""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SE2D(out_c)
        self.proj = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False) if (in_c!=out_c or stride!=1) else None
    
    def forward(self, x):
        idn = x
        x = F.relu(self.bn1(self.dw(x)))
        x = self.bn2(self.pw(x))
        x = self.se(F.relu(x))
        if self.proj is not None:
            idn = self.proj(idn)
        return F.relu(x + idn)


class MorphologyCNN(nn.Module):
    """Morphology CNN for image feature extraction"""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.b1 = DSRes2D(32, 64, stride=2)
        self.b2 = DSRes2D(64, 128, stride=2)
        self.b3 = DSRes2D(128, 128, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return self.head(x)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block"""
    def __init__(self, in_c, out_c, k=3, d=1):
        super().__init__()
        pad = (k-1)//2 * d
        self.c1 = nn.Conv1d(in_c, out_c, k, padding=pad, dilation=d)
        self.c2 = nn.Conv1d(out_c, out_c, k, padding=pad, dilation=d)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.proj = nn.Conv1d(in_c, out_c, 1) if in_c!=out_c else None
    
    def forward(self, x):
        idn = x
        x = F.relu(self.bn1(self.c1(x)))
        x = self.bn2(self.c2(x))
        if self.proj is not None:
            idn = self.proj(idn)
        return F.relu(x + idn)


class RR_TCN(nn.Module):
    def __init__(self, rr_dim, out_dim=64):
        super().__init__()
        self.rr_dim = int(rr_dim)
        if self.rr_dim <= 0:
            self.enabled = False
            self.out_dim = 0
            self.net = None
            return

        self.enabled = True
        self.out_dim = out_dim

        # 下面保持你原本RR_TCN的结构即可（Conv1d/TCN/MLP都行）
        self.net = nn.Sequential(
            nn.Linear(self.rr_dim, 128),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
            nn.ReLU(True),
        )

    def forward(self, rr):
        if not self.enabled:
            # rr 的 shape 是 (B,0) 或 (B,rr_dim)，我们返回 (B,0)
            B = rr.shape[0]
            return rr.new_zeros((B, 0))
        return self.net(rr)


print("✓ Basic modules loaded (SE2D, DSRes2D, MorphologyCNN, TCNBlock, RR_TCN)")

# ============================================================================
# RAG (Retrieval-Augmented Generation) Module
# ============================================================================

class RAGModule(nn.Module):
    """
    Retrieval-Augmented Generation for ECG Classification
    检索增强生成模块：通过检索相似样本来增强特征表示
    """
    def __init__(self, feature_dim=256, num_prototypes=200, num_classes=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        
        # 原型特征库
        self.register_buffer('prototypes', torch.randn(num_prototypes, feature_dim))
        self.register_buffer('prototype_labels', torch.zeros(num_prototypes, dtype=torch.long))
        self.register_buffer('prototype_initialized', torch.tensor(False))
        
        # 特征融合网络
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 注意力权重
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def initialize_prototypes(self, features, labels):
        """从训练数据初始化原型库"""
        if self.prototype_initialized:
            return
        
        prototypes_list = []
        labels_list = []
        samples_per_class = self.num_prototypes // self.num_classes
        
        for c in range(self.num_classes):
            class_mask = (labels == c)
            class_features = features[class_mask]
            
            if len(class_features) > 0:
                if len(class_features) >= samples_per_class:
                    indices = torch.randperm(len(class_features))[:samples_per_class]
                    selected = class_features[indices]
                else:
                    selected = class_features[torch.randint(0, len(class_features), (samples_per_class,))]
                
                prototypes_list.append(selected)
                labels_list.append(torch.full((samples_per_class,), c, dtype=torch.long))
        
        if prototypes_list:
            self.prototypes = torch.cat(prototypes_list, dim=0).to(self.prototypes.device)
            self.prototype_labels = torch.cat(labels_list, dim=0).to(self.prototype_labels.device)
            self.prototype_initialized = torch.tensor(True)
            print(f"✓ RAG prototypes initialized: {self.prototypes.shape}")
    
    def retrieve(self, query_features, k=5):
        """检索最相似的k个原型"""
        query_norm = F.normalize(query_features, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        similarities = torch.mm(query_norm, proto_norm.t())
        top_k_sim, top_k_idx = similarities.topk(k, dim=1)
        retrieved_features = self.prototypes[top_k_idx]
        retrieved_labels = self.prototype_labels[top_k_idx]
        return retrieved_features, retrieved_labels, top_k_sim
    
    def forward(self, query_features):
        """使用检索到的相似样本增强当前特征"""
        if not self.prototype_initialized:
            return query_features
        
        retrieved_features, retrieved_labels, similarities = self.retrieve(query_features, k=5)
        weights = F.softmax(similarities * 10, dim=1).unsqueeze(2)
        aggregated = (retrieved_features * weights).sum(dim=1)
        combined = torch.cat([query_features, aggregated], dim=1)
        alpha = self.attention(combined)
        fused = self.fusion(combined)
        augmented_features = query_features + alpha * fused
        return augmented_features


# =========== Model Architectures ===========

# 1. Pure CNN Model
class PureCNN(nn.Module):
    """Pure CNN architecture for ECG classification"""
    def __init__(self, num_classes, rr_dim=0, use_rag=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.rrnet = RR_TCN(rr_dim)
        fuse_in = 256 * 14 * 14 + self.rrnet.out_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fuse_in, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        
        self.use_rag = use_rag
        if use_rag:
            self.rag = RAGModule(feature_dim=256, num_prototypes=200, num_classes=num_classes)
        
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, img, rr):
        x = self.stem(img)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        fr = self.rrnet(rr)
        z = torch.cat([x, fr], dim=1)
        z = self.feature_extractor(z)
        
        if self.use_rag:
            z = self.rag(z)
        
        return self.classifier(z)


# 2. CNN-LSTM Hybrid Model
class CNNLSTM(nn.Module):
    """CNN-LSTM hybrid architecture for ECG classification"""
    def __init__(self, num_classes, rr_dim=0, use_rag=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        
        self.lstm1 = nn.LSTM(128 * 56, 256, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.3)
        
        self.rrnet = RR_TCN(rr_dim)
        fuse_in = 128 + self.rrnet.out_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(fuse_in, 256),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        
        self.use_rag = use_rag
        if use_rag:
            self.rag = RAGModule(feature_dim=256, num_prototypes=200, num_classes=num_classes)
        
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, img, rr):
        x = self.cnn(img)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, H, -1)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        fr = self.rrnet(rr)
        z = torch.cat([x, fr], dim=1)
        z = self.feature_extractor(z)
        
        if self.use_rag:
            z = self.rag(z)
        
        return self.classifier(z)


# 3. Pure LSTM Model
class PureLSTM(nn.Module):
    """Pure LSTM architecture for ECG classification"""
    def __init__(self, num_classes, rr_dim=0, use_rag=False):
        super().__init__()
        self.lstm1 = nn.LSTM(224, 256, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        
        self.rrnet = RR_TCN(rr_dim)
        fuse_in = 256 + self.rrnet.out_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(fuse_in, 256),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        
        self.use_rag = use_rag
        if use_rag:
            self.rag = RAGModule(feature_dim=256, num_prototypes=200, num_classes=num_classes)
        
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, img, rr):
        B, C, H, W = img.shape
        x = img.squeeze(1)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        fr = self.rrnet(rr)
        z = torch.cat([x, fr], dim=1)
        z = self.feature_extractor(z)
        
        if self.use_rag:
            z = self.rag(z)
        
        return self.classifier(z)


# 4. Enhanced SE-Attention
class EnhancedSE2D(nn.Module):
    """Enhanced SE attention applied after each conv block"""
    def __init__(self, c, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(c//r, 4), 1)
        self.fc2 = nn.Conv2d(max(c//r, 4), c, 1)
    
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class EnhancedDSResSE(nn.Module):
    """DS-Res with Enhanced SE"""
    def __init__(self, num_classes, rr_dim=0, use_rag=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            EnhancedSE2D(32),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.b1_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            EnhancedSE2D(64),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.b2_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            EnhancedSE2D(128),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.b3_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            EnhancedSE2D(128),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.rrnet = RR_TCN(rr_dim)
        fuse_in = 128 * 14 * 14 + self.rrnet.out_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fuse_in, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        
        self.use_rag = use_rag
        if use_rag:
            self.rag = RAGModule(feature_dim=256, num_prototypes=200, num_classes=num_classes)
        
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, img, rr):
        x = self.stem(img)
        x = self.b1_conv(x)
        x = self.b2_conv(x)
        x = self.b3_conv(x)
        x = x.view(x.size(0), -1)
        fr = self.rrnet(rr)
        z = torch.cat([x, fr], dim=1)
        z = self.feature_extractor(z)
        
        if self.use_rag:
            z = self.rag(z)
        
        return self.classifier(z)

class DualBranchSelfAttention(nn.Module):
    """
    Self-attention between morphology feature and RR feature.
    fm 和 fr 作为两个 token 做交互。
    """
    def __init__(self, fm_dim=256, fr_dim=64, attn_dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        self.fm_proj = nn.Linear(fm_dim, attn_dim)
        self.fr_proj = nn.Linear(fr_dim, attn_dim) if fr_dim > 0 else None

        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(attn_dim)
        self.norm2 = nn.LayerNorm(attn_dim)

        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, attn_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.has_rr = fr_dim > 0

    def forward(self, fm, fr=None):
        fm_tok = self.fm_proj(fm).unsqueeze(1)  # [B,1,D]

        if self.has_rr and fr is not None and fr.shape[1] > 0:
            fr_tok = self.fr_proj(fr).unsqueeze(1)  # [B,1,D]
            x = torch.cat([fm_tok, fr_tok], dim=1)  # [B,2,D]
        else:
            x = fm_tok  # [B,1,D]

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        x = x.reshape(x.size(0), -1)  # [B,2D] or [B,D]
        return x
# 5. DSRFromImages with RAG (Main Model)
class DSRFromImages(nn.Module):
    """
    Depthwise Separable Residual Network with Self-Attention + RAG
    """
    def __init__(self, num_classes, rr_dim=0, use_rag=True):
        super().__init__()
        self.morph = MorphologyCNN()
        self.rrnet = RR_TCN(rr_dim)

        self.self_attn_fusion = DualBranchSelfAttention(
            fm_dim=256,
            fr_dim=self.rrnet.out_dim,
            attn_dim=256,
            num_heads=4,
            dropout=0.1
        )

        attn_out_dim = 256 * 2 if self.rrnet.out_dim > 0 else 256

        # 保留 self.fuse 这个名字，方便你现有 infer 代码继续识别
        self.fuse = nn.Sequential(
            nn.Linear(attn_out_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )

        self.use_rag = use_rag
        if use_rag:
            self.rag = RAGModule(feature_dim=256, num_prototypes=200, num_classes=num_classes)
            print("✓ DSRFromImages with Self-Attention + RAG enabled")
        else:
            print("✓ DSRFromImages with Self-Attention enabled")

        self.cls = nn.Linear(256, num_classes)

    def forward(self, img, rr):
        fm = self.morph(img)      # [B,256]
        fr = self.rrnet(rr)       # [B,rr_out_dim] or [B,0]

        z = self.self_attn_fusion(fm, fr)
        z = self.fuse(z)

        if self.use_rag:
            z = self.rag(z)

        return self.cls(z)

class OriginalDSRAblation(nn.Module):
    """
    Model class matched with Original_DSR_Binary_Ablation.ipynb.

    Supported variants:
    - Original_DSR
    - Original_DSR_Attn
    - Original_DSR_RAG
    - Original_DSR_Attn_RAG
    """

    def __init__(self, num_classes=2, rr_dim=0, use_attn=False, use_rag=False):
        super().__init__()

        self.use_attn = bool(use_attn)
        self.use_rag = bool(use_rag)

        self.morph = MorphologyCNN()
        self.rrnet = RR_TCN(rr_dim)

        if self.use_attn:
            self.self_attn_fusion = DualBranchSelfAttention(
                fm_dim=256,
                fr_dim=self.rrnet.out_dim,
                attn_dim=256,
                num_heads=4,
                dropout=0.1,
            )

            attn_out_dim = 256 * 2 if self.rrnet.out_dim > 0 else 256

            self.fuse = nn.Sequential(
                nn.Linear(attn_out_dim, 256),
                nn.ReLU(True),
                nn.Dropout(0.2),
            )

        else:
            fuse_in = 256 + self.rrnet.out_dim

            self.fuse = nn.Sequential(
                nn.Linear(fuse_in, 256),
                nn.ReLU(True),
                nn.Dropout(0.2),
            )

        if self.use_rag:
            self.rag = RAGModule(
                feature_dim=256,
                num_prototypes=200,
                num_classes=num_classes,
            )

        self.cls = nn.Linear(256, num_classes)

    def extract_features(self, img, rr):
        fm = self.morph(img)
        fr = self.rrnet(rr)

        if self.use_attn:
            z = self.self_attn_fusion(fm, fr)
        else:
            z = torch.cat([fm, fr], dim=1)

        z = self.fuse(z)
        return z

    def forward(self, img, rr):
        z = self.extract_features(img, rr)

        if self.use_rag:
            z = self.rag(z)

        return self.cls(z)
print("="*80)
print("✓ Model architectures loaded with RAG support:")
print("  - DSRFromImages (Main: DS-Res + Wavelet + Temporal + RAG)")
print("  - EnhancedDSResSE (Enhanced SE-Attention + optional RAG)")
print("  - PureCNN (4 conv blocks + optional RAG)")
print("  - CNNLSTM (CNN + LSTM hybrid + optional RAG)")
print("  - PureLSTM (Bidirectional LSTM + optional RAG)")
print("="*80)
