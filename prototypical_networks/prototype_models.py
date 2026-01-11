import torch
import torch.nn as nn


class RobustSequenceEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, output_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, mask=None, return_temporal=False):
        # x: (B, L, D) -> (B, D, L)
        x = x.permute(0, 2, 1) 
        temporal_features = self.feature_extractor(x) # (B, Out_Dim, L)
        
        if return_temporal:
            return temporal_features.permute(0, 2, 1)
        
        # --- CRITICAL: MASKED MAX POOLING ---
        if mask is not None:
            # mask: (B, L) -> Reshape to (B, 1, L) to match features
            mask = mask.unsqueeze(1) 
            # Set padded areas to a very small number so max() ignores them
            temporal_features = temporal_features.masked_fill(mask == 0, -1e9)
        
        pooled = torch.max(temporal_features, dim=2)[0] 
        
        # L2 Normalization (Essential for Euclidean Distance stability)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

def robust_prototypical_loss(embeddings, n_way, k_shot, n_query):
    """
    embeddings: (Batch_Size * Samples_Per_Episode, Feature_Dim)
    """
    device = embeddings.device
    samples_per_episode = n_way * (k_shot + n_query)
    batch_size = embeddings.size(0) // samples_per_episode
    
    # 1. Reshape to (Batch_Size, Samples_Per_Episode, Feature_Dim)
    embeddings = embeddings.view(batch_size, samples_per_episode, -1)
    
    # 2. Split into Support and Query
    # support: (Batch_Size, n_way * k_shot, Dim)
    # query: (Batch_Size, n_way * n_query, Dim)
    support = embeddings[:, :n_way * k_shot]
    query = embeddings[:, n_way * k_shot:]
    
    # 3. Calculate prototypes for each episode in the batch
    # prototypes shape: (Batch_Size, n_way, Dim)
    prototypes = support.reshape(batch_size, n_way, k_shot, -1).mean(dim=2)
    
    # 4. Compute Squared Euclidean Distance
    # Using torch.cdist per batch item (requires some broadcasting or loop)
    # query: (B, Q_total, Dim), prototypes: (B, Way, Dim)
    distances = torch.cdist(query, prototypes)**2  # (B, Q_total, Way)
    
    # 5. Calculate Cross Entropy Loss
    # Flatten across batch and query dims
    target_labels = torch.arange(n_way).repeat_interleave(n_query).to(device)
    target_labels = target_labels.repeat(batch_size) # Repeat for each episode in batch
    
    # Reshape distances to (B * Q_total, Way)
    logits = -distances.reshape(-1, n_way)
    
    loss = torch.nn.functional.cross_entropy(logits, target_labels)
    
    return loss