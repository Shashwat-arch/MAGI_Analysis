import torch
import torch.nn as nn
import torch.nn.functional as F  # Added for normalization

class Loss(nn.Module):
    def __init__(self, temperature=0.07, scale_by_temperature=True, scale_by_weight=False):
        super(Loss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.scale_by_weight = scale_by_weight

    def forward(self, out, mask):
        device = out.device  # Simplified device selection

        row, col, val = mask.storage.row(), mask.storage.col(), mask.storage.value()
        row, col, val = row.to(device), col.to(device), val.to(device)
        batch_size = out.shape[0]

        # Normalize embeddings to get cosine similarity
        out_norm = F.normalize(out, p=2, dim=1)  # L2 normalization
        
        # Compute cosine similarity matrix
        cos_sim = torch.matmul(out_norm, out_norm.T)  # Already in [-1, 1] range
        cos_sim = torch.div(cos_sim, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(cos_sim, dim=1, keepdim=True)
        cos_sim = cos_sim - logits_max.detach()

        # Create mask to ignore self-comparisons
        logits_mask = torch.scatter(
            torch.ones(batch_size, batch_size, device=device),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0
        )

        # Compute softmax probabilities
        exp_logits = torch.exp(cos_sim) * logits_mask
        log_probs = cos_sim - torch.log(exp_logits.sum(1, keepdim=True))

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        # Calculate loss using positive pairs from random walks
        log_probs_pos = log_probs[row, col]
        loss = -log_probs_pos.mean()

        if self.scale_by_temperature:
            loss *= self.temperature
            
        return loss