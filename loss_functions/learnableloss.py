import torch
import torch.nn as nn
import torch.nn.functional as F

class BioticLoss(nn.Module):
    def __init__(self, embed_dim, temperature=0.07, 
                 init_weights='identity', scale_by_temperature=True):
        super().__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        
        # Learnable similarity parameters
        self.W = nn.Parameter(torch.empty(embed_dim, embed_dim))
        
        # Initialize weights
        if init_weights == 'identity':
            nn.init.eye_(self.W)
        elif init_weights == 'orthogonal':
            nn.init.orthogonal_(self.W)
        else:
            nn.init.xavier_uniform_(self.W)

    def forward(self, out, mask):
        device = out.device
        row, col, val = mask.storage.row(), mask.storage.col(), mask.storage.value()
        row, col = row.to(device), col.to(device)
        batch_size = out.shape[0]

        # Compute parametric similarity
        transformed = torch.matmul(out, self.W)  # [batch, dim]
        sim_matrix = torch.matmul(transformed, out.T)  # [batch, batch]
        sim_matrix = torch.div(sim_matrix, self.temperature)

        # Numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()

        # Mask self-comparisons
        logits_mask = torch.scatter(
            torch.ones(batch_size, batch_size, device=device),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0
        )

        # Compute probabilities
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_probs = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        # Calculate loss
        log_probs_pos = log_probs[row, col]
        loss = -log_probs_pos.mean()

        if self.scale_by_temperature:
            loss *= self.temperature
            
        return loss