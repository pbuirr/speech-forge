"""Local conditional sequence objectives; no semantic verification implied."""
import torch
import torch.nn.functional as F

def sequence_token_nll(logits, targets, pad_id=0):
    # Flat token classification avoids the much slower spatial CUDA loss kernel.
    losses=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),
                           ignore_index=pad_id,reduction='none').reshape_as(targets)
    return losses.sum(1)/targets.ne(pad_id).sum(1).clamp_min(1)

def conditional_contrast_loss(paired_logits, mismatched_logits, targets, valid, margin=.1, pad_id=0):
    """Encourage an observed pair to score above a mismatched source.
    Compare the SAME target so length cannot provide the contrast. False negatives
    and noisy observed pairs remain possible; valid masks exclude identical data.
    This is a training preference, not a calibrated correctness probability.
    """
    paired=sequence_token_nll(paired_logits,targets,pad_id)
    mismatched=sequence_token_nll(mismatched_logits,targets,pad_id)
    penalties=F.relu(margin+paired-mismatched)
    return (penalties*valid).sum()/valid.sum().clamp_min(1)
