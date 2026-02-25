import numpy as np
import torch
import MRzeroCore as mr0


def get_TI_from_seq(
    seq0: mr0.sequence.Sequence,
) -> torch.Tensor:
    """Get inversion time for each shot in the input sequence.

    Args:
        seq0 (mr0.sequence.Sequence): input sequence.

    Returns:
        torch.Tensor: Inversion time for each shot.
    Here the inversion time includes the duration of the inversion pulse.
    Assumes the readout flip angle to be less than 80 degrees for dummy scans.
    """ 
    times_IR, idx_IRs = get_IR(seq0)
    full_kspace = seq0.get_full_kspace()
    # compute the indices of low frequency by IR block
    idx_next_IR = idx_IRs[1:] + [len(seq0)] 
    idx_low_freq = []
    for i_IR, i_next_IR in zip(idx_IRs, idx_next_IR):
        candidates = []
        col_center_idxs = {}
        for i, r in enumerate(seq0[i_IR:i_next_IR]):
            if r.adc_usage.sum() > 0:
                candidates.append(i_IR + i)
                col_center_idxs[i_IR + i] = get_col_center_idx(r)

        if len(candidates) == 0:
            # dummy scan
            for i, r in enumerate(seq0[i_IR:i_next_IR]):
                if r.pulse.angle <= np.deg2rad(80):
                    candidates.append(i_IR + i)
                    col_center_idxs[i_IR + i] = r.gradm.shape[0]//2 # dummy value
                    
        assert len(candidates) > 0, "No readout found or dummy scan with flip angle < 80 degrees. Cannot compute TI."
        best_idx = min(
            candidates,
            key=lambda i: torch.linalg.norm(full_kspace[i][col_center_idxs[i], 1:-1])  # Distance to center phase
        )
        idx_low_freq.append(best_idx)
        
    times_after = torch.cumsum(torch.tensor([r.event_time.sum() for r in seq0]), dim=0)
    times_before = torch.tensor([ 0. ] + list(times_after[:-1]))
    times_low_freq = times_before[idx_low_freq]
    return times_low_freq-times_IR


def get_IR(seq0: mr0.sequence.Sequence):
    """Get IR times and indices from a sequence.

    Args:
        seq (mr0.sequence.Sequence): MRzero base sequence.
        
    Returns:
        torch.Tensor: timing before the inversion recovery pulses.
        torch.Tensor: Indices of the inversion recovery pulses.
    """
    # compute the indices of each inversion
    angle_max = max([r.pulse.angle for r in seq0])
    idx_IRs = [i for i, r in enumerate(seq0) if r.pulse.angle==angle_max]
    # Compute the minimum durations of the TRs
    times_after = torch.cumsum(torch.tensor([r.event_time.sum() for r in seq0]), dim=0)
    times_before = torch.tensor([ 0. ] + list(times_after[:-1]))
    times_IR = times_before[idx_IRs]
    return times_IR, idx_IRs


def get_col_center_idx(rep):
    if rep.adc_usage.sum() > 0:
        kspace = rep.gradm
        adc_mask = rep.adc_usage>0
        center = kspace[adc_mask][:,0].abs().argmin()
        return center + torch.where(adc_mask)[0][0]
    raise AssertionError('Sequence with no ADC signal !')