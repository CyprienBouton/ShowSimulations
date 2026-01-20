import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd

def extract_indices_from_kspace_coords(kspace_coords: torch.Tensor, dk: float) -> torch.Tensor:
    """
    Compute discrete k-space indices for each coordinate in the input array.
    This avoids cumulative floating-point error by rounding each gap individually.

    Args:
        kspace_coords (torch.Tensor): 1D tensor of k-space coordinates (e.g., phase-encode positions).
        dk (float): Expected k-space step size along this axis.

    Returns:
        torch.Tensor: 1D tensor of integer indices, same shape as kspace_coords.
    """
    device = kspace_coords.device

    # Get sorted unique coordinates and inverse mapping
    unique_coords, inverse = torch.unique(kspace_coords, sorted=True, return_inverse=True)

    # Compute relative step sizes between consecutive coordinates,
    # normalized by dk and rounded to the nearest integer
    gaps = torch.round(torch.diff(unique_coords) / dk).to(torch.long)

    # Insert a zero at the beginning so length matches unique_coords
    gaps = torch.cat([torch.zeros(1, dtype=torch.long, device=device), gaps])

    # Cumulative sum → integer indices for unique coords
    unique_indices = torch.cumsum(gaps, dim=0)

    # Map each original coordinate back to its index
    return unique_indices[inverse]


class RecoMRzero:
    def __init__(self, seq0, dx=None, dy=None, dz=None):
        self.seq0 = seq0
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.cartesian_kspace_grid()

    def cartesian_kspace_grid(self, precision=5e-2,
                              dtype=torch.complex64, device='cpu'):
        """
        Build a Cartesian k-space grid from non-uniform k-space coordinates and signal.

        Returns:
            grid: tensor [nx, ny, nz, ncoils]
            kx_vals, ky_vals, kz_vals: 1D tensors of grid coordinates
        """
        kspace_coords = self.seq0.get_kspace()
        kx, ky, kz = kspace_coords[:, :3].T

        # Round coordinates
        kx_rounded = ((kx / precision).round() * precision)
        ky_rounded = ((ky / precision).round() * precision)
        kz_rounded = ((kz / precision).round() * precision)

        # Get sorted unique coordinates
        kx_rounded_unique = kx_rounded.unique().sort().values
        ky_rounded_unique = ky_rounded.unique().sort().values
        kz_rounded_unique = kz_rounded.unique().sort().values

        # Determine step sizes if not provided assume fully sampled in center
        if self.dx is None:
            self.dx = kx_rounded_unique.diff()[kx_rounded_unique.argmin()]
        if self.dy is None:
            self.dy = ky_rounded_unique.diff()[ky_rounded_unique.argmin()]
        if self.dz is None:
            if len(kz_rounded_unique) > 1:
                self.dz = kz_rounded_unique.diff()[kz_rounded_unique.argmin()]
            else:
                self.dz = 1/(5e-3)  # Assume 5mm spacing if only one slice

        # Compute nearest grid indices
        # import pdb; pdb.set_trace() 
        self.ix = extract_indices_from_kspace_coords(kx_rounded, self.dx)
        self.iy = extract_indices_from_kspace_coords(ky_rounded, self.dy)
        self.iz = extract_indices_from_kspace_coords(kz_rounded, self.dz)

        # Define grid ranges
        self.nx, self.ny, self.nz = self.ix.max()+1, self.iy.max()+1, self.iz.max()+1
    
    def get_inter_scan_duration(self, eps=0.2, dtype=torch.float, device='cpu'):
        inter_scan_matrix = torch.full((self.ny, self.nz), torch.nan, dtype=dtype, device=device)
        times_after = torch.cumsum(torch.tensor([r.event_time.sum() for r in self.seq0]), dim=0)
        times_before = torch.tensor([ 0. ] + list(times_after[:-1]))
        IR_idx = [i for i, r in enumerate(self.seq0) if abs( r.pulse.angle - torch.pi ) < eps]
        times_IR = times_before[IR_idx]
        inter_scan_durations = torch.diff(times_IR, prepend=torch.tensor([torch.nan]) )
        times_IR_next = torch.cat( [times_IR[1:], torch.tensor([torch.inf])] )
        for time_IR, time_IR_next, duration in zip(times_IR, times_IR_next, inter_scan_durations):
            current_shot = (times_before>time_IR)*(times_before<time_IR_next)
            mask_signal = torch.cat([
                torch.full((int(r.adc_usage.sum()),), is_shot, dtype=torch.int64)
                for is_shot, r in zip(current_shot, self.seq0)
            ]).to(bool)
            inter_scan_matrix[self.iy[mask_signal], self.iz[mask_signal]] = duration
        return inter_scan_matrix
    
    def get_timing_matrix(self, dtype=torch.float, device='cpu'):
        timing_matrix = torch.full((self.ny, self.nz), torch.nan, dtype=dtype, device=device)
        times_after = torch.cumsum(torch.tensor([r.event_time.sum() for r in self.seq0]), dim=0)
        times_signal = torch.cat([
                torch.full((int(r.adc_usage.sum()),), t, dtype=dtype)
                for r, t in zip(self.seq0, times_after)
        ])
        timing_matrix[self.iy, self.iz] = torch.tensor(times_signal, dtype=dtype, device=device)
        return timing_matrix

    def get_reco_dataframe(self):
        timing_matrix = self.get_timing_matrix().cpu().numpy()
        inter_scan_matrix = self.get_inter_scan_duration().cpu().numpy()
        data = []
        for i in range(self.ny):
            for j in range(self.nz):
                data.append([i, j, timing_matrix[i, j], inter_scan_matrix[i, j]])
        df = pd.DataFrame(data, columns=['Lin', 'Par', 'Time', 'RD'])
        return df

if __name__ == "__main__":
    import os
    import pickle
    seq_file = os.path.expanduser('~/Simulation/16_clinical_fl3d_no_acc/TI175_T1MES_no_correction.pkl')
    seq = pickle.load(open(seq_file, 'rb'))
    reco = RecoMRzero(seq)
    reco.get_reco_dataframe()
    print("k-space grid shape:")
    