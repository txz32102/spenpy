"""SPEN forward simulation.

Ported from the existing spen simulation class in spen.py.
"""

import torch
import torch.nn.functional as F
import numpy as np
from spenpy.fft.transform import fft_kspace_to_xspace, fft_xspace_to_kspace
from spenpy.core.matrix import calcInvA


class SpenSim:
    """SPEN forward simulator.

    Generates SPEN k-space data from a given image, including chirp encoding,
    segmentation, phase errors, and noise.
    """

    def __init__(
        self,
        L: list = (4, 4),
        acq_point: list = (256, 256),
        nseg: int = 1,
        chirp_rvalue: float = 120,
        tblip: float = 128e-6,
        gamma_hz: float = 4257.4,
        device: str = "cpu",
        noise_level: float = 0.0,
    ):
        self.L = list(L)
        self.acq_point = list(acq_point)
        self.N = [acq_point[0], acq_point[1] * 16]
        self.x = torch.linspace(-L[0] / 2, L[0] / 2, self.N[0], device=device)
        self.y = torch.linspace(-L[1] / 2, L[1] / 2, self.N[1], device=device)
        self.Ydire_inhomo_coef = torch.zeros(4, device=device)
        self.nseg = nseg
        self.chirp_rvalue = chirp_rvalue
        self.tblip = tblip
        self.gamma_hz = gamma_hz
        self.device = device
        self.noise_level = noise_level
        self.sw = 250000
        self.one_shot_num = acq_point[1] / nseg
        self.Ta = (acq_point[0] / self.sw + tblip) * self.one_shot_num
        self.chirp_tp = self.Ta / 2
        self.a_sign = -1

        self.procpar_struct = {
            "np": acq_point[0],
            "ne": 1,
            "nv": acq_point[1],
            "nseg": nseg,
            "nf": acq_point[1],
            "arraydim": 1,
            "Rvol": chirp_rvalue,
            "lpe": L[0],
            "lro": L[1],
            "ppe": 0,
            "Tp": self.chirp_tp,
            "Gchip": chirp_rvalue / self.chirp_tp / L[1] / gamma_hz,
        }

        self.rfwdth = self.procpar_struct["Tp"]
        self.GPEe = self.procpar_struct["Gchip"]
        self.alfa = self.a_sign * 2 * np.pi * gamma_hz * self.GPEe * self.rfwdth / self.procpar_struct["lpe"]

    def get_InvA(self):
        """Get the reconstruction operator."""
        return calcInvA(self.alfa, self.L[0], self.N[0], 0, self.a_sign, 0, 0.9)

    @torch.no_grad()
    def sim(self, H: torch.Tensor, return_phase_map: bool = False, return_good_lr_image: bool = False):
        """Forward SPEN simulation.

        Currently supports single-shot (nseg=1) SPEN simulation.

        Args:
            H: input image [batch, H, W]
            return_phase_map: also return the phase map
            return_good_lr_image: also return the ground-truth low-res image
        """
        Hb = F.interpolate(
            H.permute(0, 2, 1).unsqueeze(1),
            size=(self.acq_point[0], self.acq_point[1] * 16),
            mode="bilinear",
        ).squeeze(1)

        good_lr_image = torch.zeros(
            (Hb.shape[0], self.acq_point[1], self.acq_point[0]),
            dtype=torch.complex64, device=self.device,
        )

        for k in range(self.nseg):
            start = -self.L[1] / 2 + k * self.L[1] / self.acq_point[1]
            step = self.L[1] / self.acq_point[1]
            n_pe_per_seg = self.acq_point[1] // self.nseg
            end = start + (n_pe_per_seg - 1) * step
            temp_yacq = torch.arange(start, end + step / 2, step, device=self.device)

            b0y = sum(c * self.y**i for i, c in enumerate(self.Ydire_inhomo_coef[:4]))
            b0acq = sum(c * temp_yacq**i for i, c in enumerate(self.Ydire_inhomo_coef[:4]))

            y_grid, temp_yacq_grid = torch.meshgrid(self.y, temp_yacq, indexing="ij")
            b0y_grid, b0acq_grid = torch.meshgrid(b0y, b0acq, indexing="ij")
            part1 = (y_grid + b0y_grid) - (temp_yacq_grid + b0acq_grid)
            part2 = temp_yacq_grid + b0acq_grid
            exp_term = torch.exp(1j * self.alfa * (part1**2 - part2**2)).to(torch.complex64)

            n_full_pe = self.acq_point[1] * 16
            pe_start = k * n_pe_per_seg * 16
            pe_end = (k + 1) * n_pe_per_seg * 16

            Hb_seg = Hb[:, :, pe_start:pe_end].to(torch.complex64)
            temprxyacq = torch.matmul(Hb_seg, exp_term)
            good_lr_image[:, k::self.nseg, :] = temprxyacq.permute(0, 2, 1)

            temprxyacq = temprxyacq.permute(0, 2, 1)
            temprxyacq = fft_xspace_to_kspace(temprxyacq, dim=1)
            img = fft_kspace_to_xspace(temprxyacq, dim=1)

            # Even/odd phase error simulation for odd single-shot
            if self.nseg == 1:
                im_odd = img[:, 0::2, :]
                im_even = img[:, 1::2, :]

                images = H
                KspaceTest = fft_xspace_to_kspace(
                    fft_xspace_to_kspace(images, dim=1), dim=0
                )
                KspaceTestZero = torch.zeros(
                    [Hb.shape[0], int(self.acq_point[0] / 2), self.acq_point[1]],
                    dtype=torch.complex64, device=self.device,
                )
                lines = 10
                sz_x = int(KspaceTestZero.shape[1] / 2)
                sz_y = int(KspaceTestZero.shape[2] / 2)
                sk_x = int(KspaceTest.shape[1] / 2)
                sk_y = int(KspaceTest.shape[2] / 2)
                KspaceTestZero[:, sz_x - lines:sz_x + lines, sz_y - lines:sz_y + lines] = \
                    KspaceTest[:, sk_x - lines:sk_x + lines, sk_y - lines:sk_y + lines]

                map_img = fft_kspace_to_xspace(
                    fft_kspace_to_xspace(KspaceTestZero, dim=0), dim=1
                )
                map_img = torch.abs(map_img) / torch.max(torch.abs(map_img))
                emap = torch.exp(1j * 2 * np.pi * map_img)

                target_h, target_w = im_even.shape[1], im_even.shape[2]
                emap_real = F.interpolate(
                    emap.real.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False
                ).squeeze(1)
                emap_imag = F.interpolate(
                    emap.imag.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False
                ).squeeze(1)
                emap_resized = emap_real + 1j * emap_imag

                im_even = im_even * emap_resized
                img[:, 1::2, :] = im_even

            temprxyacq = fft_xspace_to_kspace(img, dim=1)
            temprxyacq = fft_kspace_to_xspace(temprxyacq, dim=1)
            temprxyacq = temprxyacq / torch.max(torch.abs(temprxyacq))

            final_rxyacq = temprxyacq + self.noise_level * torch.randn_like(temprxyacq)
            final_rxyacq_ROFFT = fft_kspace_to_xspace(final_rxyacq, dim=1)

        max_vals = torch.max(torch.abs(good_lr_image).view(good_lr_image.shape[0], -1), dim=1)[0]
        good_lr_image = good_lr_image / max_vals.view(-1, 1, 1)

        phase_map_ideal = None
        if return_phase_map and return_good_lr_image:
            return final_rxyacq_ROFFT, phase_map_ideal, good_lr_image
        elif return_good_lr_image:
            return final_rxyacq_ROFFT, good_lr_image
        elif return_phase_map:
            return final_rxyacq_ROFFT, phase_map_ideal
        else:
            return final_rxyacq_ROFFT
