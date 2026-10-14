"""L0.5 + L1: isotropic noise on the sphere, with an optional data-matched spectrum.

Drop-in replacement for ``DynamicInterpolant.generator``. The interpolant calls it
as ``generator(b, c, device=...)`` and expects ``[b, c, H, W]``, which is the same
signature ``modules/diffusion/utils.py:SphereNoiseGenerator`` uses.

Why not just ``torch.randn``  (L0.5)
-------------------------------------
On a 0.25 deg equiangular grid a cell is 27.83*cos(lat) km wide, so the top row is
61 m across and 458x narrower than the equator. iid per-gridpoint noise therefore
puts equal variance into all 720 zonal modes at every latitude, but near the poles
most of those modes sit at physical scales the data does not contain. Measured on
OM4 (see MDiT/SNR_FIX.md section 3.1): at 75-88N, **85%** of iid noise variance
lands above m = 720 cos(lat), where thetao_0 has 2.4% of its power and uo_0 has
0.7%. GenCast fixes this by drawing coefficients in the spherical-harmonic domain
with a flat expected power spectrum and inverse-transforming to the grid; this
class does the same.

Truncation degree (lmax)
------------------------
Pinned explicitly to (nlat+1)//2 = 360, never left to the library default:
torch_harmonics changed that default in v0.9.0 (equiangular grids now truncate to
(nlat+1)//2 instead of nlat), so the same code silently produced a different
forward process in myenv (0.8.0 -> 720) and ocean-emu (0.9.2 -> 360).

360 is also the right value on the evidence. Measured equatorial zonal power above
wavenumber 360:

    OM4 thetao_0 data   1.13%
    noise, lmax = 720  38.87%     <- injects structure the data does not have
    noise, lmax = 360   0.00%

i.e. 720 errs by adding 39% of spurious small-scale power, 360 errs by omitting
1.1%. 360 is also what equiangular quadrature actually supports on 720 latitudes.
The cost is that the forward process injects nothing below ~55 km, so the model
has no generative spread there -- worth remembering if small-scale ensemble spread
ever matters, and recoverable via the L1 spectrum if so.

Measured with this implementation at 720x1440, lmax=360:
    pointwise std   1.010 (equator) .. 0.921 (85-90N)
    power above m = 720 cos(lat) at |lat|>75:  0.15%   (iid randn: ~85%)
    cost            (1,77,720,1440) in 12 ms on an A100; 3.8 ms/channel batched
                    vs 150 ms/channel unbatched -- ALWAYS batch.

Optional data-matched spectrum  (L1)
------------------------------------
``Cl`` scales the per-degree amplitude. Flat (``Cl=None``) is L0.5. Passing
``C_l(data) ** alpha`` interpolates toward full whitening:

    SNR at degree l  ~  C_l(data) / C_l(noise)  =  C_l ** (1 - alpha)

so alpha=0 keeps the full 4.09-nat spread across scales (and with it the
coarse-to-fine generation curriculum), alpha=1 removes it entirely (and the
curriculum with it). Sweep alpha in {0, 0.25, 0.5}; do not start at 1.

Env: needs torch_harmonics (myenv has 0.8.0, ocean-emu has 0.9.2 -- see lmax note).
"""
import torch
import torch.nn as nn
import torch_harmonics as th


class OceanSphereNoise(nn.Module):
    """Isotropic Gaussian noise on an equiangular lat-lon grid.

    Args:
        nlat, nlon: grid shape (720, 1440 for OM4 0.25 deg).
        Cl: None for a flat spectrum (L0.5), else a (lmax,) or (nchan, lmax)
            tensor of per-degree power (L1). Not normalised -- calibrate()
            absorbs the overall scale.
        mask: optional (1, nchan, nlat, nlon) land mask. When given, every draw is
            zeroed over land, so the interpolant keeps land at exactly 0 in BOTH
            training and sampling without the callers having to remember. Applied
            after calibrate(), so the calibration still targets unit variance on
            the sphere rather than unit variance over the ocean only.
    """

    def __init__(self, nlat=720, nlon=1440, Cl=None, mask=None, lmax=None):
        super().__init__()
        # lmax is pinned explicitly and never left to the library default:
        # torch_harmonics changed it in v0.9.0 (equiangular grids now truncate to
        # (nlat+1)//2, i.e. 360 instead of 720 here), which silently changes the
        # noise spectrum -- and therefore the forward process -- between envs.
        # myenv has 0.8.0, ocean-emu has 0.9.2. See the module docstring for which
        # value we use and why.
        if lmax is None:
            lmax = (nlat + 1) // 2      # 360 for the OM4 grid; see docstring
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, grid='equiangular')
        lmax, mmax = self.isht.lmax, self.isht.mmax
        self.lmax, self.mmax = lmax, mmax

        l = torch.arange(lmax).view(-1, 1)
        m = torch.arange(mmax).view(1, -1)
        # a_lm is only defined for m <= l; the rest of the (l, m) rectangle is zero.
        self.register_buffer('valid', (m <= l), persistent=False)

        if Cl is None:
            Cl = torch.ones(lmax)
        Cl = torch.as_tensor(Cl, dtype=torch.float32).view(-1, lmax)   # (K, lmax)
        self.register_buffer('amp', Cl.sqrt().unsqueeze(-1))           # (K, lmax, 1)
        self.n_spec = self.amp.shape[0]

        # Scale is fixed empirically by calibrate(), so we never have to care what
        # normalisation convention torch_harmonics uses. One entry per spectrum:
        # with a per-channel Cl each channel has a different total power, so each
        # needs its own scale to end up at unit pointwise variance.
        self.register_buffer('scale', torch.ones(self.n_spec, 1, 1))
        self._calibrated = False
        if mask is None:
            self.mask = None
        else:
            self.register_buffer('mask', torch.as_tensor(mask, dtype=torch.float32))

    def _raw(self, b, c, device, dtype=torch.float32):
        if self.n_spec == 1:
            amp = self.amp                                   # (1, lmax, 1)
        else:
            assert self.n_spec == c, \
                f'per-channel Cl has {self.n_spec} rows but {c} channels were asked for'
            amp = self.amp.unsqueeze(0)                       # (1, c, lmax, 1)

        shape = (b, c, self.lmax, self.mmax)
        re = torch.randn(shape, device=device, dtype=dtype)
        im = torch.randn(shape, device=device, dtype=dtype)
        im[..., 0] = 0.0                                      # m = 0 coefficients are real
        a = torch.complex(re, im) * (amp / 2 ** 0.5) * self.valid
        a[..., 0] = a[..., 0] * 2 ** 0.5                      # restore m=0 variance
        # torch_harmonics calls view_as_complex internally, which has no bfloat16
        # kernel, so the transform must run outside any enclosing autocast region.
        # Noise generation is cheap (24 ms for (1,77,720,1440) on an A100) and
        # carries no gradient, so keeping it in fp32 costs nothing.
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            return self.isht(a)                               # (b, c, nlat, nlon)

    @torch.no_grad()
    def calibrate(self, device, n=4):
        """Set `scale` so the field has unit pointwise variance. Call once.

        With a per-channel spectrum every channel needs its own scale, so this
        draws all n_spec channels rather than a single one.
        """
        self.scale.fill_(1.0)
        v = self._raw(n, self.n_spec, device).var(dim=(-2, -1))   # (n, n_spec)
        self.scale.copy_((1.0 / v.sqrt()).mean(dim=0).view(-1, 1, 1).cpu())
        self._calibrated = True
        return float(self.scale.mean())

    def forward(self, b, c, device, dtype=torch.complex64):
        # `dtype` is accepted (and ignored) so the signature matches the
        # SphereNoiseGenerator that DynamicInterpolant was written against.
        if not self._calibrated:
            self.calibrate(device)
        z = self._raw(b, c, device) * self.scale
        if self.mask is not None:
            z = z * self.mask
        return z


def load_conditioned_Cl(path, alpha, nchan=77, lmax=360):
    """Load the measured angular spectrum and raise it to `alpha` (L1).

    `path` points at MDiT/data/angular_spectra_conditioned.npz, produced by
    MDiT/scripts/{angular_spectra,spectrum_utils}.py. The RAW spectrum must not be
    used: land is zero in the cache so the masked SHT leaks power to high l, and
    C_l ** alpha divides by that estimate. The conditioned file is smoothed in
    log-log space with a power-law tail beyond l = 300.

    alpha == 0 returns None, i.e. a flat spectrum -- identical to plain L0.5, so
    the same code path covers the whole sweep.
    """
    if alpha == 0.0:
        return None
    import numpy as np
    d = np.load(path)
    Cl = np.asarray(d['Cl'], dtype=np.float64)                # (77, 720) as measured
    assert Cl.shape[0] == nchan, f'unexpected Cl shape {Cl.shape}'
    assert Cl.shape[1] >= lmax, f'Cl only reaches l={Cl.shape[1]-1}, need {lmax-1}'
    Cl = Cl[:, :lmax]                                         # follow the pinned lmax
    Cl = Cl / Cl[:, 1:11].mean(axis=1, keepdims=True)         # scale -> calibrate()
    return torch.from_numpy(Cl ** alpha).float()
