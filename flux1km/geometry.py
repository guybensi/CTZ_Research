"""Sun/view geometry: glint angle, angle encoding, insolation and albedo normalisation."""
from __future__ import annotations

import numpy as np

SOLAR_CONSTANT = 1361.0  # W m^-2


def glint_angle(sza, vza, raa, raa_convention: str = "specular_at_0") -> np.ndarray:
    """Sun-glint angle in degrees (Bertrand et al. 2006, Eq. 14).

    cos(g) = mu_s*mu_v + sin(sza)*sin(vza)*cos(phi). With the default convention phi = raa,
    so raa = 0 is the specular direction. Use "specular_at_180" if the data define
    RAA = 180 as specular (phi = 180 - raa). Which one CERES/MODIS files use is checked in E0.

    Example:
        >>> float(glint_angle(30.0, 30.0, 0.0))
        0.0

    Args:
        sza, vza, raa: Angles in degrees; broadcastable arrays or scalars. NaN propagates.
        raa_convention: "specular_at_0" or "specular_at_180".

    Returns:
        Glint angle in degrees, in [0, 180], with the broadcast shape of the inputs.

    Raises:
        ValueError: If raa_convention is unknown.
    """
    if raa_convention not in ("specular_at_0", "specular_at_180"):
        raise ValueError(f"unknown raa_convention {raa_convention!r}")
    sza, vza, raa = (np.radians(np.asarray(a, dtype=float)) for a in (sza, vza, raa))
    phi = raa if raa_convention == "specular_at_0" else np.pi - raa
    cos_g = np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(phi)
    return np.degrees(np.arccos(np.clip(cos_g, -1.0, 1.0)))


def encode_angles(sza, vza, raa) -> np.ndarray:
    """Encode geometry as [cos SZA, cos VZA, sin RAA, cos RAA] along a new last axis.

    Example:
        >>> encode_angles(0.0, 0.0, 90.0).round(6).tolist()
        [1.0, 1.0, 1.0, 0.0]
    """
    sza, vza, raa = (np.radians(np.asarray(a, dtype=float)) for a in (sza, vza, raa))
    sza, vza, raa = np.broadcast_arrays(sza, vza, raa)
    return np.stack([np.cos(sza), np.cos(vza), np.sin(raa), np.cos(raa)], axis=-1)


def toa_insolation(sza, solar_constant: float = SOLAR_CONSTANT, earth_sun_au: float = 1.0) -> np.ndarray:
    """Incoming TOA solar flux S0*cos(SZA)/d^2 in W m^-2, zero when the sun is below the horizon.

    Example:
        >>> round(float(toa_insolation(60.0)), 6)
        680.5
    """
    mu0 = np.cos(np.radians(np.asarray(sza, dtype=float)))
    return solar_constant * np.clip(mu0, 0.0, None) / earth_sun_au**2


def flux_to_albedo(flux, sza, **kwargs) -> np.ndarray:
    """TOA albedo = flux / insolation; NaN where the insolation is zero.

    Example:
        >>> round(float(flux_to_albedo(136.1, 0.0)), 6)
        0.1
    """
    s = toa_insolation(sza, **kwargs)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, np.asarray(flux, dtype=float) / s, np.nan)


def albedo_to_flux(albedo, sza, **kwargs) -> np.ndarray:
    """Inverse of flux_to_albedo.

    Example:
        >>> float(albedo_to_flux(0.1, 0.0))
        136.1
    """
    return np.asarray(albedo, dtype=float) * toa_insolation(sza, **kwargs)
