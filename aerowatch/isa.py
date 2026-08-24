"""International Standard Atmosphere model, analytic layers up to 20 km.

Implements the ISO 2533:1975 / ICAO Doc 7488 model:
  layer 0 (troposphere):  0 to 11 km geopotential, lapse rate 6.5 K/km
  layer 1 (tropopause+):  11 to 20 km geopotential, isothermal at 216.65 K

All functions take geopotential altitude in meters unless noted. Helpers
convert between geometric and geopotential altitude using the standard
spherical earth radius from the model.

Functions accept scalars or numpy arrays and return the same shape.
Scalars come back as plain Python floats.
"""

import numpy as np

# Primary constants (ISO 2533:1975 / ICAO Doc 7488)
T0 = 288.15          # sea level temperature, K
P0 = 101325.0        # sea level pressure, Pa
RHO0 = 1.225         # sea level density, kg/m^3
G0 = 9.80665         # standard gravity, m/s^2
R_AIR = 287.05287    # specific gas constant for dry air, J/(kg K)
GAMMA = 1.4          # ratio of specific heats for air
LAPSE = 0.0065       # tropospheric lapse rate, K/m
H_TROP = 11000.0     # tropopause geopotential altitude, m
H_TOP = 20000.0      # model ceiling, m (geopotential)
H_FLOOR = -611.0     # model floor, m (geopotential)
R_EARTH = 6356766.0  # effective earth radius for geopotential conversion, m

# Derived layer constants
T_TROP = T0 - LAPSE * H_TROP                      # 216.65 K
_TROPO_EXP = G0 / (R_AIR * LAPSE)                 # about 5.25588
P_TROP = P0 * (T_TROP / T0) ** _TROPO_EXP         # pressure at 11 km
A0 = float(np.sqrt(GAMMA * R_AIR * T0))           # sea level speed of sound

FT = 0.3048  # meters per foot


def _as_scalar_or_array(x, was_scalar):
    if was_scalar:
        return float(x)
    return x


def _validated(h):
    was_scalar = np.isscalar(h)
    arr = np.asarray(h, dtype=float)
    if np.any(arr < H_FLOOR) or np.any(arr > H_TOP):
        raise ValueError(
            "geopotential altitude out of model range [%g, %g] m" % (H_FLOOR, H_TOP)
        )
    return arr, was_scalar


def geometric_to_geopotential(z):
    """Convert geometric altitude z (m) to geopotential altitude h (m)."""
    z = np.asarray(z, dtype=float)
    out = R_EARTH * z / (R_EARTH + z)
    return _as_scalar_or_array(out, out.ndim == 0)


def geopotential_to_geometric(h):
    """Convert geopotential altitude h (m) to geometric altitude z (m)."""
    h = np.asarray(h, dtype=float)
    out = R_EARTH * h / (R_EARTH - h)
    return _as_scalar_or_array(out, out.ndim == 0)


def temperature(h):
    """Static temperature (K) at geopotential altitude h (m)."""
    arr, was_scalar = _validated(h)
    out = np.where(arr < H_TROP, T0 - LAPSE * arr, T_TROP)
    return _as_scalar_or_array(out, was_scalar)


def pressure(h):
    """Static pressure (Pa) at geopotential altitude h (m)."""
    arr, was_scalar = _validated(h)
    t_below = T0 - LAPSE * np.minimum(arr, H_TROP)
    p_below = P0 * (t_below / T0) ** _TROPO_EXP
    p_above = P_TROP * np.exp(-G0 * (arr - H_TROP) / (R_AIR * T_TROP))
    out = np.where(arr < H_TROP, p_below, p_above)
    return _as_scalar_or_array(out, was_scalar)


def density(h):
    """Density (kg/m^3) at geopotential altitude h (m)."""
    arr, was_scalar = _validated(h)
    out = pressure(arr) / (R_AIR * temperature(arr))
    return _as_scalar_or_array(out, was_scalar)


def speed_of_sound(h):
    """Speed of sound (m/s) at geopotential altitude h (m)."""
    arr, was_scalar = _validated(h)
    out = np.sqrt(GAMMA * R_AIR * temperature(arr))
    return _as_scalar_or_array(out, was_scalar)


def atmosphere(h):
    """Return (temperature K, pressure Pa, density kg/m^3, speed of sound m/s)."""
    return temperature(h), pressure(h), density(h), speed_of_sound(h)
