"""Airspeed conversions: CAS, EAS, TAS, Mach.

Conventions used here (standard subsonic compressible-flow definitions):

  Mach       M = TAS / a, with a the local speed of sound.
  Impact pressure  qc = p * ((1 + 0.2 M^2)^3.5 - 1)   (subsonic pitot formula)
  CAS        the speed that would produce the same qc at sea level ISA:
             qc = P0 * ((1 + 0.2 (CAS/A0)^2)^3.5 - 1)
  EAS        TAS * sqrt(rho / rho0), same incompressible dynamic pressure at
             sea level density.

All speeds are in m/s internally. KT converts knots to m/s. Altitudes are
geopotential meters, evaluated in the ISA model from aerowatch.isa.

Only subsonic conversions are supported. Functions raise ValueError when a
conversion implies Mach >= 1.
"""

import numpy as np

from . import isa

KT = 0.514444  # m/s per knot (exact: 1852 m per NM / 3600 s)


def _check_subsonic(mach):
    if np.any(np.asarray(mach) >= 1.0):
        raise ValueError("supersonic condition: these relations are subsonic only")


def mach_from_tas(tas, h):
    """Mach number from true airspeed (m/s) at geopotential altitude h (m)."""
    return tas / isa.speed_of_sound(h)


def tas_from_mach(mach, h):
    """True airspeed (m/s) from Mach at geopotential altitude h (m)."""
    return mach * isa.speed_of_sound(h)


def qc_from_mach(mach, p):
    """Impact pressure (Pa) from Mach and static pressure p (Pa), subsonic."""
    _check_subsonic(mach)
    return p * ((1.0 + 0.2 * np.asarray(mach) ** 2) ** 3.5 - 1.0)


def mach_from_qc(qc, p):
    """Mach from impact pressure qc (Pa) and static pressure p (Pa), subsonic."""
    mach = np.sqrt(5.0 * ((np.asarray(qc) / p + 1.0) ** (2.0 / 7.0) - 1.0))
    _check_subsonic(mach)
    return mach


def qc_from_cas(cas):
    """Impact pressure (Pa) from calibrated airspeed (m/s)."""
    _check_subsonic(np.asarray(cas) / isa.A0)
    return isa.P0 * ((1.0 + 0.2 * (np.asarray(cas) / isa.A0) ** 2) ** 3.5 - 1.0)


def cas_from_qc(qc):
    """Calibrated airspeed (m/s) from impact pressure (Pa)."""
    return isa.A0 * np.sqrt(5.0 * ((np.asarray(qc) / isa.P0 + 1.0) ** (2.0 / 7.0) - 1.0))


def cas_to_tas(cas, h):
    """Calibrated airspeed (m/s) to true airspeed (m/s) at altitude h (m), ISA."""
    p = isa.pressure(h)
    qc = qc_from_cas(cas)
    mach = mach_from_qc(qc, p)
    return tas_from_mach(mach, h)


def tas_to_cas(tas, h):
    """True airspeed (m/s) to calibrated airspeed (m/s) at altitude h (m), ISA."""
    p = isa.pressure(h)
    mach = mach_from_tas(tas, h)
    qc = qc_from_mach(mach, p)
    return cas_from_qc(qc)


def cas_from_mach(mach, h):
    """Calibrated airspeed (m/s) for a given Mach at altitude h (m), ISA."""
    p = isa.pressure(h)
    return cas_from_qc(qc_from_mach(mach, p))


def mach_from_cas(cas, h):
    """Mach for a given calibrated airspeed (m/s) at altitude h (m), ISA."""
    p = isa.pressure(h)
    return mach_from_qc(qc_from_cas(cas), p)


def tas_to_eas(tas, h):
    """True airspeed (m/s) to equivalent airspeed (m/s) at altitude h (m), ISA."""
    return np.asarray(tas) * np.sqrt(isa.density(h) / isa.RHO0)


def eas_to_tas(eas, h):
    """Equivalent airspeed (m/s) to true airspeed (m/s) at altitude h (m), ISA."""
    return np.asarray(eas) * np.sqrt(isa.RHO0 / isa.density(h))
