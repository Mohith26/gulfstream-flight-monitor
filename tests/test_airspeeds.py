"""Airspeed conversion tests against hand-computed fixtures.

The fixture numbers below were computed by hand (independent arithmetic with
the standard formulas, not by calling this package), using the constants
T0 = 288.15 K, P0 = 101325 Pa, rho0 = 1.225 kg/m^3, R = 287.05287 J/(kg K),
gamma = 1.4, g0 = 9.80665 m/s^2, lapse 6.5 K/km, 1 kt = 0.514444 m/s.

Fixture 1: sea level, CAS 250 kt = 128.611 m/s.
  A0 = sqrt(1.4 * 287.05287 * 288.15) = 340.2940 m/s
  qc = P0 ((1 + 0.2 (CAS/A0)^2)^3.5 - 1) = 10498.20 Pa

Fixture 2: h = 10668 m geopotential (FL350), Mach 0.80.
  T = 288.15 - 0.0065 * 10668 = 218.808 K
  p = P0 (T/T0)^5.25588 = 23842.27 Pa
  a = sqrt(1.4 * 287.05287 * 218.808) = 296.5354 m/s
  TAS = 0.8 a = 237.2283 m/s = 461.135 kt
  qc = p ((1 + 0.2 * 0.64)^3.5 - 1) = 12501.46 Pa
  CAS = A0 sqrt(5 ((qc/P0 + 1)^(2/7) - 1)) = 139.8918 m/s = 271.928 kt
  rho = p / (R T) = 0.3795968 kg/m^3
  EAS = TAS sqrt(rho/rho0) = 132.0565 m/s = 256.698 kt

Fixture 3: h = 3048 m (10000 ft), CAS 250 kt.
  p = 69681.64 Pa, T = 268.338 K
  M = sqrt(5 ((qc/p + 1)^(2/7) - 1)) with qc from fixture 1 = 0.452275
  TAS = M sqrt(1.4 R T) = 148.5212 m/s = 288.702 kt
"""

import numpy as np
import pytest

from aerowatch import airspeeds, isa

KT = airspeeds.KT


def test_a0_constant():
    assert isa.A0 == pytest.approx(340.2940, abs=1e-3)
    assert isa.A0 / KT == pytest.approx(661.479, abs=1e-2)


def test_fixture1_qc_from_cas_sea_level():
    qc = airspeeds.qc_from_cas(250.0 * KT)
    assert qc == pytest.approx(10498.20, rel=1e-5)


def test_fixture2_tas_from_mach():
    tas = airspeeds.tas_from_mach(0.80, 10668.0)
    assert tas == pytest.approx(237.2283, rel=1e-5)
    assert tas / KT == pytest.approx(461.135, abs=0.01)


def test_fixture2_cas_from_mach():
    cas = airspeeds.cas_from_mach(0.80, 10668.0)
    assert cas == pytest.approx(139.8918, rel=1e-5)
    assert cas / KT == pytest.approx(271.928, abs=0.01)


def test_fixture2_eas():
    tas = airspeeds.tas_from_mach(0.80, 10668.0)
    eas = airspeeds.tas_to_eas(tas, 10668.0)
    assert eas == pytest.approx(132.0565, rel=1e-5)


def test_fixture3_cas_to_tas_10000ft():
    tas = airspeeds.cas_to_tas(250.0 * KT, 3048.0)
    assert tas == pytest.approx(148.5212, rel=1e-5)
    assert tas / KT == pytest.approx(288.702, abs=0.01)


def test_sea_level_cas_equals_tas():
    for kt in (100.0, 250.0, 340.0):
        assert airspeeds.cas_to_tas(kt * KT, 0.0) == pytest.approx(kt * KT, rel=1e-9)


def test_sea_level_eas_equals_tas():
    # rho0 is defined as exactly 1.225 while p0/(R T0) = 1.2249999, so the
    # sea level density ratio is 1 only to about 1e-7 relative.
    assert airspeeds.tas_to_eas(150.0, 0.0) == pytest.approx(150.0, rel=1e-6)


@pytest.mark.parametrize("h", [0.0, 3048.0, 7620.0, 10668.0])
def test_cas_tas_round_trip(h):
    cas = 250.0 * KT
    back = airspeeds.tas_to_cas(airspeeds.cas_to_tas(cas, h), h)
    assert back == pytest.approx(cas, rel=1e-10)


def test_eas_round_trip():
    tas = 230.0
    h = 9000.0
    assert airspeeds.eas_to_tas(airspeeds.tas_to_eas(tas, h), h) == pytest.approx(
        tas, rel=1e-12
    )


def test_mach_qc_round_trip():
    p = isa.pressure(8000.0)
    qc = airspeeds.qc_from_mach(0.72, p)
    assert airspeeds.mach_from_qc(qc, p) == pytest.approx(0.72, rel=1e-12)


def test_tas_grows_with_altitude_at_fixed_cas():
    cas = 250.0 * KT
    tas = [airspeeds.cas_to_tas(cas, h) for h in (0.0, 3000.0, 6000.0, 9000.0)]
    assert all(b > a for a, b in zip(tas, tas[1:]))


def test_supersonic_raises():
    with pytest.raises(ValueError):
        airspeeds.qc_from_mach(1.05, 101325.0)


def test_supersonic_cas_raises():
    with pytest.raises(ValueError):
        airspeeds.qc_from_cas(1.2 * isa.A0)


def test_vectorized_conversion():
    cas = np.array([200.0, 250.0, 300.0]) * KT
    tas = airspeeds.cas_to_tas(cas, 5000.0)
    assert tas.shape == (3,)
    assert np.all(tas > cas)
