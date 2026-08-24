"""ISA model validation against published reference values.

Reference points and their citations live in aerowatch/reference_data.py.
"""

import numpy as np
import pytest

from aerowatch import isa
from aerowatch.reference_data import ISA_GEOMETRIC_POINTS, ISA_LAYER_POINTS


@pytest.mark.parametrize("h,T_ref,p_ref,rho_ref,tol_T,rel", ISA_LAYER_POINTS)
def test_layer_base_points(h, T_ref, p_ref, rho_ref, tol_T, rel):
    T, p, rho, _ = isa.atmosphere(h)
    assert abs(T - T_ref) <= tol_T
    assert abs(p - p_ref) / p_ref <= rel
    assert abs(rho - rho_ref) / rho_ref <= rel


@pytest.mark.parametrize("z,T_ref,p_ref,rel_rho_ref,tol_T,rel", ISA_GEOMETRIC_POINTS)
def test_geometric_table_points(z, T_ref, p_ref, rel_rho_ref, tol_T, rel):
    h = isa.geometric_to_geopotential(z)
    T, p, rho, _ = isa.atmosphere(h)
    assert abs(T - T_ref) <= tol_T
    assert abs(p - p_ref) / p_ref <= rel
    assert abs(rho / isa.RHO0 - rel_rho_ref) / rel_rho_ref <= rel


def test_sea_level_definitions():
    T, p, rho, a = isa.atmosphere(0.0)
    assert T == pytest.approx(288.15, abs=1e-12)
    assert p == pytest.approx(101325.0, abs=1e-9)
    assert rho == pytest.approx(1.225, rel=5e-5)
    # a0 = sqrt(1.4 * 287.05287 * 288.15) = 340.294 m/s
    assert a == pytest.approx(340.294, abs=1e-3)


def test_tropopause_continuity():
    eps = 1e-6
    below = isa.pressure(isa.H_TROP - eps)
    above = isa.pressure(isa.H_TROP + eps)
    assert below == pytest.approx(above, rel=1e-9)
    assert isa.temperature(isa.H_TROP - eps) == pytest.approx(
        isa.temperature(isa.H_TROP + eps), abs=1e-4
    )


def test_pressure_strictly_decreasing():
    h = np.linspace(-600.0, 20000.0, 500)
    p = isa.pressure(h)
    assert np.all(np.diff(p) < 0)


def test_density_strictly_decreasing():
    h = np.linspace(-600.0, 20000.0, 500)
    rho = isa.density(h)
    assert np.all(np.diff(rho) < 0)


def test_geometric_geopotential_round_trip():
    z = np.array([0.0, 1000.0, 11000.0, 20000.0])
    back = isa.geopotential_to_geometric(isa.geometric_to_geopotential(z))
    assert np.allclose(back, z, atol=1e-6)


def test_geopotential_below_geometric():
    assert isa.geometric_to_geopotential(10000.0) < 10000.0


def test_out_of_range_low_raises():
    with pytest.raises(ValueError):
        isa.temperature(-1000.0)


def test_out_of_range_high_raises():
    with pytest.raises(ValueError):
        isa.pressure(20001.0)


def test_scalar_in_scalar_out():
    assert isinstance(isa.temperature(5000.0), float)
    assert isinstance(isa.pressure(5000.0), float)


def test_array_in_array_out():
    h = np.array([0.0, 5000.0, 15000.0])
    p = isa.pressure(h)
    assert isinstance(p, np.ndarray)
    assert p.shape == h.shape


def test_isothermal_layer_temperature():
    h = np.linspace(11000.0, 20000.0, 50)
    T = isa.temperature(h)
    assert np.allclose(T, 216.65, atol=1e-9)
