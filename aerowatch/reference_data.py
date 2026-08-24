"""Published reference values used to validate the ISA implementation.

Two independent published sources:

Source A: ISO 2533:1975 / ICAO Doc 7488 layer base values, as tabulated in
the "International Standard Atmosphere" reference table (layer bases at
geopotential altitude; temperature K, pressure Pa, density kg/m^3):
  0 m:      288.15 K, 101325 Pa, 1.225 kg/m^3
  11000 m:  216.65 K, 22632 Pa,  0.3639 kg/m^3
  20000 m:  216.65 K, 5474.9 Pa, 0.0880 kg/m^3
These are the standard layer-base constants of the model and are quoted to
the precision published in the table.

Source B: Engineering ToolBox, "International Standard Atmosphere" table
(https://www.engineeringtoolbox.com/international-standard-atmosphere-d_985.html).
That table lists elevation z as GEOMETRIC altitude, pressure in bar to 4
significant figures, temperature to 0.1 K, and relative density rho/rho0 to
4 significant figures. The geometric convention is verifiable from the table
itself: at z = 11000 m it lists 216.8 K, which is the ISA temperature at the
corresponding geopotential altitude of 10981 m, not the 216.65 K value at
h = 11000 m geopotential.

Tolerances below reflect the rounding of the published figures plus a small
allowance for constant differences between editions:
  temperature: 0.05 K for Source A, 0.1 K for Source B (rounded to 0.1 K)
  pressure and density: 2e-3 relative (4 significant figure tables)
"""

# Source A: geopotential altitude (m), T (K), p (Pa), rho (kg/m^3), tol_T (K), rel tol
ISA_LAYER_POINTS = [
    (0.0, 288.15, 101325.0, 1.225, 0.05, 5e-4),
    (11000.0, 216.65, 22632.0, 0.3639, 0.05, 5e-4),
    (20000.0, 216.65, 5474.9, 0.0880, 0.05, 2e-3),
]

# Source B: geometric altitude z (m), T (K), p (Pa), rho/rho0, tol_T (K), rel tol
ISA_GEOMETRIC_POINTS = [
    (1000.0, 281.7, 0.8988e5, 0.9075, 0.1, 2e-3),
    (2000.0, 275.2, 0.7950e5, 0.8217, 0.1, 2e-3),
    (3000.0, 268.7, 0.7012e5, 0.7423, 0.1, 2e-3),
    (5000.0, 255.7, 0.5405e5, 0.6012, 0.1, 2e-3),
    (8000.0, 236.2, 0.3565e5, 0.4292, 0.1, 2e-3),
    (10000.0, 223.3, 0.2650e5, 0.3376, 0.1, 2e-3),
    (12000.0, 216.7, 0.1940e5, 0.2546, 0.1, 2e-3),
    (15000.0, 216.7, 0.1211e5, 0.1590, 0.1, 2e-3),
    (18000.0, 216.7, 0.07565e5, 0.09930, 0.1, 2e-3),
    (20000.0, 216.7, 0.05529e5, 0.07258, 0.1, 2e-3),
]
