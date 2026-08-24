"""Compare the ISA implementation against published reference points and
write results/isa_validation.json with the measured deviations."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aerowatch import isa
from aerowatch.reference_data import ISA_GEOMETRIC_POINTS, ISA_LAYER_POINTS


def main():
    rows = []
    for h, T_ref, p_ref, rho_ref, tol_T, rel in ISA_LAYER_POINTS:
        T, p, rho, _ = isa.atmosphere(h)
        rows.append(
            {
                "source": "ISO2533/ICAO7488 layer table",
                "altitude_m": h,
                "altitude_kind": "geopotential",
                "T_err_K": T - T_ref,
                "p_rel_err": (p - p_ref) / p_ref,
                "rho_rel_err": (rho - rho_ref) / rho_ref,
                "tol_T_K": tol_T,
                "rel_tol": rel,
            }
        )
    for z, T_ref, p_ref, rel_rho_ref, tol_T, rel in ISA_GEOMETRIC_POINTS:
        h = isa.geometric_to_geopotential(z)
        T, p, rho, _ = isa.atmosphere(h)
        rows.append(
            {
                "source": "EngineeringToolBox ISA table",
                "altitude_m": z,
                "altitude_kind": "geometric",
                "T_err_K": T - T_ref,
                "p_rel_err": (p - p_ref) / p_ref,
                "rho_rel_err": (rho / isa.RHO0 - rel_rho_ref) / rel_rho_ref,
                "tol_T_K": tol_T,
                "rel_tol": rel,
            }
        )
    summary = {
        "n_points": len(rows),
        "max_abs_T_err_K": max(abs(r["T_err_K"]) for r in rows),
        "max_abs_p_rel_err": max(abs(r["p_rel_err"]) for r in rows),
        "max_abs_rho_rel_err": max(abs(r["rho_rel_err"]) for r in rows),
        "all_within_tolerance": all(
            abs(r["T_err_K"]) <= r["tol_T_K"]
            and abs(r["p_rel_err"]) <= r["rel_tol"]
            and abs(r["rho_rel_err"]) <= r["rel_tol"]
            for r in rows
        ),
        "points": rows,
    }
    out = os.path.join(os.path.dirname(__file__), "..", "results", "isa_validation.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "points"}, indent=2))


if __name__ == "__main__":
    main()
