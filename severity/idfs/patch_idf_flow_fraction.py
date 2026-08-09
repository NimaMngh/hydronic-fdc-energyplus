# -*- coding: utf-8 -*-
"""
patch_idf_flow_fraction.py
==========================
Helper script for the fault-severity parametric sweep.

Patches the stuck-closed flow cap inside the EnergyManagementSystem:Program
block of an EnergyPlus IDF file.

It finds the fault-injection line in FaultInjection_Program:

    SET Act_CoreRetail_CoilIn_MdotMax = <flow>,

and rewrites <flow> as PEAK_FLOW_KG_S * phi, so phi is the fraction of peak
flow the valve still passes. The cap is written as an absolute value rather
than as a runtime multiplier on UH_MdotMax_Observed: the restriction then
stays identical across severity levels instead of tracking whatever peak a
given run happens to observe.

The ELSE branch (`= Null`) is never matched, since the pattern requires a
numeric right-hand side.

Usage
-----
python patch_idf_flow_fraction.py \\
    --template  ../../RetailStandalone_stuckclosed_detect.idf \\
    --phi       0.10 \\
    --output    RetailStandalone_sc_s10_detect.idf

Author : Nima Monghasemi
Date   : March 2026
"""

import argparse
import re
import sys
from pathlib import Path

# Peak hot-water flow through the Core_Retail coil in the fault-free baseline,
# kg/s. The moderate S3 case (phi = 0.30) caps the coil at 0.594050002, which
# fixes this value. Every severity level is PEAK_FLOW_KG_S * phi.
PEAK_FLOW_KG_S = 1.980166674


# ──────────────────────────────────────────────────────────────────────────────
# Core patching function
# ──────────────────────────────────────────────────────────────────────────────

def patch_idf(template_path: Path, phi: float, output_path: Path,
              peak_flow: float = PEAK_FLOW_KG_S) -> None:
    """
    Read *template_path*, reset the flow cap in the FaultInjection_Program
    EMS block to *phi* of peak flow, and write to *output_path*.

    Parameters
    ----------
    template_path : Path to the source IDF file (e.g. *_stuckclosed_detect.idf*)
    phi           : Target flow fraction (0 < phi <= 1).
                    e.g. 0.10 = 10 % of peak flow (severe restriction).
    output_path   : Destination IDF path.
    peak_flow     : Baseline peak coil flow in kg/s.

    Raises
    ------
    ValueError  if the target line is not found in the template.
    """
    if not (0.0 < phi <= 1.0):
        raise ValueError(f"phi must be in (0, 1].  Got: {phi}")

    text = template_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Target line (case-insensitive, tolerates varying whitespace):
    #
    #   SET Act_CoreRetail_CoilIn_MdotMax = 0.594050002,  !- Program Line 2
    #
    # Captured as:
    #   group 1 — everything up to and including the "= "
    #   group 2 — the current flow cap in kg/s
    #   group 3 — the terminator plus any inline comment to end-of-line
    #
    # Requiring digits in group 2 keeps this off the ELSE branch, which
    # assigns Null and must be left alone.
    # ------------------------------------------------------------------ #
    pattern = re.compile(
        r"(SET\s+Act_CoreRetail_CoilIn_MdotMax\s*=\s*)"
        r"([0-9]+(?:\.[0-9]*)?)"
        r"(\s*[,;].*)",   # ERL lines end with a comma, or a semicolon if last
        re.IGNORECASE,
    )

    match = pattern.search(text)
    if match is None:
        raise ValueError(
            "Could not find the fault-injection line:\n"
            "  SET Act_CoreRetail_CoilIn_MdotMax = <flow>,\n"
            f"in template: {template_path}\n\n"
            "Check that the EMS:Program block named 'FaultInjection_Program'\n"
            "uses exactly 'Act_CoreRetail_CoilIn_MdotMax' as the actuator variable."
        )

    # Absolute cap in kg/s, at the same 9-decimal precision as the templates
    phi_str = f"{peak_flow * phi:.9f}"

    new_line = match.group(1) + phi_str + match.group(3)
    new_text = text[: match.start()] + new_line + text[match.end() :]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(new_text, encoding="utf-8")

    old_flow = match.group(2)
    print(f"  Patched: {template_path.name}  ->  {output_path.name}")
    print(f"           flow cap  {old_flow}  ->  {phi_str} kg/s  (phi = {phi:g})")


# ──────────────────────────────────────────────────────────────────────────────
# Batch generation helper (called from generate_severity_idfs.py)
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_sc_idfs(project_root: Path, output_dir: Path) -> None:
    """
    Generate the 8 stuck-closed IDF files required by the severity sweep
    (4 severity levels × 2 scenarios: detect / comp).

    Severity levels
    ---------------
    sc_s10 : phi = 0.10  (10 % of peak flow — most severe restriction)
    sc_s20 : phi = 0.20
    sc_s50 : phi = 0.50
    sc_s70 : phi = 0.70  (70 % — mildest new level; existing S3 = 0.30)

    Note: S3 (phi = 0.30) already exists in the project root as
          RetailStandalone_stuckclosed_detect.idf /
          RetailStandalone_stuckclosed_comp.idf
    """
    SC_LEVELS = [
        ("sc_s10", 0.10),
        ("sc_s20", 0.20),
        ("sc_s50", 0.50),
        ("sc_s70", 0.70),
    ]

    templates = {
        "detect": project_root / "RetailStandalone_stuckclosed_detect.idf",
        "comp":   project_root / "RetailStandalone_stuckclosed_comp.idf",
    }

    for scenario, tmpl in templates.items():
        if not tmpl.exists():
            print(f"  [WARNING] Template not found: {tmpl}  — skipping {scenario}")
            continue
        for tag, phi in SC_LEVELS:
            out_name = f"RetailStandalone_{tag}_{scenario}.idf"
            patch_idf(tmpl, phi, output_dir / out_name)

    print("\nAll stuck-closed IDF variants generated.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Patch the stuck-closed flow fraction in an EnergyPlus IDF."
    )
    p.add_argument(
        "--template", default=None,
        help="Path to the source IDF (e.g. RetailStandalone_stuckclosed_detect.idf). "
             "Required unless --batch is used."
    )
    p.add_argument(
        "--phi", type=float, default=None,
        help="Flow fraction to inject (e.g. 0.10 for 10%% of peak). "
             "Required unless --batch is used."
    )
    p.add_argument(
        "--output", default=None,
        help="Destination IDF path (will be created if needed). "
             "Required unless --batch is used."
    )
    p.add_argument(
        "--batch", action="store_true",
        help="Generate ALL 8 stuck-closed IDF variants automatically. "
             "Uses --project-root and --output-dir; ignores --template/--phi/--output."
    )
    p.add_argument(
        "--project-root", default=".",
        help="Project root directory containing the template IDFs (used with --batch)"
    )
    p.add_argument(
        "--output-dir", default=".",
        help="Directory to write patched IDFs (used with --batch)"
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.batch:
        # --batch mode: no need for --template / --phi / --output
        generate_all_sc_idfs(
            project_root=Path(args.project_root),
            output_dir=Path(args.output_dir),
        )
    else:
        # Single-file mode: validate that the three required args are present
        missing = []
        if args.template is None: missing.append("--template")
        if args.phi      is None: missing.append("--phi")
        if args.output   is None: missing.append("--output")
        if missing:
            import sys
            print(f"error: the following arguments are required in single-file mode: "
                  f"{', '.join(missing)}")
            print("Tip: use --batch to generate all 8 stuck-closed IDF variants at once.")
            sys.exit(2)
        patch_idf(
            template_path=Path(args.template),
            phi=args.phi,
            output_path=Path(args.output),
        )


if __name__ == "__main__":
    main()
