"""
cal_parser.py

Parses an ATI .cal calibration file (XML) and extracts the calibration
matrix, per-axis max ratings, and units.

Per ATI's own instructions inside the .cal file: use the "UserAxis"
elements (not "Axis") to build the calibration matrix -- UserAxis is
already in real engineering units (e.g. N, N-m) and has ATI's internal
scaling/transform baked in already. "Axis" is for ATI's internal use only.
"""

import xml.etree.ElementTree as ET
import numpy as np

AXIS_ORDER = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']


def load_ati_calibration(cal_file_path):
    """
    Returns a dict with:
        matrix       : 6x6 numpy array. wrench = matrix @ raw_voltages
        max_ratings  : dict of {axis_name: max_value} from the file
        force_units  : e.g. "N"
        torque_units : e.g. "N-m"
        serial       : sensor serial number, e.g. "FT75200"
    """
    tree = ET.parse(cal_file_path)
    root = tree.getroot()
    calibration = root.find('Calibration')

    matrix_rows = []
    max_ratings = {}

    for axis_name in AXIS_ORDER:
        user_axis = calibration.find(f"UserAxis[@Name='{axis_name}']")
        if user_axis is None:
            raise ValueError(f"Could not find UserAxis '{axis_name}' in {cal_file_path}")
        values = [float(v) for v in user_axis.get('values').split()]
        if len(values) != 6:
            raise ValueError(f"Expected 6 values for {axis_name}, got {len(values)}")
        matrix_rows.append(values)
        max_ratings[axis_name] = float(user_axis.get('max'))

    return {
        'matrix': np.array(matrix_rows),
        'max_ratings': max_ratings,
        'force_units': calibration.get('ForceUnits'),
        'torque_units': calibration.get('TorqueUnits'),
        'serial': root.get('Serial'),
    }


if __name__ == "__main__":
    # Quick sanity check -- run this file directly to print out what
    # gets parsed from your .cal file before trusting it in a real run.
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "FT75200.cal"
    cal = load_ati_calibration(path)
    print(f"Serial: {cal['serial']}")
    print(f"Units: {cal['force_units']} / {cal['torque_units']}")
    print(f"Max ratings: {cal['max_ratings']}")
    print("Calibration matrix:")
    print(cal['matrix'])