"""
Barometric pressure depth-adjustment logic for the AquaTroll pipeline.

Compares each site's own logged pressure reading (from the AquaTroll's
onboard barometric/pressure channel, already present in the per-site data
from process_site_file() in dataValidation.py) against the BOM weather
station reading (BomBaro, merged in by dataValidation.py from
get_bom_baro_data()), and applies the AquaTroll drift-correction formula
when the two agree within a sane range.

This module is only imported/called by dataValidation.py's
consolidate_csv_files() when the PRESSURE_ADJUSTMENT environment variable
is set to 'true'. get_bom_baro_data() and the BoM merge itself stay in
dataValidation.py and run unconditionally, since BomBaro/Rainfall also
feed rainfall reporting and testsAB.py's pressure-anomaly checks.

When PRESSURE_ADJUSTMENT is unset or 'false', dataValidation.py never
calls calculate_adjusted_depth(), and the Depth(m)adjusted column is
omitted from every output file (verification CSV, SQL CSV) rather than
existing with placeholder values.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('data_validation')

# --- Constants ---
CONVERSION_FACTOR = 0.70307     # psi -> metres of water column
HPA_TO_PSI = 0.0145038
PSI_TO_HPA = 1 / HPA_TO_PSI      # derived, not independently maintained

# Drift thresholds (hPa), matching the AquaTroll correction guidance:
MIN_DEPTH_THRESHOLD = 0.3        # m   - shallow water, skip adjustment
MIN_BARO_DIFF_THRESHOLD = 5.0    # hPa - sensor noise, skip adjustment
MAX_BARO_DIFF_THRESHOLD = 20.0   # hPa - upper bound of the "apply correction" band
EXCESSIVE_DRIFT_THRESHOLD = 50.0  # hPa - sensor failure, skip adjustment


def calculate_adjusted_depth(df: pd.DataFrame, water_density: float = 1000.0,
                              reference_density: float = 1000.0) -> pd.DataFrame:
    """
    Calculate adjusted depth using AquaTrolls drift correction formula.

    THRESHOLDS:
    - Drift <= 5 hPa: No adjustment (sensor noise)
    - 5 < Drift <= 20 hPa: Apply correction
    - 20 < Drift <= 50 hPa: No adjustment, flag excessive drift
    - Drift > 50 hPa: No adjustment (sensor failure)
    - Depth <= 0.3m: No adjustment (shallow water)

    Expects df to already have 'Depth(m)raw' and 'BomBaro' columns
    (populated by dataValidation.py before this is called).
    """
    df = df.copy()
    df['Depth(m)adjusted'] = np.nan

    # Calculate pressure difference
    def get_pressure_diff(row):
        if pd.notna(row.get('Pressure(RAW)[Main Buffer] (PSI)')):
            bom_psi = row['BomBaro'] * HPA_TO_PSI
            logger_psi = row['Pressure(RAW)[Main Buffer] (PSI)']
            return abs(bom_psi - logger_psi) / HPA_TO_PSI
        elif pd.notna(row.get('Barometric Pressure(RAW)[Main Buffer] (hPa)')):
            return abs(row['BomBaro'] - row['Barometric Pressure(RAW)[Main Buffer] (hPa)'])
        return np.nan

    df['pressure_diff_hpa'] = df.apply(get_pressure_diff, axis=1)

    # Valid adjustment mask (5 < diff <= 20 hPa, depth > 0.3m)
    valid_mask = (
        df['Depth(m)raw'].notna() &
        (df['Depth(m)raw'] > MIN_DEPTH_THRESHOLD) &
        df['BomBaro'].notna() &
        df['pressure_diff_hpa'].notna() &
        (df['pressure_diff_hpa'] > MIN_BARO_DIFF_THRESHOLD) &
        (df['pressure_diff_hpa'] <= MAX_BARO_DIFF_THRESHOLD)
    )

    # Apply adjustment
    if valid_mask.any():
        valid_df = df[valid_mask]

        if 'Pressure(RAW)[Main Buffer] (PSI)' in valid_df.columns:
            has_psi = valid_df['Pressure(RAW)[Main Buffer] (PSI)'].notna()
        else:
            has_psi = pd.Series(False, index=valid_df.index)

        pressure_diff_psi = pd.Series(np.nan, index=valid_df.index)

        if has_psi.any():
            bom_psi = valid_df.loc[has_psi, 'BomBaro'] * HPA_TO_PSI
            logger_psi = valid_df.loc[has_psi, 'Pressure(RAW)[Main Buffer] (PSI)']
            pressure_diff_psi.loc[has_psi] = bom_psi - logger_psi

        if (~has_psi).any():
            bom_hpa = valid_df.loc[~has_psi, 'BomBaro']
            logger_hpa = valid_df.loc[~has_psi, 'Barometric Pressure(RAW)[Main Buffer] (hPa)']
            pressure_diff_psi.loc[~has_psi] = (bom_hpa - logger_hpa) * HPA_TO_PSI

        specific_gravity = water_density / reference_density
        depth_correction = (CONVERSION_FACTOR * pressure_diff_psi) / specific_gravity
        adjusted_depth = valid_df['Depth(m)raw'] + depth_correction

        df.loc[valid_mask, 'Depth(m)adjusted'] = adjusted_depth
        logger.info(f"Calculated adjusted depth for {valid_mask.sum()} rows")

        # Set negative values to NaN
        negative_mask = (df['Depth(m)adjusted'] < 0) & df['Depth(m)adjusted'].notna()
        if negative_mask.any():
            logger.warning(f"Set {negative_mask.sum()} negative adjusted depths to NaN")
            df.loc[negative_mask, 'Depth(m)adjusted'] = np.nan

    # For non-adjusted rows, copy raw to adjusted with appropriate comments
    df['OTHER - Comments - Text'] = df.get('OTHER - Comments - Text', '').fillna('')

    # Define skip conditions
    shallow = (df['Depth(m)raw'].notna()) & (df['Depth(m)raw'] > 0) & (df['Depth(m)raw'] <= MIN_DEPTH_THRESHOLD)
    large_diff = (df['pressure_diff_hpa'] > EXCESSIVE_DRIFT_THRESHOLD) & df['pressure_diff_hpa'].notna()
    small_diff = (df['pressure_diff_hpa'] <= MIN_BARO_DIFF_THRESHOLD) & df['pressure_diff_hpa'].notna()
    excessive_drift = (
        (df['Depth(m)raw'] > MIN_DEPTH_THRESHOLD) &
        (df['pressure_diff_hpa'] > MAX_BARO_DIFF_THRESHOLD) &
        (df['pressure_diff_hpa'] <= EXCESSIVE_DRIFT_THRESHOLD)
    )
    no_bom = (df['Depth(m)raw'] > 0) & df['BomBaro'].isna()
    no_logger_baro = (
        (df['Depth(m)raw'] > 0) &
        df['Barometric Pressure(RAW)[Main Buffer] (hPa)'].isna() &
        df.get('Pressure(RAW)[Main Buffer] (PSI)', pd.Series([np.nan] * len(df))).isna()
    )

    skip_conditions = [
        (shallow, "Very shallow depth: no adjustment applied"),
        (large_diff, "Large barometric difference observed: no adjustment applied"),
        (small_diff, "Small barometric difference: no adjustment applied (possible sensor noise)"),
        (excessive_drift, "Large sensor drift detected (>20 hPa): adjusted value formula not applied"),
        (no_bom, "No weather station data: Adjustments can not be applied"),
        (no_logger_baro, "No AquaTroll pressure data. Adjustments can not be applied")
    ]

    for condition, comment in skip_conditions:
        apply_mask = condition & (df['OTHER - Comments - Text'].str.strip() == '')
        if apply_mask.any():
            df.loc[apply_mask, 'Depth(m)adjusted'] = df.loc[apply_mask, 'Depth(m)raw']
            df.loc[apply_mask, 'OTHER - Comments - Text'] = comment
            logger.info(f"Applied '{comment}' to {apply_mask.sum()} rows")

    return df