import numpy as np
import pandas as pd
from datetime import datetime

def generate_lab_turnaround_dataset(
    n_rows: int = 1_000_000,
    seed: int = 42,
    output_csv: str = "synthetic_lab_turnaround.csv",
    sample_csv: str = "synthetic_lab_turnaround_sample_50k.csv",
) -> pd.DataFrame:
    """
    Generate a large synthetic laboratory operations dataset for regression.

    Target: turnaround_time_minutes

    Notes:
    - This is synthetic (no real patient data).
    - Designed to reflect plausible lab workflow dynamics:
      workload/queueing, priority handling, batching, instrument capacity,
      staffing by shift, weekend impact, contamination/redo events, etc.
    """
    rng = np.random.default_rng(seed)

    # --- Categorical dimensions (tunable) ---
    test_types = np.array([
        "Blood_Culture",
        "Urine_Culture",
        "PCR_Respiratory",
        "PCR_GI",
        "Serology",
        "Biochemistry_Panel",
        "Hematology_FBC",
        "MRSA_Screen",
        "Wound_Swab",
        "Stool_Culture",
    ])

    specimen_types = np.array([
        "blood", "urine", "swab", "stool", "sputum", "tissue"
    ])

    priorities = np.array(["routine", "urgent", "stat"])
    priority_probs = np.array([0.78, 0.18, 0.04])  # most routine

    locations = np.array([
        "GP", "ED", "Inpatient", "ICU", "Outpatient", "Theatre"
    ])
    location_probs = np.array([0.28, 0.22, 0.26, 0.08, 0.12, 0.04])

    instruments = np.array([
        "Analyzer_A", "Analyzer_B", "PCR_Rig_1", "PCR_Rig_2", "Culture_Incubator"
    ])

    # --- Generate base columns ---
    test_type = rng.choice(test_types, size=n_rows, replace=True)
    specimen_type = rng.choice(specimen_types, size=n_rows, replace=True)

    priority = rng.choice(priorities, size=n_rows, replace=True, p=priority_probs)
    location = rng.choice(locations, size=n_rows, replace=True, p=location_probs)

    # day_of_week: 0=Mon ... 6=Sun
    day_of_week = rng.integers(0, 7, size=n_rows)

    # hour_received: 0-23
    hour_received = rng.integers(0, 24, size=n_rows)

    # shift: day/evening/night (roughly aligned to hours)
    shift = np.where((hour_received >= 7) & (hour_received <= 15), "day",
            np.where((hour_received >= 16) & (hour_received <= 22), "evening", "night"))

    # staffing level: lower at night, slightly lower weekend
    weekend = (day_of_week >= 5).astype(int)
    base_staff = np.where(shift == "day", rng.integers(6, 13, size=n_rows),
                 np.where(shift == "evening", rng.integers(4, 9, size=n_rows),
                          rng.integers(2, 6, size=n_rows)))
    staff_on_shift = np.clip(base_staff - weekend, 1, None)

    # samples_in_queue: workload proxy (heavier during day/evening)
    queue_lambda = np.where(shift == "day", 55, np.where(shift == "evening", 40, 25))
    samples_in_queue = rng.poisson(lam=queue_lambda, size=n_rows)

    # batching: some tests are batchy (PCR, serology); cultures less so
    batch_size = rng.integers(1, 40, size=n_rows)
    is_batch_test = np.isin(test_type, ["PCR_Respiratory", "PCR_GI", "Serology", "MRSA_Screen"]).astype(int)

    # instrument choice based on test type (simple mapping)
    instrument = np.empty(n_rows, dtype=object)
    instrument[test_type == "Biochemistry_Panel"] = "Analyzer_A"
    instrument[test_type == "Hematology_FBC"] = "Analyzer_B"
    instrument[np.isin(test_type, ["PCR_Respiratory", "PCR_GI", "MRSA_Screen"])] = rng.choice(
        ["PCR_Rig_1", "PCR_Rig_2"], size=np.sum(np.isin(test_type, ["PCR_Respiratory", "PCR_GI", "MRSA_Screen"])))
    instrument[np.isin(test_type, ["Blood_Culture", "Urine_Culture", "Stool_Culture", "Wound_Swab"])] = "Culture_Incubator"
    instrument[instrument == None] = rng.choice(instruments, size=np.sum(instrument == None), replace=True)

    # instrument capacity: PCR rigs limited capacity; incubator is slow but parallel; analyzers moderate
    instrument_capacity = np.where(np.isin(instrument, ["PCR_Rig_1", "PCR_Rig_2"]), 24,
                          np.where(instrument == "Culture_Incubator", 200,
                                   80))

    # sample quality issues and rework (rare but real)
    contamination_flag = (rng.random(n_rows) < 0.012).astype(int)  # 1.2%
    insufficient_sample_flag = (rng.random(n_rows) < 0.007).astype(int)  # 0.7%
    rerun_flag = ((contamination_flag == 1) | (insufficient_sample_flag == 1)).astype(int)

    # downtime events (rare)
    instrument_downtime_flag = (rng.random(n_rows) < 0.01).astype(int)

    # --- Base turnaround time by test type (minutes) ---
    # (Rough, plausible: cultures longer; hematology/biochem shorter)
    base_tat_map = {
        "Hematology_FBC": 60,
        "Biochemistry_Panel": 90,
        "Serology": 240,
        "MRSA_Screen": 180,
        "PCR_Respiratory": 210,
        "PCR_GI": 240,
        "Wound_Swab": 1440,        # 1 day
        "Stool_Culture": 2160,     # 1.5 days
        "Urine_Culture": 2880,     # 2 days
        "Blood_Culture": 4320,     # 3 days
    }
    base_tat = np.vectorize(base_tat_map.get)(test_type).astype(float)

    # --- Priority effect (urgent/stat get shorter, but not miracles for cultures) ---
    priority_multiplier = np.where(priority == "routine", 1.00,
                          np.where(priority == "urgent", 0.80, 0.65))
    # Limit how much priority can reduce culture-based tests
    is_culture_like = np.isin(test_type, ["Blood_Culture", "Urine_Culture", "Stool_Culture", "Wound_Swab"]).astype(int)
    priority_multiplier = np.where(is_culture_like == 1, np.maximum(priority_multiplier, 0.90), priority_multiplier)

    # --- Queue/workload effect: more queue => longer ---
    queue_effect = 1.0 + (samples_in_queue / 250.0)  # e.g. 50 queue => +20%

    # --- Staffing effect: more staff => shorter ---
    staff_effect = 1.0 - np.clip((staff_on_shift - 4) * 0.03, -0.15, 0.25)
    # staff_effect range approx [0.75..1.15]

    # --- Shift effect: nights slower (handover, fewer staff) ---
    shift_effect = np.where(shift == "day", 1.00, np.where(shift == "evening", 1.08, 1.18))

    # --- Weekend effect (slower for many non-urgent processes) ---
    weekend_effect = 1.0 + weekend * 0.12
    weekend_effect = np.where(priority == "stat", 1.0 + weekend * 0.04, weekend_effect)

    # --- Batch effect: batch tests may wait to fill a run; larger batch reduces per-sample time but may add wait ---
    # Add "batch wait" for batch tests: if batch_size small, wait longer (they're waiting to fill run)
    batch_wait = np.where(is_batch_test == 1, (30.0 * (1.0 - np.clip(batch_size / 30.0, 0, 1))), 0.0)
    batch_processing_efficiency = np.where(is_batch_test == 1, 1.0 - np.clip((batch_size - 1) * 0.004, 0, 0.12), 1.0)

    # --- Instrument capacity: if queue is high relative to capacity, slower ---
    capacity_pressure = 1.0 + np.clip((samples_in_queue - instrument_capacity) / 400.0, 0, 0.35)

    # --- Quality issue rework: adds extra time ---
    rework_minutes = rerun_flag * rng.integers(60, 360, size=n_rows)  # 1h to 6h

    # --- Downtime: adds delay ---
    downtime_minutes = instrument_downtime_flag * rng.integers(30, 240, size=n_rows)

    # --- Random noise: lognormal so it's positively skewed like real waiting times ---
    noise = rng.lognormal(mean=0.0, sigma=0.25, size=n_rows)  # around ~1.0 multiplier

    turnaround_time = (
        base_tat
        * priority_multiplier
        * queue_effect
        * staff_effect
        * shift_effect
        * weekend_effect
        * batch_processing_efficiency
        * capacity_pressure
        * noise
        + batch_wait
        + rework_minutes
        + downtime_minutes
    )

    # Add a few big outliers (rare extreme delays)
    outlier_flag = (rng.random(n_rows) < 0.003).astype(int)  # 0.3%
    turnaround_time += outlier_flag * rng.integers(600, 7200, size=n_rows)  # +10h to +5d

    # Clip to sensible max (e.g., 14 days)
    turnaround_time = np.clip(turnaround_time, 10, 14 * 24 * 60).astype(int)

    df = pd.DataFrame({
        "test_type": test_type,
        "specimen_type": specimen_type,
        "priority": priority,
        "location": location,
        "day_of_week": day_of_week,
        "hour_received": hour_received,
        "shift": shift,
        "staff_on_shift": staff_on_shift,
        "samples_in_queue": samples_in_queue,
        "batch_size": batch_size,
        "is_batch_test": is_batch_test,
        "instrument": instrument,
        "instrument_capacity": instrument_capacity,
        "contamination_flag": contamination_flag,
        "insufficient_sample_flag": insufficient_sample_flag,
        "instrument_downtime_flag": instrument_downtime_flag,
        "turnaround_time_minutes": turnaround_time,
    })

    # Save full dataset
    df.to_csv(output_csv, index=False)

    # Save a smaller sample for quick experiments
    sample = df.sample(n=min(50_000, n_rows), random_state=seed)
    sample.to_csv(sample_csv, index=False)

    print(f"✅ Generated {n_rows:,} rows")
    print(f"Saved full dataset: {output_csv}")
    print(f"Saved sample dataset: {sample_csv}")
    print("\nColumns:", ", ".join(df.columns))

    return df


if __name__ == "__main__":
    generate_lab_turnaround_dataset(
        n_rows=1_000_000,  # change to 2_000_000+ if you want even bigger
        seed=42
    )
