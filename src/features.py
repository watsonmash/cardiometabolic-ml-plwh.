"""Load the study table (real or synthetic) and derive modelling features."""
from pathlib import Path
import numpy as np
import pandas as pd

ANTIHTN = (r"nifedip|nefedip|amlodip|amylopid|enalapril|hct|htc|indapamide|losartan|"
           r"locatine|atenolol|bisoprolol|carvedilol|hydrochloroth")


def load(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:  # study workbook: grouped header row above the column names
        df = pd.read_excel(path, header=1)
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def derive(df):
    out = pd.DataFrame(index=df.index)
    out["age"] = df["Age (yrs)"].astype(float)
    out["male"] = df["Sex"].astype(str).str.startswith("M").astype(int)
    out["bmi"] = df["Weight (kg)"] / (df["Height (cm)"] / 100) ** 2
    out["albumin"] = df["Albumin (ALB) g/L"].astype(float)
    out["hba1c"] = df["HbA1c (%)"].astype(float)
    out["ga_pct"] = df["Glyc. Albumin (GA) g/L"] / df["Albumin (ALB) g/L"] * 100
    out["aip"] = np.log10(df["TG (mmol/L)"] / df["HDL (mmol/L)"])
    out["tg"] = df["TG (mmol/L)"].astype(float)
    out["hdl"] = df["HDL (mmol/L)"].astype(float)
    out["tc"] = df["TC (mmol/L)"].astype(float)
    out["sbp"] = df["BP Systolic (mmHg)"].astype(float)
    out["dbp"] = df["BP Diastolic (mmHg)"].astype(float)
    meds = df["Chronic Disease Meds"].fillna("").astype(str).str.lower()
    on_tx = meds.str.contains(ANTIHTN, regex=True)
    out["htn"] = (((out["sbp"] >= 140) | (out["dbp"] >= 90)) | on_tx).astype(int)
    # signed marker discordance on a common (z) scale: positive = GA higher than HbA1c implies
    z = lambda s: (s - s.mean()) / s.std()
    out["discordance"] = z(out["ga_pct"]) - z(out["hba1c"])
    # physiological plausibility screen (same limits as the dissertation pipeline)
    out["qc_flag"] = ~(out["bmi"].between(12, 60) & out["albumin"].between(15, 55)
                       & out["ga_pct"].between(5, 35) & out["hdl"].between(0.3, 3.0))
    return out
