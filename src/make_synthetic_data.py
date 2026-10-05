"""Generate a synthetic dataset with the same schema as the study data.

No real participant records are used or reproduced. Values are drawn from a
hand-specified parametric model whose parameters are rounded, population-level
approximations (means, spreads, prevalences and a few deliberate associations).
The file exists so that the analysis pipeline can be run end to end in public.

Usage:  python src/make_synthetic_data.py --n 121 --seed 2026 --out data/synthetic_study_data.csv
"""
import argparse
import numpy as np
import pandas as pd


def generate(n=121, seed=2026):
    rng = np.random.default_rng(seed)
    z = lambda: rng.standard_normal(n)

    male = rng.random(n) < 0.33
    age = np.clip(np.round(39 + 11.5 * z()), 19, 75).astype(int)
    age_z = (age - 39) / 11.5

    height = np.where(male, 172 + 7 * z(), 161 + 6.5 * z()).round(0)
    bmi = np.exp(np.log(25) + 0.22 * z() + 0.05 * age_z)
    weight = (bmi * (height / 100) ** 2).round(1)
    bmi = weight / (height / 100) ** 2
    bmi_z = (bmi - 26) / 6

    # glycaemia: shared latent signal + marker-specific error (weak HbA1c-GA coupling)
    g = z() + 0.15 * age_z
    dm = rng.random(n) < 0.08
    hba1c = 5.5 + 0.35 * g + np.where(dm, rng.gamma(2.0, 1.0, n), 0) + 0.15 * z()
    hba1c = np.clip(hba1c, 4.2, 13).round(2)

    alb = np.clip(np.round(36.5 + 3 * z()), 26, 48)
    ga_pct = 11.8 + 0.6 * g + 0.25 * (alb - 36.5) + np.where(dm, rng.gamma(2.0, 1.5, n), 0) + 1.6 * z()
    ga_pct = np.clip(ga_pct, 6, 35)
    ga_gl = (ga_pct * alb / 100).round(2)

    # lipids: AIP rises with age and BMI; TG derived from HDL and AIP
    hdl = np.clip(np.exp(np.log(0.95) + 0.3 * z() - 0.04 * male), 0.35, 2.5).round(2)
    aip = -0.02 + 0.10 * age_z + 0.08 * bmi_z + 0.30 * z()
    tg = np.clip(hdl * 10 ** aip, 0.3, 6.5).round(2)
    tc = np.clip(4.0 + 0.25 * age_z + 0.85 * z(), 2.3, 7.5).round(2)
    ldl = np.clip(tc - hdl - tg / 2.2, 0.6, None).round(2)

    # blood pressure and treatment
    sbp = np.round(128 + 5 * age_z + 4 * bmi_z + 14 * z()).astype(int)
    dbp = np.round(0.45 * sbp + 24 + 8 * z()).astype(int)
    p_tx = 1 / (1 + np.exp(-(-1.6 + 0.8 * age_z + 0.4 * bmi_z)))
    on_tx = rng.random(n) < p_tx
    meds_pool = ["Nifedipine", "Amlodipine", "HCT", "Nifedipine, HCT", "Amlodipine, Losartan",
                 "HCT, Nifedipine, Enalapril"]
    meds = np.where(on_tx, rng.choice(meds_pool, n), "no")
    other = rng.choice(["Asthma", "Arthritis", "Ulcers"], n)
    has_other = rng.random(n) < 0.05
    cond = np.where(on_tx, "HPT", np.where(has_other, other, "No"))
    chronic = np.where(on_tx | has_other, "Yes", "No")

    def pick(levels, probs):
        return rng.choice(levels, n, p=np.array(probs) / sum(probs))

    vl = pick(["<20", "0", "30", "40", "129", "1200"], [55, 30, 5, 4, 4, 2])
    smoker = rng.random(n) < 0.07
    drinker = rng.random(n) < 0.26

    df = pd.DataFrame({
        "Study ID": [f"SYN-{i:03d}" for i in range(1, n + 1)],
        "Visit No.": pick([2, 3], [1, 3]),
        "Age (yrs)": age,
        "Sex": np.where(male, "Male", "Female"),
        "Marital Status": pick(["Married", "Single", "Divorced", "Widowed"], [54, 33, 7, 6]),
        "Occupation": pick(["vendor", "teacher", "unemployed", "self employed", "security guard",
                            "housewife", "driver", "hairdresser", "retired", "administrator"],
                           [16, 8, 20, 12, 6, 10, 6, 6, 6, 10]),
        "Education Level": pick(["Primary", "Secondary", "Tertiary"], [11, 71, 18]),
        "Avg Income": np.where(rng.random(n) < 0.07, np.nan,
                               np.round(np.clip(rng.gamma(1.6, 140, n), 0, 1000), -1)),
        "ART Regimen": "TLD",
        "ART Duration (months)": ">6",
        "Weight (kg)": weight,
        "Height (cm)": height,
        "BP Systolic (mmHg)": sbp,
        "BP Diastolic (mmHg)": dbp,
        "Waist Circ. (cm)": np.where(rng.random(n) < 0.68, np.nan, np.round(80 + 12 * z())),
        "Smoker (Y/N)": np.where(smoker, "Yes", "No"),
        "Cigarettes / Day": np.where(smoker, rng.integers(2, 15, n), 0),
        "Drinker (Y/N)": np.where(drinker, "Yes", "No"),
        "Pints / Day": np.where(drinker, rng.integers(1, 6, n), 0),
        "Illicit Drug Use (Y/N)": np.where(rng.random(n) < 0.03, "Yes", "No"),
        "Food (Last 24 hrs)": "Not recorded (synthetic)",
        "Exercise Program (Y/N)": pick(["No", "Yes"], [58, 42]),
        "Exercise Frequency/week": pd.Series(pick(["Never", "Once", "Twice", "3 times", "More than 3 times", None],
                                                  [37, 11, 6, 4, 18, 24])),
        "Chronic Disease (Y/N)": chronic,
        "Chronic Condition(s)": cond,
        "Chronic Disease Meds": meds,
        "Herbal Use (Y/N)": np.where(rng.random(n) < 0.03, "Yes", "No"),
        "Herb Name": "",
        "Sleep (hrs/day)": np.where(rng.random(n) < 0.2, np.nan, np.clip(np.round(7.8 + 1.4 * z()), 4, 12)),
        "Stress Level": np.where(rng.random(n) < 0.2, np.nan, np.clip(np.round(rng.gamma(1.3, 2.5, n)), 1, 10)),
        "Viral Load (copies/mL)": vl,
        "CD4 at ART Init (cells/µL)": np.where(rng.random(n) < 0.7, np.nan, np.round(rng.gamma(1.8, 110, n))),
        "Sample Type": "Whole Blood & Serum",
        "HbA1c (%)": hba1c,
        "Glyc. Albumin (GA) g/L": ga_gl,
        "TC (mmol/L)": tc,
        "HDL (mmol/L)": hdl,
        "TG (mmol/L)": tg,
        "LDL (mmol/L)": ldl,
        "Albumin (ALB) g/L": alb.astype(int),
    })
    df["Exercise Frequency/week"] = df["Exercise Frequency/week"].replace({"None": np.nan})
    df["GA % (GA/ALB x100)"] = (df["Glyc. Albumin (GA) g/L"] / df["Albumin (ALB) g/L"] * 100).round(4)
    df["GA / HbA1c Ratio"] = (df["GA % (GA/ALB x100)"] / df["HbA1c (%)"]).round(4)
    df["AIP (log10 TG/HDL)"] = np.log10(df["TG (mmol/L)"] / df["HDL (mmol/L)"]).round(4)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=121)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/synthetic_study_data.csv")
    a = ap.parse_args()
    d = generate(a.n, a.seed)
    d.to_csv(a.out, index=False)
    print(f"wrote {len(d)} synthetic records to {a.out}")
