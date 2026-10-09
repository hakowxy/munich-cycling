"""Exports docs/data.json for the dashboard: daily counts, hourly profiles and the weather model.

Run from the repo root:  python scripts/build_dashboard_data.py
"""
import glob
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = {"zaehlstelle": "station", "gesamt": "total", "min-temp": "temp_min", "max-temp": "temp_max",
           "niederschlag": "rain_mm", "bewoelkung": "cloud_pct", "sonnenstunden": "sun_hours"}

# Keep these in sync with notebooks/analysis.ipynb (section 4).
SCHOOL_HOLIDAYS = [("2025-01-01", "2025-01-03"), ("2025-03-03", "2025-03-07"), ("2025-04-14", "2025-04-25"),
                   ("2025-06-10", "2025-06-20"), ("2025-08-01", "2025-09-15"), ("2025-11-03", "2025-11-07"),
                   ("2025-11-19", "2025-11-19"), ("2025-12-22", "2025-12-31")]
EVENT_DAYS = pd.to_datetime(["2025-04-26", "2025-04-27", "2025-07-16", "2025-07-19", "2025-10-12"])
FORMULA = ("log_total ~ C(station) * off_day + C(weekday) + holiday + school_holiday + christmas"
           " + temp_max + I(temp_max ** 2) + np.log1p(rain_mm) + sun_hours")

daily = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in sorted(glob.glob(str(ROOT / "data/raw/rad_2025_*_tage.csv")))])
daily = daily.rename(columns=COLUMNS)
daily["date"] = pd.to_datetime(daily["datum"], format="%Y.%m.%d")

public_holidays = set(holidays.Germany(subdiv="BY", years=2025)) | {pd.Timestamp("2025-08-15").date()}
daily["holiday"] = daily["date"].dt.date.isin(public_holidays)
daily["school_holiday"] = False
for start, end in SCHOOL_HOLIDAYS:
    daily.loc[daily["date"].between(start, end), "school_holiday"] = True
daily["christmas"] = daily["date"].between("2025-12-24", "2025-12-31") | daily["date"].between("2025-01-01", "2025-01-06")
daily["weekday"] = daily["date"].dt.day_name().str[:3]
daily["off_day"] = (daily["date"].dt.dayofweek >= 5) | daily["holiday"]
daily["log_total"] = np.log(daily["total"])
daily["event"] = (daily["station"] == "Olympia") & daily["date"].isin(EVENT_DAYS)

model = smf.ols(FORMULA, data=daily[(daily["station"] != "Kreuther") & ~daily["event"]]).fit()

# A few reference predictions so the dashboard's JavaScript can be checked against statsmodels.
checks = daily[daily["station"] != "Kreuther"].sample(5, random_state=1)
reference = [{"station": r.station, "weekday": r.weekday, "holiday": bool(r.holiday),
              "school_holiday": bool(r.school_holiday), "christmas": bool(r.christmas),
              "temp_max": r.temp_max, "rain_mm": r.rain_mm, "sun_hours": r.sun_hours,
              "expected": float(np.exp(model.predict(checks.loc[[i]]).iloc[0]))}
             for i, r in checks.iterrows()]

quarter = []
for month in ["06", "12"]:
    q = pd.read_csv(ROOT / f"data/raw/rad_2025_{month}_15min.csv", encoding="utf-8-sig")
    q["ts"] = pd.to_datetime(q["datum"] + " " + q["uhrzeit_start"], format="%Y.%m.%d %H:%M")
    quarter.append(q)
quarter = pd.concat(quarter)
quarter["hour"] = quarter["ts"].dt.hour
quarter["day_type"] = np.where(quarter["ts"].dt.dayofweek >= 5, "weekend", "weekday")
quarter["month"] = quarter["ts"].dt.month.map({6: "June", 12: "December"})
days = quarter.groupby(["zaehlstelle", "month", "day_type"])["ts"].apply(lambda s: s.dt.normalize().nunique())
hourly = quarter.groupby(["zaehlstelle", "month", "day_type", "hour"])["gesamt"].sum()
hourly = (hourly / days.reindex(hourly.index.droplevel("hour")).values).round(1)
profiles = {}
for (station, month, day_type), series in hourly.groupby(level=[0, 1, 2]):
    profiles.setdefault(station, {}).setdefault(month, {})[day_type] = series.tolist()

out = {
    "stations": ["Erhardt", "Olympia", "Hirsch", "Arnulf", "Kreuther"],
    "daily": {
        "date": daily["date"].dt.strftime("%Y-%m-%d").tolist(),
        "station": daily["station"].tolist(),
        "total": daily["total"].tolist(),
        "temp_max": daily["temp_max"].tolist(),
        "rain_mm": daily["rain_mm"].tolist(),
        "sun_hours": daily["sun_hours"].tolist(),
    },
    "profiles": profiles,
    "model": {"params": model.params.round(6).to_dict(), "r2": round(model.rsquared, 3), "reference": reference},
}
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/data.json").write_text(json.dumps(out, separators=(",", ":")))
print(f"docs/data.json written; model R² = {model.rsquared:.3f}")
