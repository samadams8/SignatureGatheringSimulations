# Signature Gathering Simulations

Written and tested in Python 3.10.12.

## What this script does

Runs a simulation of signature gathering by district using the provided CSV inputs, producing summary outputs for different partisan-lean scenarios.

## Monte Carlo methodology

- Builds daily signature gains from historical data, then repeatedly samples days from an "optimistic" pool (top-performing days by a deficit-weighted score).
- Applies a partisan efficiency adjustment by comparing observed district progress to partisan-lean expectations, boosting underperforming districts.
- Models hidden signatures via a shadow ratio derived from percentile gaps in daily throughput, with a sandbagging window and an age-based release on the target date.
- Runs many trials (default 100,000) to estimate district and statewide success probabilities.

## Features

- Optimistically samples subset of days that most improve signature counts in underperforming districts (sampling from the top quantile, based on the dot product of deficit vector and daily verification across districts)
- Diagnostic correlation between partisan lean and progress. Simulated samples of underperforming districts (below the partisanship trendline) are boosted by a multiplier based on their relative underperformance.
- Hidden signature scenarios create a store of hidden (unverified) signatures for each day.
    - If the hidden ratio is 1.5, this means that for every 1 signature 'verified' on that day, 1.5 signatures are added to the hidden store to be added to the final count.
    - For historical portion, a default window of 30 days of hidden signatures accumulate additional signatures that will be included in the final count.
    - For future portion, simulated days before the end of signature collection (2/15) also accumulate hidden signatures.
    - Two hidden signature scenarios (conservative and aggressive) are implemented by default.
- Plots saved to the [Hidden_w_PartisanBoost](Hidden_w_PartisanBoost) folder.

## Install

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Execute the simulation script from the repo root:

```bash
python hidden_w_partisan_boost.py
```

## Charts

![Daily verification rates](Hidden_w_PartisanBoost/00_daily_verification_rates.png)
![Partisan diagnostic](Hidden_w_PartisanBoost/01_partisan_diagnostic.png)
![District probabilities comparison](Hidden_w_PartisanBoost/02_district_probabilities_comparison.png)
![Districts passed distribution](Hidden_w_PartisanBoost/03_districts_distribution.png)
![Statewide totals distribution](Hidden_w_PartisanBoost/04_statewide_distribution.png)