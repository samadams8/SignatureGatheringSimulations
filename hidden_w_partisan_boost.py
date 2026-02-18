import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy import stats
import warnings
import os
from pandas.tseries.holiday import USFederalHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay, Week

warnings.filterwarnings("ignore")

# =============
# CONFIGURATION
# =============
PIVOT_FILE = "Date-District Pivot.csv"
PARTISAN_FILE = "UT_Senate_Partisan_Lean.csv"
TARGET_DATE = "2026-03-08" 
HIDE_DEADLINE = "2026-02-15"  # After this date, no new signatures can be hidden
N_SIMULATIONS = 100000
BLOCK_SIZE = 5
SIGN_AGE_LIMIT = 16  # Number of days signatures are held back
                     # Tuned to give 200k-220k total in aggressive scenario

# OPTIMISM SETTING
# 0.75 = Use top 25% of days
OPTIMISM_THRESHOLD = 0.50

# SANDBAGGING SCENARIOS
# Conservative: Assumes median trickle, 85th percentile capacity
SCENARIO_A_BASE_PCT = 0.50
SCENARIO_A_PEAK_PCT = 0.85

# Aggressive: Assumes low trickle (25th), high capacity (90th)
SCENARIO_B_BASE_PCT = 0.25
SCENARIO_B_PEAK_PCT = 0.90

# ==========================================
# BUSINESS DAY CALENDAR (excluding holidays)
# ==========================================
class CampaignHolidayCalendar(USFederalHolidayCalendar):
    rules = [
        Holiday('MLK Day', month=1, day=1, offset=[Week(weekday=0), Week(weekday=0), Week(weekday=0)]),
    ]

CAMPAIGN_BD = CustomBusinessDay(calendar=CampaignHolidayCalendar())

# ==========================================
# FIGURES DIRECTORY SETUP
# ==========================================
FIGURES_DIR = "Hidden_w_PartisanBoost"
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)
    print(f"Created {FIGURES_DIR} directory")

# ======================
# HARDCODED REQUIREMENTS
# ======================
DISTRICT_REQS = {
    1: 5238, 2: 4687, 3: 4737, 4: 5099, 5: 4115, 
    6: 4745, 7: 5294, 8: 4910, 9: 4805, 10: 2975, 
    11: 4890, 12: 3248, 13: 4088, 14: 5680, 15: 4596, 
    16: 4347, 17: 5368, 18: 5093, 19: 5715, 20: 5292, 
    21: 5684, 22: 5411, 23: 4253, 24: 3857, 25: 4929, 
    26: 5178, 27: 5696, 28: 5437, 29: 5382
}
# DISTRICT_REQS = {
#     1: 5238, 2: 4686, 3: 4737, 4: 5099, 5: 4115,
#     6: 4745, 7: 5294, 8: 4890, 9: 4431, 10: 2975, 
#     11: 4890, 12: 3248, 13: 4088, 14: 5680, 15: 4596, 
#     16: 4347, 17: 5368, 18: 5093, 19: 5715, 20: 5292, 
#     21: 5684, 22: 5411, 23: 4253, 24: 3857, 25: 4929, 
#     26: 5178, 27: 5696, 28: 5437, 29: 5382
# }
STATEWIDE_REQ = 140748

# ============
# DATA LOADING
# ============
def load_data(pivot_path, partisan_path):
    print(f"Loading data...")
    
    # 1. Load Pivot
    try:
        df_raw = pd.read_csv(pivot_path, header=None, nrows=10)
    except FileNotFoundError:
        print("Pivot file not found.")
        return None, None

    header_row_idx = 1 
    for idx, row in df_raw.iterrows():
        row_str = row.astype(str).tolist()
        if '1' in row_str and '29' in row_str:
            header_row_idx = idx
            break
    
    df_pivot = pd.read_csv(pivot_path, header=header_row_idx)
    df_pivot.rename(columns={df_pivot.columns[0]: 'Date'}, inplace=True)
    df_pivot['Date'] = pd.to_datetime(df_pivot['Date'], errors='coerce')
    df_pivot = df_pivot.dropna(subset=['Date']).sort_values('Date').set_index('Date')
    
    valid_cols = []
    rename_map = {}
    for col in df_pivot.columns:
        try:
            d_num = int(float(str(col)))
            if 1 <= d_num <= 29:
                valid_cols.append(col)
                rename_map[col] = d_num
        except:
            continue
            
    df_pivot = df_pivot[valid_cols].rename(columns=rename_map)
    all_districts = list(range(1, 30))
    df_pivot = df_pivot.reindex(columns=all_districts, fill_value=0).sort_index(axis=1)
    
    # 2. Load Partisan Data
    try:
        df_politics = pd.read_csv(partisan_path)
    except FileNotFoundError:
        print("Partisan file not found.")
        return None, None
        
    return df_pivot, df_politics

# ==========================================
# DIAGNOSTIC: CORRELATION (PARTISAN LEAN)
# ==========================================
def run_diagnostic(current_counts, df_politics):
    print("\n--- PARTISAN DIAGNOSTIC ---")
    
    # Prepare Dataframe
    data = []
    for d in range(1, 30):
        req = DISTRICT_REQS[d]
        curr = current_counts[d-1]
        pct_met = curr / req
        
        try:
            row = df_politics[df_politics['District'] == d].iloc[0]
            partisan_val = row['Partisan_Lean_Val'] 
            partisan_str = row['Partisan_Lean_Str']
        except:
            partisan_val = 0.0
            partisan_str = "N/A"
            
        data.append({
            'District': d,
            'Partisan_Lean': partisan_val,
            'Percent_Met': pct_met,
            'Partisan_Str': partisan_str
        })
    
    df_diag = pd.DataFrame(data)
    
    # Correlation
    corr, _ = stats.pearsonr(df_diag['Partisan_Lean'], df_diag['Percent_Met'])
    
    print(f"Correlation (Partisan Lean vs. Signatures): {corr:.3f}")
    if corr > 0.5:
        print("Diagnosis: STRONG Democratic skew (Positive Correlation).")
    elif corr < -0.5:
        print("Diagnosis: STRONG Republican skew (Negative Correlation).")
    elif abs(corr) < 0.3:
        print("Diagnosis: Non-Partisan / Geographical pattern.")
    else:
        direction = "Democratic" if corr > 0 else "Republican"
        print(f"Diagnosis: Weak/Moderate {direction} skew.")
        
    return df_diag, corr

# ==========================================
# SHADOW RATIO CALCULATION
# ==========================================
def calculate_shadow_ratio(df_pivot, base_pct, peak_pct, label):
    """
    Estimate the ratio of hidden signatures based on historical performance.
    """
    daily_gains = df_pivot.diff()
    daily_gains.iloc[0] = df_pivot.iloc[0]
    daily_gains = daily_gains.fillna(0)
    
    daily_totals = daily_gains.sum(axis=1)
    active_days = daily_totals[daily_totals > 10]
    
    base_flow = active_days.quantile(base_pct)
    true_capacity = active_days.quantile(peak_pct)
    shadow_gap = max(0, true_capacity - base_flow)
    
    ratio = 0.0
    if base_flow > 0:
        ratio = shadow_gap / base_flow
        
    print(f"\n[{label}] Shadow Ratio Estimation")
    print(f"  > Typical Visible (Q{base_pct}): {int(base_flow):,}")
    print(f"  > Demonstrated Capacity (Q{peak_pct}): {int(true_capacity):,}")
    print(f"  > Implied Hidden Daily: {int(shadow_gap):,}")
    print(f"  > SHADOW RATIO: {ratio:.2f} (hidden per visible)")
    
    return ratio

# ==========================================
# PARTISAN EFFICIENCY MODEL
# ==========================================
def select_optimistic_pool(boosted_history, current_counts, req_vector, log=False):
    """
    Score boosted history against current deficits and return the optimistic pool.
    """
    deficits = np.maximum(req_vector - current_counts, 0)

    if deficits.sum() > 0:
        daily_effectiveness = boosted_history.dot(deficits)
        metric_label = "deficit-weighted effectiveness"
    else:
        daily_effectiveness = boosted_history.sum(axis=1)
        metric_label = "statewide totals (no deficits)"
    
    threshold = np.percentile(daily_effectiveness, OPTIMISM_THRESHOLD * 100)
    optimistic_pool = boosted_history[daily_effectiveness >= threshold]

    if log:
        print(f"\n--- OPTIMISM FILTER APPLIED ---")
        print(f"Metric: {metric_label}")
        print(f"Retaining {len(optimistic_pool)} days (Threshold: >{int(threshold)} by metric)")

    if len(optimistic_pool) == 0:
        optimistic_pool = boosted_history
    
    return optimistic_pool


def get_probabilistic_boosted_pool(df_pivot, df_diag):
    """
    Applies a probabilistic boost to underperforming districts.
    """
    daily_gains = df_pivot.diff()
    daily_gains.iloc[0] = df_pivot.iloc[0]
    daily_gains = daily_gains.fillna(0)
    
    # 1. Fit a Trend Line (Linear Regression)
    slope, intercept, _, _, std_err = stats.linregress(df_diag['Partisan_Lean'], df_diag['Percent_Met'])
    
    # 2. Calculate Residuals
    predictions = slope * df_diag['Partisan_Lean'] + intercept
    residuals = df_diag['Percent_Met'] - predictions
    residual_std = residuals.std()
    
    print("\n--- PROBABILISTIC SLEEPER DETECTION ---")
    print(f"Model: Expected % = {slope:.2f} * Lean + {intercept:.2f}")
    print(f"Natural Variance (Std Dev): {residual_std:.2f}")
    print(f"{'District':<10} {'Partisan':<12} {'Actual %':<12} {'Expected %':<12} {'Base Boost'}")
    print("-" * 65)
    
    # 3. Calculate Base Boost Factors
    base_boost_factors = np.ones(29)
    
    for d in range(1, 30):
        idx = d - 1
        row = df_diag[df_diag['District'] == d].iloc[0]
        
        actual = row['Percent_Met']
        expected = slope * row['Partisan_Lean'] + intercept
        
        if actual < expected and actual > 0.01:
            ratio = expected / actual
            boost = min(ratio, 2.5)
            base_boost_factors[idx] = boost
            print(f"{d:<10} {row['Partisan_Lean']:.2f}         {actual:.1%}      {expected:.1%}      x{boost:.2f}")
            
    # 4. Apply boost to the history
    boosted_history = daily_gains.values * base_boost_factors

    # 5. Score and select optimistic pool
    current_counts = df_pivot.iloc[-1].values
    req_vector = np.array([DISTRICT_REQS[d] for d in range(1, 30)])
    optimistic_pool = select_optimistic_pool(boosted_history, current_counts, req_vector, log=True)
    
    return optimistic_pool, len(optimistic_pool), boosted_history, base_boost_factors

# ==========================================
# SIMULATION WITH SANDBAGGING
# ==========================================
def run_simulation_with_sandbagging(current_counts, last_data_date, boosted_history,
                                     historical_gains_df, target_date_str, shadow_ratio, hide_deadline_str, base_boost_factors):
    """
    Run simulation with partisan efficiency model AND sandbagging.
    Signatures are released when they reach age limit or on target date (end of period).
    """
    target_date = pd.to_datetime(target_date_str)
    hide_deadline = pd.to_datetime(hide_deadline_str)
    start_sim_date = last_data_date + timedelta(days=1)
    
    # Create business day range and ensure TARGET_DATE is included
    sim_days = pd.bdate_range(start=start_sim_date, end=target_date, freq=CAMPAIGN_BD)
    if sim_days[-1] < target_date:
        sim_days = sim_days.append(pd.DatetimeIndex([target_date]))
    
    num_days = len(sim_days)
    req_vector = np.array([DISTRICT_REQS[d] for d in range(1, 30)])
    
    final_results = np.zeros((N_SIMULATIONS, 29))
    np.random.seed(42)
    
    # Seed historical hidden signatures once (same for all simulations)
    hist_gains_subset = historical_gains_df.loc[:last_data_date]
    initial_hidden_by_day = {}
    for hist_date, base_gain in hist_gains_subset.iterrows():
        if hist_date > hide_deadline:
            continue
        if base_gain.sum() <= 0:
            continue
        age_days = (start_sim_date - hist_date).days
        if age_days >= SIGN_AGE_LIMIT:
            continue
        # Apply partisan efficiency boost to hidden signatures
        boosted_gain = base_gain.values * base_boost_factors
        initial_hidden_by_day[hist_date] = boosted_gain * shadow_ratio
    
    for i in range(N_SIMULATIONS):
        sim_counts = current_counts.copy()
        # Track hidden signatures by collection date (copy of initial historical hidden)
        hidden_by_day = initial_hidden_by_day.copy()
        days_remaining = num_days
        day_pointer = 0

        while day_pointer < num_days:
            # Refresh pool every 5 days
            pool = select_optimistic_pool(boosted_history, sim_counts, req_vector)
            draw = min(BLOCK_SIZE, days_remaining)
            
            for _ in range(draw):
                if day_pointer >= num_days:
                    break
                    
                current_date = sim_days[day_pointer]
                
                # 1. Draw base gain from pool
                random_idx = np.random.randint(0, len(pool))
                base_gain = pool[random_idx]
                sim_counts += base_gain
                
                # 2. Release aged signatures (at least SIGN_AGE_LIMIT days old)
                for hidden_date in list(hidden_by_day.keys()):
                    if (current_date - hidden_date).days >= SIGN_AGE_LIMIT:
                        aged_signatures = hidden_by_day.pop(hidden_date)
                        sim_counts += aged_signatures
                
                # 3. Release all remaining hidden signatures on target date
                if current_date.date() == target_date.date():
                    for hidden_sigs in hidden_by_day.values():
                        sim_counts += hidden_sigs
                    hidden_by_day = {}
                
                # 4. Accumulate Hidden (only before hide deadline)
                if current_date <= hide_deadline and base_gain.sum() > 0:
                    daily_hidden = base_gain * shadow_ratio
                    hidden_by_day[current_date] = daily_hidden
                
                day_pointer += 1
            
            days_remaining -= draw

        final_results[i] = sim_counts

    return final_results

# ==========================================
# PLOTTING
# ==========================================
def plot_diagnostic(df_diag, last_data_date):
    plt.figure(figsize=(10, 6))
    
    x_pct = (-df_diag['Partisan_Lean']) * 100.0
    y_pct = df_diag['Percent_Met'] * 100.0

    plt.scatter(x_pct, y_pct, color='purple', s=100, alpha=0.6, label='Districts')
    
    z = np.polyfit(x_pct, y_pct, 1)
    p = np.poly1d(z)
    plt.plot(x_pct, p(x_pct), "r--", alpha=0.5, label='Trendline')
    
    for _, row in df_diag.iterrows():
        plt.text((-row['Partisan_Lean']) * 100.0, (row['Percent_Met'] * 100.0) + 0.5, 
                str(int(row['District'])), ha='center', fontsize=8)
        
    plt.title('Signature Progress vs. District Partisan Lean')
    plt.xlabel('Partisan Lean % (← D | R →)')
    plt.ylabel('Goal Met (%)')
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gcf().text(0.99, 0.01, f"Data through {last_data_date.date()}", fontsize=8, color='gray', ha='right')
    plt.savefig(os.path.join(FIGURES_DIR, "01_partisan_diagnostic.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/01_partisan_diagnostic.png")
    plt.show()

def plot_daily_verification_rates(df_pivot, last_data_date):
    """
    Plot the daily observed signature verification rate with percentile markers.
    """
    daily_gains = df_pivot.diff()
    daily_gains.iloc[0] = df_pivot.iloc[0]
    daily_gains = daily_gains.fillna(0)
    
    daily_totals = daily_gains.sum(axis=1)
    active_days = daily_totals[daily_totals > 10]
    
    # Calculate percentiles using scenario-defined thresholds
    scenario_b_base = active_days.quantile(SCENARIO_B_BASE_PCT)
    scenario_a_base = active_days.quantile(SCENARIO_A_BASE_PCT)
    scenario_a_peak = active_days.quantile(SCENARIO_A_PEAK_PCT)
    scenario_b_peak = active_days.quantile(SCENARIO_B_PEAK_PCT)
    
    plt.figure(figsize=(14, 7))
    
    # Plot daily totals
    plt.plot(active_days.index, active_days.values, marker='o', linestyle='-', 
             color='steelblue', alpha=0.6, markersize=4, label='Daily Verifications')
    
    # Plot percentile lines (using scenario-defined thresholds)
    plt.axhline(y=scenario_a_base, color='green', linestyle='--', linewidth=1.5, alpha=0.7, 
                label=f'{scenario_a_base:,.0f} [{SCENARIO_A_BASE_PCT:.0%}]')
    plt.axhline(y=scenario_a_peak, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                label=f'{scenario_a_peak:,.0f} [{SCENARIO_A_PEAK_PCT:.0%}]')
    plt.axhline(y=scenario_b_base, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, 
                label=f'{scenario_b_base:,.0f} [{SCENARIO_B_BASE_PCT:.0%}]')
    plt.axhline(y=scenario_b_peak, color='darkred', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'{scenario_b_peak:,.0f} [{SCENARIO_B_PEAK_PCT:.0%}]')
    
    plt.title('Daily Signature Verification Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Verified Signatures', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.gcf().text(0.99, 0.01, f"Data through {last_data_date.date()}", fontsize=8, color='gray', ha='right')
    plt.savefig(os.path.join(FIGURES_DIR, "00_daily_verification_rates.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/00_daily_verification_rates.png")
    plt.show()

def plot_comparison_results(res_a, res_b, pool_size, ratio_a, ratio_b, last_data_date):
    req_vector = np.array([DISTRICT_REQS[d] for d in range(1, 30)])
    
    # Calculate success metrics
    districts_passed_a = res_a >= req_vector
    districts_passed_b = res_b >= req_vector
    
    count_a = districts_passed_a.sum(axis=1)
    count_b = districts_passed_b.sum(axis=1)
    
    success_a = (res_a.sum(axis=1) >= STATEWIDE_REQ) & (count_a >= 26)
    success_b = (res_b.sum(axis=1) >= STATEWIDE_REQ) & (count_b >= 26)
    
    print(f"\n--- RESULTS SUMMARY ---")
    print(f"Conservative Scenario (hidden ratio={ratio_a:.2f}): {success_a.mean()*100:.1f}% success")
    print(f"Aggressive Scenario (hidden ratio={ratio_b:.2f}): {success_b.mean()*100:.1f}% success")
    
    # Plot 1: District Probabilities Comparison (Grouped Bar Chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    probs_a = districts_passed_a.mean(axis=0)
    probs_b = districts_passed_b.mean(axis=0)
    
    x = np.arange(1, 30)
    width = 0.35
    
    ax.bar(x - width/2, probs_a, width, color='blue', alpha=0.8, 
           label=f'Conservative Scenario (hidden ratio={ratio_a:.2f})', edgecolor='blue')
    ax.bar(x + width/2, probs_b, width, color='orange', alpha=0.8, 
           label=f'Aggressive Scenario (hidden ratio={ratio_b:.2f})', edgecolor='orange')
    
    ax.axhline(y=0.5, color='grey', linestyle='--', linewidth=1, alpha=0.7, label='50% threshold')
    ax.axhline(y=1.0, color='grey', linestyle='-', alpha=0.3)
    ax.set_title(f'Probability of Meeting Requirement by District\n(Hidden signatures, partisan efficiency, sampling best {pool_size} days)')
    ax.set_xlabel('Senate District')
    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.text(0.99, 0.01, f"Data through {last_data_date.date()}", fontsize=8, color='gray', ha='right')
    plt.savefig(os.path.join(FIGURES_DIR, "02_district_probabilities_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/02_district_probabilities_comparison.png")
    plt.show()
    
    # Plot 2: Districts Passed Distribution (overlapping histograms)
    plt.figure(figsize=(12, 6))
    bins = np.arange(0, 31) - 0.5
    
    weights_a = np.ones_like(count_a) * 100.0 / len(count_a)
    weights_b = np.ones_like(count_b) * 100.0 / len(count_b)
    
    plt.hist(count_a, bins=bins, weights=weights_a, color='blue', alpha=0.6, 
             label=f'Conservative Scenario (hidden ratio={ratio_a:.2f})', edgecolor='blue')
    plt.hist(count_b, bins=bins, weights=weights_b, color='orange', alpha=0.6, 
             label=f'Aggressive Scenario (hidden ratio={ratio_b:.2f})', edgecolor='orange')
    
    plt.axvline(x=26, color='red', linestyle='--', linewidth=2, label='Requirement (26)')
    plt.title(f'Distribution of Districts Passing\n(Hidden signatures, partisan efficiency, sampling best {pool_size} days)')
    plt.xlabel('Number of Districts Passed')
    plt.ylabel('Relative frequency (%)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.gcf().text(0.99, 0.01, f"Data through {last_data_date.date()}", fontsize=8, color='gray', ha='right')
    plt.savefig(os.path.join(FIGURES_DIR, "03_districts_distribution.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/03_districts_distribution.png")
    plt.show()
    
    # Plot 3: Statewide Totals Distribution (overlapping histograms)
    plt.figure(figsize=(12, 6))
    
    totals_a = res_a.sum(axis=1)
    totals_b = res_b.sum(axis=1)
    
    min_val = min(totals_a.min(), totals_b.min())
    max_val = max(totals_a.max(), totals_b.max())
    bins = np.linspace(min_val, max_val, 50)
    
    weights_a = np.ones_like(totals_a) * 100.0 / len(totals_a)
    weights_b = np.ones_like(totals_b) * 100.0 / len(totals_b)
    
    plt.hist(totals_a, bins=bins, weights=weights_a, color='blue', alpha=0.6, 
             label=f'Conservative Scenario (hidden ratio={ratio_a:.2f})', edgecolor='blue')
    plt.hist(totals_b, bins=bins, weights=weights_b, color='orange', alpha=0.6, 
             label=f'Aggressive Scenario (hidden ratio={ratio_b:.2f})', edgecolor='orange')
    
    plt.axvline(x=STATEWIDE_REQ, color='red', linestyle='--', linewidth=2, 
                label=f'Requirement ({STATEWIDE_REQ:,})')
    plt.xlabel('Total Verified Signatures')
    plt.ylabel('Relative frequency (%)')
    plt.title(f'Final Statewide Totals\n(Hidden signatures, partisan efficiency, sampling best {pool_size} days)')
    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.gcf().text(0.99, 0.01, f"Data through {last_data_date.date()}", fontsize=8, color='gray', ha='right')
    plt.savefig(os.path.join(FIGURES_DIR, "04_statewide_distribution.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/04_statewide_distribution.png")
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df_pivot, df_politics = load_data(PIVOT_FILE, PARTISAN_FILE)
    
    if df_pivot is not None and df_politics is not None:
        current = df_pivot.iloc[-1].values
        last_date = df_pivot.index[-1]
        historical_gains_df = df_pivot.diff()
        historical_gains_df.iloc[0] = df_pivot.iloc[0]
        historical_gains_df = historical_gains_df.fillna(0)
        
        # 1. Diagnostic
        df_diag, corr = run_diagnostic(current, df_politics)
        plot_diagnostic(df_diag, last_date)
        
        # 2. Plot Daily Verification Rates
        plot_daily_verification_rates(df_pivot, last_date)
        
        # 3. Calculate Shadow Ratios
        ratio_a = calculate_shadow_ratio(df_pivot, SCENARIO_A_BASE_PCT, SCENARIO_A_PEAK_PCT, "Conservative Scenario")

        ratio_b = calculate_shadow_ratio(df_pivot, SCENARIO_B_BASE_PCT, SCENARIO_B_PEAK_PCT, "Aggressive Scenario")
        
        # 4. Prepare Efficiency Boosted Pool
        if abs(corr) > 0.2:
            _, pool_size, boosted_history, base_boost_factors = get_probabilistic_boosted_pool(df_pivot, df_diag)
            
            # 5. Run Simulations
            print("\n--- Running Conservative Scenario Simulation ---")
            res_a = run_simulation_with_sandbagging(current, last_date, boosted_history,
                                                   historical_gains_df, TARGET_DATE, ratio_a, HIDE_DEADLINE, base_boost_factors)
            
            print("\n--- Running Aggressive Scenario Simulation ---")
            res_b = run_simulation_with_sandbagging(current, last_date, boosted_history,
                                                   historical_gains_df, TARGET_DATE, ratio_b, HIDE_DEADLINE, base_boost_factors)
            
            # 6. Plot Comparison
            plot_comparison_results(res_a, res_b, pool_size, ratio_a, ratio_b, last_date)
        else:
            print("\nCorrelation too weak to run Partisan Efficiency model.")
