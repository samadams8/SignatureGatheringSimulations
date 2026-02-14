import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
import os
from pandas.tseries.holiday import USFederalHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay, Week

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# BUSINESS DAY CALENDAR (excluding holidays)
# ==========================================
class CampaignHolidayCalendar(USFederalHolidayCalendar):
    rules = [
        Holiday('MLK Day', month=1, day=1, offset=[Week(weekday=0), Week(weekday=0), Week(weekday=0)]),
        # Holiday('Presidents Day', month=2, day=1, offset=[Week(weekday=0), Week(weekday=0), Week(weekday=0)]),
    ]

CAMPAIGN_BD = CustomBusinessDay(calendar=CampaignHolidayCalendar())

# ==========================================
# FIGURES DIRECTORY SETUP
# ==========================================
FIGURES_DIR = "HistoricalBootstrap"
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)
    print(f"Created {FIGURES_DIR} directory")

# ==========================================
# CONFIGURATION
# ==========================================
PIVOT_FILE = "Date-District Pivot.csv"
TARGET_DATE = "2026-03-08" 
N_SIMULATIONS = 100000

# OPTIMISM SETTING
# 0.75 = Use top 25% of days
OPTIMISM_THRESHOLD = 0.75

# ==========================================
# HARDCODED REQUIREMENTS (2026)
# ==========================================
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

# ==========================================
# DATA LOADING
# ==========================================
def load_and_clean_data(pivot_path):
    print(f"Loading data from {pivot_path}...")
    try:
        df_raw = pd.read_csv(pivot_path, header=None, nrows=10)
    except FileNotFoundError:
        print(f"ERROR: File '{pivot_path}' not found.")
        return None

    # Auto-detect header
    header_row_idx = 1 # Default fallback
    for idx, row in df_raw.iterrows():
        row_str = row.astype(str).tolist()
        if '1' in row_str and '29' in row_str:
            header_row_idx = idx
            break
    
    # Reload with correct header
    df_pivot = pd.read_csv(pivot_path, header=header_row_idx)
    df_pivot.rename(columns={df_pivot.columns[0]: 'Date'}, inplace=True)
    df_pivot['Date'] = pd.to_datetime(df_pivot['Date'], errors='coerce')
    df_pivot = df_pivot.dropna(subset=['Date']).sort_values('Date').set_index('Date')
    
    # Filter for Districts 1-29
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
    
    # Ensure all 29 districts exist (fill missing with 0)
    all_districts = list(range(1, 30))
    df_pivot = df_pivot.reindex(columns=all_districts, fill_value=0).sort_index(axis=1)
    
    print(f"Data Loaded: {df_pivot.index.min().date()} to {df_pivot.index.max().date()}")
    return df_pivot

# ==========================================
# SIMULATION LOGIC
# ==========================================
def get_simulation_pools(df_pivot):
    """
    Creates two pools of daily gains:
    1. Standard: All historical days.
    2. Optimistic: The top performing days based on effectiveness in districts with most need.
    """
    current_counts = df_pivot.iloc[-1].values
    last_data_date = df_pivot.index[-1]
    
    # Calculate daily gains
    daily_gains = df_pivot.diff().fillna(0)
    if len(daily_gains) > 1:
        daily_gains = daily_gains.iloc[1:]
    
    # 1. Standard Pool (All days)
    standard_pool = daily_gains.values
    
    # 2. Optimistic Pool (Top days based on deficit-weighted effectiveness)
    req_vector = np.array([DISTRICT_REQS[d] for d in range(1, 30)])
    deficits = np.maximum(req_vector - current_counts, 0)
    
    if deficits.sum() > 0:
        # Score each day by how much it helps deficit districts
        daily_effectiveness = daily_gains.values.dot(deficits)
        metric_label = "deficit-weighted effectiveness"
    else:
        # If no deficits, fall back to statewide totals
        daily_effectiveness = daily_gains.sum(axis=1).values
        metric_label = "statewide totals (no deficits)"
    
    threshold = np.percentile(daily_effectiveness, OPTIMISM_THRESHOLD * 100)
    optimistic_pool = daily_gains.values[daily_effectiveness >= threshold]
    
    top_pct = int((1 - OPTIMISM_THRESHOLD) * 100)
    print(f"\nSimulation Basis:")
    print(f"  - Standard Pool:  {len(standard_pool)} days history")
    print(f"  - Optimistic Pool: {len(optimistic_pool)} days history (Top {top_pct}% by {metric_label})")
    print(f"    (Threshold: >{int(threshold)} by metric)")
    
    return current_counts, last_data_date, standard_pool, optimistic_pool

def run_simulation(current_counts, last_data_date, gain_pool, target_date_str, sim_name):
    # Setup
    target_date = pd.to_datetime(target_date_str)
    start_sim_date = last_data_date + timedelta(days=1)
    sim_days = pd.bdate_range(start=start_sim_date, end=target_date, freq=CAMPAIGN_BD)
    num_days = len(sim_days)
    
    print(f"\nRunning '{sim_name}' Simulation...")
    print(f"  Simulating {num_days} business days remaining.")

    final_results = np.zeros((N_SIMULATIONS, 29))
    np.random.seed(42) 
    
    for i in range(N_SIMULATIONS):
        random_indices = np.random.randint(0, len(gain_pool), size=num_days)
        simulated_gains = np.sum(gain_pool[random_indices], axis=0)
        final_results[i] = current_counts + simulated_gains

    return final_results

# ==========================================
# ANALYSIS
# ==========================================
def analyze_results(final_results, current_counts, sim_name):
    req_vector = np.array([DISTRICT_REQS[d] for d in range(1, 30)])
    
    # 1. Statewide Success
    sim_statewide_totals = final_results.sum(axis=1)
    pass_statewide = sim_statewide_totals >= STATEWIDE_REQ
    prob_statewide = pass_statewide.mean()
    
    # 2. Geographic Success (26/29)
    districts_passed = final_results >= req_vector
    count_districts_passed = districts_passed.sum(axis=1)
    pass_geographic = count_districts_passed >= 26
    
    # Combined Success
    success = pass_statewide & pass_geographic
    prob_success = success.mean()
    
    print("-" * 50)
    print(f"RESULTS: {sim_name.upper()}")
    print("-" * 50)
    print(f"  > Probability of Statewide Goal:   {prob_statewide*100:.1f}%")
    print(f"  > Probability of 26/29 Districts:  {prob_success*100:.1f}%")
    
    # District Stats
    district_probs = districts_passed.mean(axis=0)
    df_summary = pd.DataFrame({
        'District': range(1, 30),
        'Required': req_vector,
        'Current': current_counts,
        'Proj_Final': final_results.mean(axis=0).astype(int),
        'Prob_Success': district_probs
    })

    return df_summary, count_districts_passed, sim_statewide_totals

def plot_all_scenarios(summary_std, counts_std, totals_std, summary_opt, counts_opt, totals_opt, last_data_date, standard_days, optimistic_days):
    # --- PLOT 1: District Probabilities ---
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    x = np.arange(1, 30)
    width = 0.35
    
    ax1.bar(x - width/2, summary_std['Prob_Success'], width, 
            label=f'Current Trend ({standard_days} days)', color='salmon', alpha=0.9)
    ax1.bar(x + width/2, summary_opt['Prob_Success'], width, 
            label=f'Optimistic (Top {optimistic_days} days)', color='skyblue', alpha=0.9)
    
    ax1.axhline(y=0.5, color='grey', linestyle='--', linewidth=1, alpha=0.7, label='50% threshold')
    ax1.axhline(y=1.0, color='grey', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Senate District')
    ax1.set_title(f'Probability of Meeting Requirement by District\n(Historical bootstrap, sampling all {standard_days} days vs. best {optimistic_days} days)')
    ax1.set_xticks(x)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    # Footer with data currency (bottom right to stay within bounds)
    fig1.text(0.99, 0.01, f"Data through {last_data_date.date()}", fontsize=8, color='gray', ha='right')
    plt.savefig(os.path.join(FIGURES_DIR, "01_district_probabilities.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/01_district_probabilities.png")
    plt.show()

    # --- PLOT 2: Districts Passed Distribution ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    bins = np.arange(0, 31) - 0.5
    
    weights_std = np.ones_like(counts_std) * 100.0 / len(counts_std)
    weights_opt = np.ones_like(counts_opt) * 100.0 / len(counts_opt)
    
    ax2.hist(counts_std, bins=bins, weights=weights_std, color='salmon', alpha=0.6, 
             label=f'Current Trend ({standard_days} days)', edgecolor='black')
    ax2.hist(counts_opt, bins=bins, weights=weights_opt, color='skyblue', alpha=0.6, 
             label=f'Optimistic (Top {optimistic_days} days)', edgecolor='black')
    
    ax2.axvline(x=26, color='red', linestyle='--', linewidth=2, label='Requirement (26)')
    ax2.set_xlabel('Number of Districts Passed')
    ax2.set_ylabel('Relative frequency (%)')
    ax2.set_title(f'Distribution of Districts Passing\n(Historical bootstrap, sampling all {standard_days} days vs. best {optimistic_days} days)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    # Footer with data currency (bottom right to stay within bounds)
    fig2.text(0.99, 0.01, f"Data through {last_data_date.date()}", fontsize=8, color='gray', ha='right')
    plt.savefig(os.path.join(FIGURES_DIR, "02_districts_distribution.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/02_districts_distribution.png")
    plt.show()
    
    # --- PLOT 3: Statewide Total Distribution ---
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    # Calculate common bins
    min_val = min(totals_std.min(), totals_opt.min())
    max_val = max(totals_std.max(), totals_opt.max())
    bins = np.linspace(min_val, max_val, 50)

    weights_std = np.ones_like(totals_std) * 100.0 / len(totals_std)
    weights_opt = np.ones_like(totals_opt) * 100.0 / len(totals_opt)

    ax3.hist(totals_std, bins=bins, weights=weights_std, color='salmon', alpha=0.6, 
             label=f'Current Trend ({standard_days} days)', edgecolor='grey')
    ax3.hist(totals_opt, bins=bins, weights=weights_opt, color='skyblue', alpha=0.6, 
             label=f'Optimistic (Top {optimistic_days} days)', edgecolor='grey')
    
    ax3.axvline(x=STATEWIDE_REQ, color='red', linestyle='--', linewidth=2, 
                label=f'Requirement ({STATEWIDE_REQ:,})')
    
    ax3.set_xlabel('Total Verified Signatures')
    ax3.set_ylabel('Relative frequency (%)')
    ax3.set_title(f'Final Statewide Totals\n(Historical bootstrap, sampling all {standard_days} days vs. best {optimistic_days} days)')
    ax3.legend(loc='upper left')
    ax3.grid(axis='y', alpha=0.3)
    # Format x-axis with commas
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    # Footer with data currency (bottom right to stay within bounds)
    fig3.text(0.99, 0.01, f"Data through {last_data_date.date()}", fontsize=8, color='gray', ha='right')
    plt.savefig(os.path.join(FIGURES_DIR, "03_statewide_distribution.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/03_statewide_distribution.png")
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df = load_and_clean_data(PIVOT_FILE)
    
    if df is not None:
        # Prepare Pools
        curr, last_date, pool_std, pool_opt = get_simulation_pools(df)
        std_days = len(pool_std)
        opt_days = len(pool_opt)
        
        # Run Standard
        res_std = run_simulation(curr, last_date, pool_std, TARGET_DATE, "Current Trend")
        sum_std, counts_std, totals_std = analyze_results(res_std, curr, "Current Trend")
        
        # Run Optimistic
        res_opt = run_simulation(curr, last_date, pool_opt, TARGET_DATE, "Optimistic (Best Case)")
        sum_opt, counts_opt, totals_opt = analyze_results(res_opt, curr, "Optimistic (Best Case)")
        
        # Plot All 3
        plot_all_scenarios(sum_std, counts_std, totals_std, sum_opt, counts_opt, totals_opt, last_date, std_days, opt_days)