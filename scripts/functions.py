import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, norm
from collections import defaultdict
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from fastdtw import fastdtw

##############################################################################################################################
## PARAMETERS
##############################################################################################################################

# Game Parameters
N_PLAYERS = 6  # Players per game
ROUNDS = 10  # Number of rounds
GOAL = 120  # Collective goal

# Game distribution
GAME_TYPES = {
    "EQUAL": 6,
    "UNEQUAL-L": 27,
    "UNEQUAL-H": 44
}

# Game distribution for hybrid
GAME_TYPES_hybrid = {
    "EQUAL": 10000,
    "UNEQUAL-L": 10000,
    "UNEQUAL-H": 10000
}

# Endowment configurations
ENDOWMENT_CONFIGS = {
    "EQUAL": [40, 40, 40, 40, 40, 40],
    "UNEQUAL-L": [24, 24, 48, 48, 48, 48],
    "UNEQUAL-H": [30, 30, 30, 30, 60, 60]
}
# Define color palette for control wealth types
wealth_colors = {
    'EQUAL': '#004080',     # Dark Blue – fair, serious, and stable
    'UNEQUAL-L': '#006400', # Dark Green – mature and grounded
    'UNEQUAL-H': '#800020'  # Granate – deep and dramatic
}

# Probabilities for Round 1 Contributions (Based on Initial Endowment)
round1_probabilities = {
    24: [0.074074, 0.500000, 0.425926],
    30: [0.045455, 0.613636, 0.340909],
    40: [0.055556, 0.500000, 0.444444],
    48: [0.027778, 0.555556, 0.416667],
    60: [0.045455, 0.352273, 0.602273]
}

# Define round columns
round_columns = [f'R{i}' for i in range(1, 11)]

##############################################################################################################################
## SIMULATION FUNCTIONS 
##############################################################################################################################

# Function to initialize contributions for Round 1
def initialize_round1(endowments,rng):
    """Initialize Round 1 contributions based on initial endowment probabilities."""
    contributions_history = np.zeros((N_PLAYERS, ROUNDS))

    for i in range(N_PLAYERS):
        endowment = endowments[i]
        probabilities = np.array(round1_probabilities[endowment]) # Get probabilities for this endowment
        probabilities = probabilities / probabilities.sum()  # Normalize probabilities to sum to 1
        contributions_history[i, 0] = rng.choice([0, 2, 4], p=probabilities)  # Select R1 contribution

    return contributions_history

# Defining contribution function depending on the hypothesis
def choose_contribution(df, endowment, condition_value, remaining_coins, hypothesis,rng):
    """Select a contribution (0, 2, or 4) based on dataset probabilities and available endowment."""

    # Convert numerical condition_value to correct bin label for H1, H2 and H3
    if hypothesis == "H1":
        bins = [0, 1, 2, 3, float("inf")]
        labels = ["[0,1)", "[1,2)", "[2,3)", "[3,4]"]      
        condition_label = pd.cut([condition_value], bins=bins, labels=labels, include_lowest=True, right=False)[0]

    elif hypothesis == "H2":
        bins = [0, 24, 48, 72, 96, 120, float("inf")]
        labels = ["[0,24)", "[24,48)", "[48,72)", "[72,96)", "[96,120)", "≥120"]
        condition_label = pd.cut([condition_value], bins=bins, labels=labels, include_lowest=True, right=False)[0]

    elif hypothesis == "H3":
        bins = [0, 3/10, 6/10, float("inf")]
        labels = ["[0,30%)", "[30%,60%)", "≥60%"]
        condition_label = pd.cut([condition_value], bins=bins, labels=labels, include_lowest=True, right=False)[0]

    else:  # For H0, no conversion needed
        condition_label = condition_value

    # Find the matching row in the dataset
    subset = df[(df["endowment_initial"] == endowment) & (df.iloc[:, 1] == condition_label)]

    if subset.empty:
        return 0  # Default to 0 if no matching condition is found

    # Extract probabilities directly from the dataset
    probabilities = subset.iloc[:, 2:5].values.flatten()  # Columns for 0, 2, 4 contributions
    contributions = np.array([0, 2, 4])

    # Adjust for remaining endowment (can't contribute more than available)
    valid_indices = contributions <= remaining_coins
    contributions = contributions[valid_indices]
    probabilities = probabilities[valid_indices]

    # If only one valid contribution remains, return it directly
    if len(contributions) == 1:
        return contributions[0]

    # Ensure probabilities remain valid for selection
    probabilities = probabilities / probabilities.sum()  # Re-normalize to sum 1 if some options were removed

    return rng.choice(contributions, p=probabilities)

# Function to simulate games and return structured player-level data
def simulate_games_dataframe(df, hypothesis, rng):
    """Simulate games and return a structured dataframe with player-level data."""
    data_records = []  # List to store each player's data
    game_id = 0  # Unique identifier for each game
    
    for game_type, num_games in GAME_TYPES.items():
        for _ in range(num_games):
            game_id += 1
            endowments = ENDOWMENT_CONFIGS[game_type][:]
            payoffs = ENDOWMENT_CONFIGS[game_type][:]
            contributions_history = initialize_round1(endowments, rng)  # R1
            total_contributions = np.sum(contributions_history[:, 0])
            goal_reached = False

            for i in range(N_PLAYERS):  # Adjust initial payoffs
                payoffs[i] -= contributions_history[i, 0]

            for round_idx in range(1, ROUNDS):
                for i in range(N_PLAYERS):
                    remaining_coins = payoffs[i]

                    if hypothesis == "Hybridisation":
                        contrib_choices = {}
                        for hyp in ["H0", "H1", "H2", "H3"]:
                            if hyp == "H0":
                                condition_val = contributions_history[i, round_idx - 1] if round_idx > 0 else 0
                            elif hyp == "H1":
                                others = [contributions_history[j, round_idx - 1] for j in range(N_PLAYERS) if j != i] if round_idx > 0 else 0
                                condition_val = np.mean(others)
                            elif hyp == "H2":
                                condition_val = total_contributions
                            elif hyp == "H3":
                                condition_val = (endowments[i] - remaining_coins) / endowments[i]

                            contrib = choose_contribution(df[hyp], endowments[i], condition_val, remaining_coins, hyp, rng)
                            contrib_choices[hyp] = contrib

                        # Get most frequent contribution
                        contribs = list(contrib_choices.values())
                        most_common = max(set(contribs), key=contribs.count)
                        candidates = [c for c in set(contribs) if contribs.count(c) == contribs.count(most_common)]
                        contribution = rng.choice(candidates) if len(candidates) > 1 else most_common

                    else:  # Regular hypothesis
                        if hypothesis == "H0":
                            condition_val = contributions_history[i, round_idx - 1] if round_idx > 0 else 0
                        elif hypothesis == "H1":
                            others = [contributions_history[j, round_idx - 1] for j in range(N_PLAYERS) if j != i] if round_idx > 0 else 0
                            condition_val = np.mean(others)
                        elif hypothesis == "H2":
                            condition_val = total_contributions
                        elif hypothesis == "H3":
                            condition_val = (endowments[i] - remaining_coins) / endowments[i]

                        contribution = choose_contribution(df[hypothesis], endowments[i], condition_val, remaining_coins, hypothesis, rng)

                    contributions_history[i, round_idx] = contribution
                    payoffs[i] -= contribution

                total_contributions += np.sum(contributions_history[:, round_idx])
                if total_contributions >= GOAL:
                    goal_reached = True
                #    break  # You can keep this to cut early
            # Record player-level data
            for i in range(N_PLAYERS):
                user_id = game_id * 10 + i
                total_contributed = np.sum(contributions_history[i, :])
                winnings = (endowments[i] - total_contributed) if goal_reached else 0
                goalreached = 1 if goal_reached else 0

                player_data = {
                    "user_id": user_id,
                    "partida_id": game_id,
                    "endowment_initial": endowments[i],
                    "control_wealth": game_type,
                    "endowment_current": endowments[i] - total_contributed,
                    "contributed_public_goods": total_contributed,
                    "winnings_public_goods": winnings,
                    "goal_reached": goalreached
                }

                for round_idx in range(ROUNDS):
                    player_data[f"R{round_idx+1}"] = contributions_history[i, round_idx]

                data_records.append(player_data)

    df_result = pd.DataFrame(data_records)
    
    return df_result

def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d

# Hybrid Function segregating by Successful and Unsuccessful games
def simulate_games_dataframe_hybrid(df, hypothesis, rng):
    """Simulate games and return a structured dataframe with player-level data."""
    data_records = []
    game_id = 0

    if hypothesis == "Hybridisation":
        Hybridisation_counter = pd.DataFrame(0, index=[0, 2, 4], columns=["H0", "H1", "H2", "H3"])

        def init_counters():
            return {
                "winning_hypothesis_counter": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))),
                "hypothesis_pair_matrix": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))),
                "Hybridisation_difficulty": defaultdict(lambda: defaultdict(list)),
                "contribution_variability": defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            }

        # Temporary counters during a single game
        temp_counters = init_counters()

        # Final aggregated counters for each outcome type
        counters_successful = init_counters()
        counters_unsuccessful = init_counters()

    for game_type, num_games in GAME_TYPES_hybrid.items():
        for _ in range(num_games):
            game_id += 1
            endowments = ENDOWMENT_CONFIGS[game_type][:]
            payoffs = ENDOWMENT_CONFIGS[game_type][:]
            contributions_history = initialize_round1(endowments, rng)
            total_contributions = np.sum(contributions_history[:, 0])
            goal_reached = False

            for i in range(N_PLAYERS):
                payoffs[i] -= contributions_history[i, 0]

            if hypothesis == "Hybridisation":
                temp_counters = init_counters()  # Reset temp for each game

            for round_idx in range(1, ROUNDS):
                for i in range(N_PLAYERS):
                    remaining_coins = payoffs[i]

                    if hypothesis == "Hybridisation":
                        contrib_choices = {}
                        for hyp in ["H0", "H1", "H2", "H3"]:
                            if hyp == "H0":
                                condition_val = contributions_history[i, round_idx - 1]
                            elif hyp == "H1":
                                others = [contributions_history[j, round_idx - 1] for j in range(N_PLAYERS) if j != i]
                                condition_val = np.mean(others)
                            elif hyp == "H2":
                                condition_val = total_contributions
                            elif hyp == "H3":
                                condition_val = (endowments[i] - remaining_coins) / endowments[i]

                            contrib = choose_contribution(df[hyp], endowments[i], condition_val, remaining_coins, hyp, rng)
                            contrib_choices[hyp] = contrib
                            Hybridisation_counter.at[contrib, hyp] += 1

                        contribs = list(contrib_choices.values())
                        most_common = max(set(contribs), key=contribs.count)
                        candidates = [c for c in set(contribs) if contribs.count(c) == contribs.count(most_common)]
                        contribution = rng.choice(candidates) if len(candidates) > 1 else most_common

                        # Store counters temporarily (to copy at end based on goal)
                        endowment = endowments[i]

                        # 1. Winning Hypothesis by Contribution
                        for hyp, val in contrib_choices.items():
                            if val == contribution:
                                temp_counters["winning_hypothesis_counter"][endowment][round_idx][hyp][val] += 1

                        # 2. Hypothesis Pair Co-Agreement Matrix by Contribution
                        matching_hyps = [h for h, v in contrib_choices.items() if v == contribution]
                        for i1 in range(len(matching_hyps)):
                            for i2 in range(i1 + 1, len(matching_hyps)):
                                h1, h2 = sorted((matching_hyps[i1], matching_hyps[i2]))
                                temp_counters["hypothesis_pair_matrix"][endowment][round_idx][(h1, h2)][contribution] += 1

                        # 3. Hybridisation Difficulty
                        unique_contribs = set(contrib_choices.values())
                        temp_counters["Hybridisation_difficulty"][endowment][round_idx].append(len(unique_contribs))

                        # 4. Contribution Variability
                        for hyp, val in contrib_choices.items():
                            temp_counters["contribution_variability"][endowment][round_idx][hyp].append(val)

                    else:
                        if hypothesis == "H0":
                            condition_val = contributions_history[i, round_idx - 1]
                        elif hypothesis == "H1":
                            others = [contributions_history[j, round_idx - 1] for j in range(N_PLAYERS) if j != i]
                            condition_val = np.mean(others)
                        elif hypothesis == "H2":
                            condition_val = total_contributions
                        elif hypothesis == "H3":
                            condition_val = (endowments[i] - remaining_coins) / endowments[i]

                        contribution = choose_contribution(df[hypothesis], endowments[i], condition_val, remaining_coins, hypothesis, rng)

                    contributions_history[i, round_idx] = contribution
                    payoffs[i] -= contribution

                total_contributions += np.sum(contributions_history[:, round_idx])
                if total_contributions >= GOAL:
                    goal_reached = True

            # At the end of the game, assign counters to success/failure bins
            if hypothesis == "Hybridisation":
                target = counters_successful if goal_reached else counters_unsuccessful
                
                for key in temp_counters:
                    for endowment in temp_counters[key]:
                        for round_idx in temp_counters[key][endowment]:
            
                            # 1. Winning Hypothesis & Hypothesis Pair Matrix
                            if key in ["winning_hypothesis_counter", "hypothesis_pair_matrix"]:
                                for subkey in temp_counters[key][endowment][round_idx]:
                                    for contrib, count in temp_counters[key][endowment][round_idx][subkey].items():
                                        target[key][endowment][round_idx][subkey][contrib] += count
            
                            # 2. Contribution Variability (per hypothesis)
                            elif key == "contribution_variability":
                                for hyp in temp_counters[key][endowment][round_idx]:
                                    target[key][endowment][round_idx].setdefault(hyp, [])
                                    target[key][endowment][round_idx][hyp].extend(
                                        temp_counters[key][endowment][round_idx][hyp]
                                    )
            
                            # 3. Hybridisation Difficulty (per round)
                            elif key == "Hybridisation_difficulty":
                                target[key][endowment].setdefault(round_idx, [])
                                target[key][endowment][round_idx].extend(
                                    temp_counters[key][endowment][round_idx]
                                )


            for i in range(N_PLAYERS):
                user_id = game_id * 10 + i
                total_contributed = np.sum(contributions_history[i, :])
                winnings = (endowments[i] - total_contributed) if goal_reached else 0
                goalreached = 1 if goal_reached else 0

                player_data = {
                    "user_id": user_id,
                    "partida_id": game_id,
                    "endowment_initial": endowments[i],
                    "control_wealth": game_type,
                    "endowment_current": endowments[i] - total_contributed,
                    "contributed_public_goods": total_contributed,
                    "winnings_public_goods": winnings,
                    "goal_reached": goalreached
                }

                for round_idx in range(ROUNDS):
                    player_data[f"R{round_idx+1}"] = contributions_history[i, round_idx]

                data_records.append(player_data)

    df_result = pd.DataFrame(data_records)

    if hypothesis == "Hybridisation":
        return (
            df_result, 
            Hybridisation_counter,
            convert_defaultdict_to_dict(counters_successful),
            convert_defaultdict_to_dict(counters_unsuccessful)
        )
    else:
        return df_result

##############################################################################################################################
## MACRO INDICATORS FUNCTIONS 
##############################################################################################################################

# http://www.ellipsix.net/blog/2012/11/the-gini-coefficient-for-distribution-inequality.html
def gini_coeff(x):
    """
    Compute the Gini coefficient of a non-negative array x.
    Handles cases where x is empty or contains all zeros.
    """
    x = np.asarray(x)  # Ensure input is a NumPy array
    n = len(x)
    s = x.sum()
    if n == 0 or s == 0:
        return 0  # Return 0 if the array is empty or sum is zero
    r = np.argsort(np.argsort(-x))  # Compute zero-based ranks
    return 1.0 - (2.0 * (r * x).sum() + s) / (n * s)

def lorenz_curve(x):
    x = np.sort(np.asarray(x))
    n = len(x)
    if n == 0:
        return np.array([0]), np.array([0])
    
    if x.sum() == 0:
        # Everyone has 0 → return perfect equality line
        cumulative_share = np.linspace(0, 1, n + 1)
    else:
        cumulative_share = np.cumsum(x) / np.sum(x)
        cumulative_share = np.insert(cumulative_share, 0, 0)
    
    population_share = np.linspace(0, 1, n + 1)
    return population_share, cumulative_share

def compute_macro_observables(df_users):
    """
    Computes macro-observables by initial endowment for a dataset of the form df_users.
    
    Returns:
    DataFrame with columns: 
        endowment_initial, Gini_initial, Average_gini_after, Average_round_reached,
        Payoff, Payoff normalized relative fariness, Proportion_contributed, Average_contribution, 
        Average_contribution_round (for each round R1 to R10).
    """
    
    # Define round columns
    round_columns = [f'R{i}' for i in range(1, 11)]
    
    # Placeholder for storing the round where each partida_id reaches the goal
    goal_reach_results = []
    # Placeholder for results
    results = []
    
    df = df_users.copy()
    
    # Iterate over games (partida_id)
    for partida_id, game_data in df.groupby("partida_id"):
        cumulative_contribution = 0  # Track cumulative contribution over rounds
        goal_reached_round = None
        games_reached=0
        gini_initial_part = gini_coeff(game_data['endowment_initial'])
        #gini_after_part = gini_coeff(game_data['winnings_public_goods'])
        gini_after_part = gini_coeff(game_data['endowment_current'])

        for round_index, round_col in enumerate(round_columns):
            # Update cumulative contribution for all players up to this round
            cumulative_contribution += game_data[round_col].sum()

            # Check if the goal of 120 coins has been reached
            if cumulative_contribution >= 120 and goal_reached_round is None:
                goal_reached_round = round_index + 1  # Store the round (1-based index)
                games_reached=1
                break  # Stop checking further rounds once the goal is reached

        # Store the result for each initial endowment within the game
        for endowment in game_data["endowment_initial"].unique():
            goal_reach_results.append({
                "partida_id": partida_id,
                "control_wealth": game_data["control_wealth"].iloc[0],
                "endowment_initial": endowment,
                "goal_reached_round": goal_reached_round,
                "game_reached": games_reached,
                "gini_initial": gini_initial_part,
                "gini_after": gini_after_part
            })      

    # Convert to DataFrame
    df_goal_reach = pd.DataFrame(goal_reach_results)

    # Compute the average round when the goal was reached, grouped by endowment and the gini after
    avg_goal_reach_by_endowment = df_goal_reach.groupby("endowment_initial")["goal_reached_round"].mean()
    avg_gini_initial = df_goal_reach.groupby("endowment_initial")["gini_initial"].mean()
    avg_gini_after = df_goal_reach.groupby("endowment_initial")["gini_after"].mean()
    count_goals = df_goal_reach.groupby("endowment_initial")["game_reached"].sum()

    # Group by initial endowment
    for endowment, group in df.groupby("endowment_initial"):
        # Initial wealth Gini index
        gini_initial = avg_gini_initial[endowment]

        # Payoff Gini index (wealth after game)
        gini_after = avg_gini_after[endowment]

        # Average round reached (when reaching 120 cumulative)
        average_round_reached =  avg_goal_reach_by_endowment[endowment]

        # Payoff: Average remaining savings
        payoff = group["endowment_current"].mean()

        relative_fairness_payoff = payoff / (0.5*endowment)
        
        # Proportion contributed (total contributed over initial endowment)
        proportion_contributed = (group["contributed_public_goods"] / group["endowment_initial"]).mean()
        
        # Average contribution per user
        avg_contribution = group[round_columns].mean().mean()

        # Average contribution per round
        avg_contribution_rounds = group[round_columns].mean().to_dict()

        # Store results
        results.append({
            "endowment_initial": endowment,
            "Gini_initial": gini_initial,
            "Average_gini_after": gini_after,
            "Average_round_reached": average_round_reached,
            "Count_games_reached": count_goals[endowment],
            "Proportion_games_reached": count_goals[endowment]/len(group["partida_id"].unique()),
            "Payoff": payoff,
            "Payoff_relative_fairness": relative_fairness_payoff,
            "Proportion_contributed": proportion_contributed,
            "Average_contribution": avg_contribution,
            **avg_contribution_rounds
        })

    # Convert to DataFrame
    return pd.DataFrame(results)

##############################################################################################################################
## SIMILARITIES FUNCTIONS 
##############################################################################################################################

def compare_macro_observables(df1, df2):
    """
    Compares two macro-observable DataFrames computed from `compute_macro_observables()`.
    
    Returns:
    - Difference DataFrame by endowment and by macro-observable.
    - Correlation of all columns.
    """
    
    # Merge DataFrames on endowment_initial
    merged = df1.merge(df2, on="endowment_initial", suffixes=('_df1', '_df2'))
    
    # Compute differences
    diff_columns = [col for col in df1.columns if col != "endowment_initial"]
    differences = merged[["endowment_initial"]].copy()
    
    for col in diff_columns:
        differences[col] = merged[f"{col}_df1"] - merged[f"{col}_df2"]
    
    # Compute correlation between the two dataframes
    correlation_results = {}
    for col in diff_columns:
        if f"{col}_df1" in merged and f"{col}_df2" in merged:
            correlation_results[col] = pearsonr(merged[f"{col}_df1"], merged[f"{col}_df2"])[0]

    return differences, correlation_results

# Helpers for propagated error on grouped data
def propagated_plus_sem(std_array, mean_array):
    """
    std_array: per-observation std (here, df['std'] of each decision vector)
    mean_array: per-observation mean (here, df['mean'] of each decision vector)
    returns total_error = sqrt( (sqrt(sum(std_i^2))/n)^2 + (std(mean_array)/sqrt(n))^2 )
    """
    n = len(mean_array)
    if n == 0:
        return np.nan
    sigma_prop = np.sqrt(np.sum(np.square(std_array))) / n
    sem = np.std(mean_array, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return np.sqrt(sigma_prop**2 + sem**2)

# Wilson score for the CI 95% of the Proportion of games archieved.
def compute_wilson_ci(successes, n, alpha=0.05):
    if n == 0:
        return (0.0, 0.0)
    
    z = norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
    
    lower_bound = (centre - margin) / denominator
    upper_bound = (centre + margin) / denominator
    return lower_bound, upper_bound

# Fisher CI for Pearson
def pearson_confidence_interval(r, n, alpha=0.05):
    if n <= 3:
        return None, None
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    z_low, z_high = z - z_crit * se, z + z_crit * se
    r_low = (np.exp(2 * z_low) - 1) / (np.exp(2 * z_low) + 1)
    r_high = (np.exp(2 * z_high) - 1) / (np.exp(2 * z_high) + 1)
    return r_low, r_high

# Normalized MSE by variance
def nMSE_variance(y_true, y_pred):
    var = np.var(y_true)
    return mean_squared_error(y_true, y_pred) / var if var > 0 else np.nan

# Normalized MSE by range
def nMSE_range(y_true, y_pred):
    range_sq = (np.max(y_true) - np.min(y_true)) ** 2
    return mean_squared_error(y_true, y_pred) / range_sq if range_sq > 0 else np.nan

# Bootstrap CI for nMSE (variance-based)
def bootstrap_nMSE_variance_ci(y_true, y_pred, n_boot=1000, alpha=0.05):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    scores = []

    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        y_t, y_p = y_true[idx], y_pred[idx]
        var = np.var(y_t)
        if var > 0:
            scores.append(mean_squared_error(y_t, y_p) / var)

    if scores:
        lower = np.percentile(scores, 100 * (alpha / 2))
        upper = np.percentile(scores, 100 * (1 - alpha / 2))
        return lower, upper
    return None, None

# Bootstrap CI for nMSE (range-based)
def bootstrap_nMSE_range_ci(y_true, y_pred, n_boot=1000, alpha=0.05):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    scores = []

    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        y_t, y_p = y_true[idx], y_pred[idx]
        range_sq = (np.max(y_t) - np.min(y_t))**2
        if range_sq > 0:
            scores.append(mean_squared_error(y_t, y_p) / range_sq)

    if scores:
        lower = np.percentile(scores, 100 * (alpha / 2))
        upper = np.percentile(scores, 100 * (1 - alpha / 2))
        return lower, upper
    return None, None

# Similarity metrics for average contributions by endowment

def similarities_Average(df1, df2):
    df_users_rounds_1 = df1[['endowment_initial'] + round_columns]
    df_users_rounds_2 = df2[['endowment_initial'] + round_columns]

    mean_contributions_1 = df_users_rounds_1.groupby('endowment_initial').mean()
    mean_contributions_2 = df_users_rounds_2.groupby('endowment_initial').mean()

    all_metrics = {}
    means1all, means2all = [], []

    for endowment in df1['endowment_initial'].unique():
        y_true = mean_contributions_1.loc[endowment].values
        y_pred = mean_contributions_2.loc[endowment].values
        n = len(y_true)

        r, p = pearsonr(y_true, y_pred)
        r_ci = pearson_confidence_interval(r, n)

        nmse_var = nMSE_variance(y_true, y_pred)
        nmse_rng = nMSE_range(y_true, y_pred)

        nmse_var_ci = bootstrap_nMSE_variance_ci(y_true, y_pred)
        nmse_rng_ci = bootstrap_nMSE_range_ci(y_true, y_pred)

        all_metrics[(endowment, 'average_pearson')] = r
        all_metrics[(endowment, 'average_pearson_p')] = p
        all_metrics[(endowment, 'average_pearson_ci')] = r_ci
        all_metrics[(endowment, 'average_nMSE_variance')] = nmse_var
        all_metrics[(endowment, 'average_nMSE_variance_ci')] = nmse_var_ci
        all_metrics[(endowment, 'average_nMSE_range')] = nmse_rng
        all_metrics[(endowment, 'average_nMSE_range_ci')] = nmse_rng_ci

        means1all.extend(y_true)
        means2all.extend(y_pred)

    # Global "ALL" metrics
    r_all, p_all = pearsonr(means1all, means2all)
    r_ci_all = pearson_confidence_interval(r_all, len(means1all))

    nmse_var_all = nMSE_variance(means1all, means2all)
    nmse_rng_all = nMSE_range(means1all, means2all)
    nmse_var_ci_all = bootstrap_nMSE_variance_ci(means1all, means2all)
    nmse_rng_ci_all = bootstrap_nMSE_range_ci(means1all, means2all)

    all_metrics[("ALL", 'average_pearson')] = r_all
    all_metrics[("ALL", 'average_pearson_p')] = p_all
    all_metrics[("ALL", 'average_pearson_ci')] = r_ci_all
    all_metrics[("ALL", 'average_nMSE_variance')] = nmse_var_all
    all_metrics[("ALL", 'average_nMSE_variance_ci')] = nmse_var_ci_all
    all_metrics[("ALL", 'average_nMSE_range')] = nmse_rng_all
    all_metrics[("ALL", 'average_nMSE_range_ci')] = nmse_rng_ci_all

    return all_metrics

def similarities_Average_all(successful_df, simulated_dfs):
    """
    Compare each hypothesis simulation to real data and return metrics and LaTeX.
    """
    all_results = {}

    for hypothesis, sim_df in simulated_dfs.items():
        result = similarities_Average(successful_df, sim_df)
        for key, value in result.items():
            if key not in all_results:
                all_results[key] = {}
            all_results[key][hypothesis] = value

    return all_results, generate_latex_table_from_results(all_results)

def generate_latex_table_from_results(results_dict):
    """
    Generate LaTeX table showing similarity metrics, combining value and CI in one line,
    keeping previous structure, star notation, and bolding the best hypothesis.
    """
    hypotheses = ["H0", "H1", "H2", "H3","Hybridisation"]
    endowments = sorted({k[0] for k in results_dict if k[0] != "ALL"})

    metrics = [
        ("average_pearson", "$\\text{Pearson}~(r)$"),
        ("average_nMSE_variance", "$\\text{nMSE}_{\\text{var}}$"),
        ("average_nMSE_range", "$\\text{nMSE}_{\\text{range}}$")
    ]

    def find_best(endowment, metric):
        values = []
        for hypo in hypotheses:
            v = results_dict.get((endowment, metric), {}).get(hypo, None)
            if isinstance(v, float):
                values.append((hypo, v))
        if not values:
            return None
        if metric == "average_pearson":
            best = max(values, key=lambda x: x[1])  # Highest r is better
        else:
            best = min(values, key=lambda x: x[1])  # Lowest error is better
        return best[0]

    def get_value(endowment, metric, hypo, best_hypo):
        value = results_dict.get((endowment, metric), {}).get(hypo, "")
        if value == "":
            return ""

        if isinstance(value, float):
            if metric == "average_pearson":
                pval = results_dict.get((endowment, "average_pearson_p"), {}).get(hypo, 1.0)
                star = "*" if isinstance(pval, float) and pval < 0.05 else ""
            else:
                star = ""

            ci_key = metric + "_ci"
            ci = results_dict.get((endowment, ci_key), {}).get(hypo, None)

            if isinstance(ci, tuple) and all(isinstance(v, float) for v in ci):
                val_text = f"{value:.3f}{star} [{ci[0]:.2f}, {ci[1]:.2f}]"
            else:
                val_text = f"{value:.3f}{star}"

            # Bold if this is the best hypo
            if hypo == best_hypo:
                val_text = f"\\textbf{{{val_text}}}"
            return val_text

        return str(value)

    latex = []
    latex.append("\\begin{table}[bt!]")
    latex.append("\\centering")
    latex.append("\\caption{Similarities between real and simulated average contributions.}")
    latex.append("\\label{tab:macro_contributions}")
    latex.append("\\begin{tabular}{llccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Endowment} & \\textbf{Metric} & $H_0$ & $H_1$ & $H_2$ & $H_3$ & $Hybridisation$\\\\")
    latex.append("\\midrule")

    for endowment in endowments + ["ALL"]:
        for i, (metric_key, label) in enumerate(metrics):
            best_hypo = find_best(endowment, metric_key)
            row = []
            row.append(f"\\multirow{{{len(metrics)}}}{{*}}{{\\textit{{{endowment}}}}}" if i == 0 else "")
            row.append(label)
            row += [get_value(endowment, metric_key, h, best_hypo) for h in hypotheses]
            latex.append(" & ".join(row) + " \\\\")
        latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


#Similarity metrics for payoff distributions

def compute_payoff_similarity_metrics(real_df, simulated_dfs, bin_width=2):
    """
    Computes KL divergence, JS divergence, and MSE across quartiles for raw payoff distributions.
    """
    results = {}
    real_df = real_df.copy()
    real_df['payoff'] = real_df['endowment_current']  # Use raw payoff

    for hypo, sim_df in simulated_dfs.items():
        sim_df = sim_df.copy()
        sim_df['payoff'] = sim_df['endowment_current']

        for endowment in sorted(real_df['endowment_initial'].unique()):
            real_vals = real_df[real_df['endowment_initial'] == endowment]['payoff'].values
            sim_vals = sim_df[sim_df['endowment_initial'] == endowment]['payoff'].values

            min_val = min(real_vals.min(), sim_vals.min())
            max_val = max(real_vals.max(), sim_vals.max())
            bin_edges = np.arange(min_val, max_val + bin_width, bin_width)

            hist_real, _ = np.histogram(real_vals, bins=bin_edges, density=True)
            hist_sim, _ = np.histogram(sim_vals, bins=bin_edges, density=True)

            # Add epsilon to avoid 0
            epsilon = 1e-8
            hist_real += epsilon
            hist_sim += epsilon

            hist_real /= hist_real.sum()
            hist_sim /= hist_sim.sum()

            kl = entropy(hist_real, hist_sim)
            js = jensenshannon(hist_real, hist_sim) ** 2
            q_real = np.percentile(real_vals, [25, 50, 75])
            q_sim = np.percentile(sim_vals, [25, 50, 75])
            mse_q = np.mean((q_real - q_sim) ** 2)

            results[(endowment, 'KL', hypo)] = kl
            results[(endowment, 'JSD', hypo)] = js
            results[(endowment, 'QuartileMSE', hypo)] = mse_q

    return results

def generate_latex_similarity_table(results):
    """
    Generates a LaTeX table from payoff similarity results.
    """
    hypotheses = ['H0', 'H1', 'H2', 'H3','Hybridisation']
    metrics = ['KL', 'JSD', 'QuartileMSE']
    endowments = sorted({k[0] for k in results})

    table = []
    table.append("\\begin{table}[bt!]")
    table.append("\\centering")
    table.append("\\caption{Similarity between real and simulated payoff distributions.}")
    table.append("\\label{tab:payoff_similarity}")
    table.append("\\begin{tabular}{llccccc}")
    table.append("\\toprule")
    table.append("\\textbf{Endowment} & \\textbf{Metric} & $H_0$ & $H_1$ & $H_2$ & $H_3$ & $Hybridisation$ \\\\")
    table.append("\\midrule")

    for e in endowments:
        for m in metrics:
            row = [str(e), m]
            vals = [results.get((e, m, h), np.nan) for h in hypotheses]
            min_val = min(vals)
            for v in vals:
                if v == min_val:
                    row.append(f"\\textbf{{{v:.3f}}}")
                else:
                    row.append(f"{v:.3f}")
            table.append(" & ".join(row) + " \\\\")

    # Add global average row
    table.append("\\midrule")
    for m in metrics:
        row = ["\\textbf{All}", m]
        avg_vals = []
        for h in hypotheses:
            vals = [results.get((e, m, h), np.nan) for e in endowments]
            mean_val = np.mean(vals)
            avg_vals.append(mean_val)
        min_val = min(avg_vals)
        for v in avg_vals:
            if v == min_val:
                row.append(f"\\textbf{{{v:.3f}}}")
            else:
                row.append(f"{v:.3f}")
        table.append(" & ".join(row) + " \\\\")

    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    table.append("\\end{table}")
    return "\n".join(table)

# Similarity metrics for Lorenz curves

def similarities_Lorenz(df_real, df_simulated):
    """
    Compute similarity metrics between real and simulated Lorenz curves
    for final payoffs, grouped by control_wealth.
    Metrics: DTW, nMSE_variance (+CI), nMSE_range (+CI)
    """

    # Compute Lorenz Curves for Real
    lorenz_real = {}
    for cw, group in df_real.groupby("control_wealth"):
        curves = []
        for _, game in group.groupby("partida_id"):
            x = game["endowment_current"].values
            _, c = lorenz_curve(x)
            curves.append(c)
        lorenz_real[cw] = np.mean(curves, axis=0)  # mean Lorenz curve

    # Compute Lorenz Curves for Simulated 
    lorenz_sim = {}
    for cw, group in df_simulated.groupby("control_wealth"):
        curves = []
        for _, game in group.groupby("partida_id"):
            x = game["endowment_current"].values
            _, c = lorenz_curve(x)
            curves.append(c)
        lorenz_sim[cw] = np.mean(curves, axis=0)

    all_metrics = {}
    all_real, all_sim = [], []

    # Compute Similarities per Wealth
    for cw in lorenz_real.keys():

        y_real = lorenz_real[cw]
        y_sim = lorenz_sim[cw]

        # DTW
        dtw_dist, _ = fastdtw(y_real, y_sim, dist=lambda x, y: abs(x - y))

        # nMSE
        nmse_var = nMSE_variance(y_real, y_sim)
        nmse_rng = nMSE_range(y_real, y_sim)

        # Confidence Intervals (bootstrap)
        nmse_var_ci = bootstrap_nMSE_variance_ci(y_real, y_sim)
        nmse_rng_ci = bootstrap_nMSE_range_ci(y_real, y_sim)

        all_metrics[(cw, "lorenz_dtw")] = dtw_dist
        all_metrics[(cw, "lorenz_nMSE_variance")] = nmse_var
        all_metrics[(cw, "lorenz_nMSE_variance_ci")] = nmse_var_ci
        all_metrics[(cw, "lorenz_nMSE_range")] = nmse_rng
        all_metrics[(cw, "lorenz_nMSE_range_ci")] = nmse_rng_ci

        # Store for ALL
        all_real.extend(y_real)
        all_sim.extend(y_sim)

    # Compute Global "ALL" metrics
    y_real_all = np.array(all_real)
    y_sim_all = np.array(all_sim)

    dtw_all, _ = fastdtw(y_real_all, y_sim_all, dist=lambda x, y: abs(x - y))
    nmse_var_all = nMSE_variance(y_real_all, y_sim_all)
    nmse_rng_all = nMSE_range(y_real_all, y_sim_all)

    nmse_var_ci_all = bootstrap_nMSE_variance_ci(y_real_all, y_sim_all)
    nmse_rng_ci_all = bootstrap_nMSE_range_ci(y_real_all, y_sim_all)

    all_metrics[("ALL", "lorenz_dtw")] = dtw_all
    all_metrics[("ALL", "lorenz_nMSE_variance")] = nmse_var_all
    all_metrics[("ALL", "lorenz_nMSE_variance_ci")] = nmse_var_ci_all
    all_metrics[("ALL", "lorenz_nMSE_range")] = nmse_rng_all
    all_metrics[("ALL", "lorenz_nMSE_range_ci")] = nmse_rng_ci_all

    return all_metrics

def similarities_Lorenz_all(successful_df, simulated_dfs_success):
    """
    Apply similarities_Lorenz for each hypothesis and combine results into dict.
    """
    all_results = {}

    for hypothesis, sim_df in simulated_dfs_success.items():
        result = similarities_Lorenz(successful_df, sim_df)
        for key, value in result.items():
            if key not in all_results:
                all_results[key] = {}
            all_results[key][hypothesis] = value

    # You can reuse your previous LaTeX generator, 
    # or create a specific one if you want separate tables for Lorenz metrics.
    return all_results, generate_latex_table_lorenz_dtwmse(all_results)

def generate_latex_table_lorenz_dtwmse(results_dict):
    """
    Generate LaTeX table showing similarity metrics (DTW and nMSE) for Lorenz curves,
    combining value and CI in one line, keeping formatting, and bolding best results.
    """
    hypotheses = ["H0", "H1", "H2", "H3", "Hybridisation"]
    endowments = sorted({k[0] for k in results_dict if k[0] != "ALL"})

    metrics = [
        ("lorenz_dtw", "$\\text{DTW}$"),
        ("lorenz_nMSE_variance", "$\\text{nMSE}_{\\text{var}}$"),
        ("lorenz_nMSE_range", "$\\text{nMSE}_{\\text{range}}$")
    ]

    def find_best(endowment, metric):
        values = []
        for hypo in hypotheses:
            v = results_dict.get((endowment, metric), {}).get(hypo, None)
            if isinstance(v, float):
                values.append((hypo, v))
        if not values:
            return None
        return min(values, key=lambda x: x[1])[0]  # Smaller is always better

    def get_value(endowment, metric, hypo, best_hypo):
        value = results_dict.get((endowment, metric), {}).get(hypo, "")
        if value == "":
            return ""

        if isinstance(value, float):
            star = ""

            ci_key = metric + "_ci"
            ci = results_dict.get((endowment, ci_key), {}).get(hypo, None)

            if isinstance(ci, tuple) and all(isinstance(v, float) for v in ci):
                val_text = f"{value:.4f}{star} [{ci[0]:.2f}, {ci[1]:.2f}]"
            else:
                val_text = f"{value:.4f}{star}"

            # Bold if this is the best hypo
            if hypo == best_hypo:
                val_text = f"\\textbf{{{val_text}}}"
            return val_text

        return str(value)

    latex = []
    latex.append("\\begin{table}[bt!]")
    latex.append("\\centering")
    latex.append("\\caption{Similarities between real and simulated Lorenz curves.}")
    latex.append("\\label{tab:lorenz_metrics}")
    latex.append("\\begin{tabular}{llccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Wealth treatment} & \\textbf{Metric} & $H_0$ & $H_1$ & $H_2$ & $H_3$ & $Hybridisation$\\\\")
    latex.append("\\midrule")

    for endowment in endowments + ["ALL"]:
        for i, (metric_key, label) in enumerate(metrics):
            best_hypo = find_best(endowment, metric_key)
            row = []
            row.append(f"\\multirow{{{len(metrics)}}}{{*}}{{\\textit{{{endowment}}}}}" if i == 0 else "")
            row.append(label)
            row += [get_value(endowment, metric_key, h, best_hypo) for h in hypotheses]
            latex.append(" & ".join(row) + " \\\\")
        latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)

#Similarities metrics for common funds

def similarities_Fund_DTWMSE(df_real, df_simulated):
    def compute_cumulative(df):
        cumulative = []
        for partida_id, group in df.groupby('partida_id'):
            cum_sum = np.cumsum(group[round_columns].sum().values)
            cumulative.append({
                "control_wealth": group["control_wealth"].iloc[0],
                **{f"R{i+1}": cum_sum[i] for i in range(len(cum_sum))}
            })
        return pd.DataFrame(cumulative)

    cum_real = compute_cumulative(df_real)
    cum_sim = compute_cumulative(df_simulated)

    mean_real = cum_real.groupby('control_wealth').mean()
    mean_sim = cum_sim.groupby('control_wealth').mean()

    results = {}
    all_real, all_sim = [], []

    for wealth_type in mean_real.index:
        y_real = mean_real.loc[wealth_type, round_columns].values
        y_sim = mean_sim.loc[wealth_type, round_columns].values

        dtw_dist, _ = fastdtw(y_real, y_sim, dist=lambda x, y: abs(x - y))
        nmse_var = nMSE_variance(y_real, y_sim)
        nmse_rng = nMSE_range(y_real, y_sim)

        nmse_var_ci = bootstrap_nMSE_variance_ci(y_real, y_sim)
        nmse_rng_ci = bootstrap_nMSE_range_ci(y_real, y_sim)

        results[(wealth_type, 'fund_dtw')] = dtw_dist
        results[(wealth_type, 'fund_nMSE_variance')] = nmse_var
        results[(wealth_type, 'fund_nMSE_range')] = nmse_rng
        results[(wealth_type, 'fund_nMSE_variance_ci')] = nmse_var_ci
        results[(wealth_type, 'fund_nMSE_range_ci')] = nmse_rng_ci

        all_real.append(y_real)
        all_sim.append(y_sim)

    # Global ALL
    y_real_all = np.concatenate(all_real)
    y_sim_all = np.concatenate(all_sim)

    dtw_all, _ = fastdtw(y_real_all, y_sim_all, dist=lambda x, y: abs(x - y))
    nmse_var_all = nMSE_variance(y_real_all, y_sim_all)
    nmse_rng_all = nMSE_range(y_real_all, y_sim_all)
    nmse_var_ci_all = bootstrap_nMSE_variance_ci(y_real_all, y_sim_all)
    nmse_rng_ci_all = bootstrap_nMSE_range_ci(y_real_all, y_sim_all)

    results[('ALL', 'fund_dtw')] = dtw_all
    results[('ALL', 'fund_nMSE_variance')] = nmse_var_all
    results[('ALL', 'fund_nMSE_range')] = nmse_rng_all
    results[('ALL', 'fund_nMSE_variance_ci')] = nmse_var_ci_all
    results[('ALL', 'fund_nMSE_range_ci')] = nmse_rng_ci_all

    return results

def similarities_Fund_DTWMSE_all(df_real, simulated_dfs_success):
    all_results = {}
    for hypo, sim_df in simulated_dfs_success.items():
        result = similarities_Fund_DTWMSE(df_real, sim_df)
        for key, val in result.items():
            if key not in all_results:
                all_results[key] = {}
            all_results[key][hypo] = val
    return all_results

def generate_latex_table_fund_dtwmse(results_dict):
    """
    Generate LaTeX table showing similarity metrics (DTW and nMSE) for common fund,
    combining value and CI in one line, keeping formatting, and bolding best results.
    """
    hypotheses = ["H0", "H1", "H2", "H3", "Hybridisation"]
    endowments = sorted({k[0] for k in results_dict if k[0] != "ALL"})

    metrics = [
        ("fund_dtw", "$\\text{DTW}$"),
        ("fund_nMSE_variance", "$\\text{nMSE}_{\\text{var}}$"),
        ("fund_nMSE_range", "$\\text{nMSE}_{\\text{range}}$")
    ]

    def find_best(endowment, metric):
        values = []
        for hypo in hypotheses:
            v = results_dict.get((endowment, metric), {}).get(hypo, None)
            if isinstance(v, float):
                values.append((hypo, v))
        if not values:
            return None
        return min(values, key=lambda x: x[1])[0]  # Lower is better

    def get_value(endowment, metric, hypo, best_hypo):
        value = results_dict.get((endowment, metric), {}).get(hypo, "")
        if value == "":
            return ""

        if isinstance(value, float):
            ci_key = metric + "_ci"
            ci = results_dict.get((endowment, ci_key), {}).get(hypo, None)

            if isinstance(ci, tuple) and all(isinstance(v, float) for v in ci):
                val_text = f"{value:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"
            else:
                val_text = f"{value:.3f}"

            if hypo == best_hypo:
                val_text = f"\\textbf{{{val_text}}}"
            return val_text

        return str(value)

    latex = []
    latex.append("\\begin{table}[bt!]")
    latex.append("\\centering")
    latex.append("\\caption{Similarities between real and simulated common funds.}")
    latex.append("\\label{tab:fund_dtw_nMSE}")
    latex.append("\\begin{tabular}{llccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Wealth treatment} & \\textbf{Metric} & $H_0$ & $H_1$ & $H_2$ & $H_3$ & $Hybridisation$\\\\")
    latex.append("\\midrule")

    for endowment in endowments + ["ALL"]:
        for i, (metric_key, label) in enumerate(metrics):
            best_hypo = find_best(endowment, metric_key)
            row = []
            row.append(f"\\multirow{{{len(metrics)}}}{{*}}{{\\textit{{{endowment}}}}}" if i == 0 else "")
            row.append(label)
            row += [get_value(endowment, metric_key, h, best_hypo) for h in hypotheses]
            latex.append(" & ".join(row) + " \\\\")
        latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)