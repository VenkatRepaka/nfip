"""Shared settings for the NFIP project notebooks.

Everything here is also explained in the notebooks. It lives in one file so that every
research question uses exactly the same feature lists and the same train/test split.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path('/Users/home/workspace/School/DataMining/nfip_claims')
RAW_DIR = PROJECT_DIR / 'data_raw'
PROCESSED_DIR = PROJECT_DIR / 'data_processed'
TABLE_DIR = PROJECT_DIR / 'tables'
FIGURE_DIR = PROJECT_DIR / 'figures'
SEED = 42

TRAIN_YEARS = [2020, 2021, 2022, 2023]      # models learn from these loss years
TEST_YEARS = [2024, 2025]                   # and are judged on these (Helene, Milton, Debby, Francine ...)

# ---------------------------------------------------------------------------
# Feature sets. Only facts known when a claim is filed after a known storm.
# ---------------------------------------------------------------------------
NFIP_NUMERIC_FEATURES = [
    'log_building_coverage', 'log_contents_coverage', 'has_contents_coverage', 'building_deductible_usd',
    'elevationDifference', 'has_elevation_certificate_data', 'building_age_at_loss', 'construction_year_was_missing',
    'policy_tenure_years', 'crsClassCode', 'numberOfUnits', 'policyCount',
    'elevatedBuildingIndicator', 'postFIRMConstructionIndicator', 'primaryResidenceIndicator',
    'rentalPropertyIndicator', 'floodproofedIndicator', 'in_special_flood_hazard_area', 'rated_by_risk_rating_2',
    'latitude', 'longitude', 'loss_month', 'hurricane_season',
]
NFIP_CATEGORICAL_FEATURES = [
    'flood_zone_group', 'occupancy_group', 'foundation_group', 'floors_group', 'basement_group',
    'condominiumCoverageTypeCode', 'construction_era', 'state',
]
EVENT_FEATURES = [               # PRISM rain + HURDAT2 storm + coast distance
    'rain_loss_day_mm', 'rain_3day_mm', 'rain_7day_mm', 'rain_max_daily_7day_mm',
    'storm_distance_km', 'storm_wind_at_closest_kt', 'storm_lifetime_max_wind_kt', 'storm_category_at_closest',
    'storm_wind_proxy_at_claim_kt', 'no_storm_within_window', 'storm_within_200km', 'coast_distance_km',
]
CONTEXT_FEATURES = [             # ACS neighbourhood + FEMA National Risk Index (county)
    'acs_median_household_income', 'acs_median_home_value', 'acs_pct_owner_occupied', 'acs_median_year_built',
    'acs_pct_below_poverty', 'acs_pct_non_hispanic_white', 'acs_population',
    'nri_overall_risk_score', 'nri_social_vulnerability_score', 'nri_community_resilience_score',
    'nri_coastal_flood_risk_score', 'nri_inland_flood_risk_score', 'nri_hurricane_risk_score',
    'nri_coastal_flood_annual_frequency', 'nri_inland_flood_annual_frequency',
]
FEATURE_SETS = {
    'A. NFIP only': (NFIP_NUMERIC_FEATURES, NFIP_CATEGORICAL_FEATURES),
    'B. NFIP + event (PRISM, HURDAT2, coast)': (NFIP_NUMERIC_FEATURES + EVENT_FEATURES, NFIP_CATEGORICAL_FEATURES),
    'C. NFIP + event + context (ACS, NRI)': (NFIP_NUMERIC_FEATURES + EVENT_FEATURES + CONTEXT_FEATURES, NFIP_CATEGORICAL_FEATURES),
}

# Feature blocks of the outside data, used for the ablation in notebooks 03 and 04
# ("which outside source makes the gain?"). Together they are exactly EVENT_FEATURES.
RAIN_FEATURES = ['rain_loss_day_mm', 'rain_3day_mm', 'rain_7day_mm', 'rain_max_daily_7day_mm']
STORM_FEATURES = ['storm_distance_km', 'storm_wind_at_closest_kt', 'storm_lifetime_max_wind_kt', 'storm_category_at_closest',
                  'storm_wind_proxy_at_claim_kt', 'no_storm_within_window', 'storm_within_200km']
COAST_FEATURES = ['coast_distance_km']
assert sorted(RAIN_FEATURES + STORM_FEATURES + COAST_FEATURES) == sorted(EVENT_FEATURES)

LEAKING_COLUMNS = [
    'amountPaidOnBuildingClaim', 'netBuildingPaymentAmount', 'buildingDamageAmount', 'contentsDamageAmount',
    'amountPaidOnContentsClaim', 'netContentsPaymentAmount', 'amountPaidOnIncreasedCostOfComplianceClaim',
    'netIccPaymentAmount', 'nonPaymentReasonBuilding', 'nonPaymentReasonContents', 'buildingPropertyValue',
    'contentsPropertyValue', 'waterDepth', 'exteriorWaterDepth', 'interiorWaterDepth', 'causeOfDamage',
    'openDate', 'mostRecentPaymentDate', 'floodZoneCurrent', 'flag_payment_above_damage',
    'building_paid_flag', 'large_loss_flag', 'damage_to_value_ratio', 'substantial_damage_flag',
    # Hidden leak found in notebook 01, Section 9.2: replacement cost is filled in during claim adjustment,
    # so whether it is missing (and the value we impute) tells the model if the claim was inspected/paid.
    'buildingReplacementCost', 'log_replacement_cost', 'coverage_to_replacement_ratio', 'replacement_cost_was_missing',
]


def load_clean_claims():
    claims = pd.read_parquet(PROCESSED_DIR / 'claims_clean_2020_2025.parquet')
    claims['event_group'] = claims['floodEvent'].fillna('no_named_event_' + claims['loss_year'].astype(str))
    return claims


def check_no_leakage(feature_list):
    """Stop the notebook if a leaking column slipped into a feature list."""
    leaked = sorted(set(feature_list) & set(LEAKING_COLUMNS))
    if leaked:
        raise ValueError(f'LEAKAGE: these columns must not be features: {leaked}')
    return True


def make_preprocessor(numeric_features, categorical_features, scale=True):
    """Impute -> (scale) numeric columns; one-hot encode categoricals (rare levels pooled).
    It is always fitted on the training rows only, inside a Pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        numeric_steps.append(('scale', StandardScaler()))
    categorical_steps = [('impute', SimpleImputer(strategy='constant', fill_value='missing')),
                         ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=300,
                                                  sparse_output=False))]
    return ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), numeric_features),
        ('categorical', Pipeline(categorical_steps), categorical_features),
    ], verbose_feature_names_out=False)


def to_category_frame(data, numeric_features, categorical_features):
    """LightGBM can use pandas 'category' columns directly."""
    frame = data[numeric_features + categorical_features].copy()
    for column in categorical_features:
        frame[column] = frame[column].astype(str).astype('category')
    return frame


def lift_at_top(y_true, scores, top_share=0.10):
    """How many times more positives are in the top x% of scores than on average."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(scores))
    n_top = max(1, int(len(order) * top_share))
    return y_true[order[:n_top]].mean() / y_true.mean()


def event_block_bootstrap(data, truth_column, score_columns, metric_functions, baseline,
                          group_column='event_group', n_repeats=500, seed=SEED):
    """95% confidence interval for the GAIN of each model over a baseline model, resampling whole events.

    Claims from one storm are not independent (one storm = thousands of similar claims), so we do not resample
    single claims. We draw events with replacement (a block bootstrap), recompute every metric on the drawn
    claims, and keep the difference "model - baseline". The 2.5th and 97.5th percentiles give the 95% CI.

    data             : one row per claim (the test set)
    truth_column     : the true label
    score_columns    : {model name: column holding that model's score (or predicted label)}
    metric_functions : {metric name: function(y_true, score) -> number}
    baseline         : the model name we compare against (for example 'A. NFIP only')
    Returns one row per (metric, model): point value, gain over the baseline, 95% CI of the gain,
    and the share of repeats where the gain was above 0.
    """
    rng = np.random.default_rng(seed)
    data = data.reset_index(drop=True)
    rows_by_event = list(data.groupby(group_column).indices.values())
    truth = data[truth_column].to_numpy()
    scores = {name: data[column].to_numpy() for name, column in score_columns.items()}
    point_values = {(metric, name): function(truth, scores[name])
                    for metric, function in metric_functions.items() for name in scores}
    gains = {key: [] for key in point_values if key[1] != baseline}
    valid_repeats = 0
    for _ in range(n_repeats):
        chosen_events = rng.integers(0, len(rows_by_event), len(rows_by_event))
        index = np.concatenate([rows_by_event[i] for i in chosen_events])
        y_resampled = truth[index]
        if len(np.unique(y_resampled)) < 2:
            continue
        valid_repeats += 1
        for metric, function in metric_functions.items():
            baseline_value = function(y_resampled, scores[baseline][index])
            for name in scores:
                if name != baseline:
                    gains[(metric, name)].append(function(y_resampled, scores[name][index]) - baseline_value)
    rows = []
    for (metric, name), value in point_values.items():
        row = {'metric': metric, 'model': name, 'value': value}
        if name != baseline:
            gain_draws = np.array(gains[(metric, name)])
            row.update({'gain_vs_baseline': value - point_values[(metric, baseline)],
                        'ci_95_low': np.quantile(gain_draws, 0.025), 'ci_95_high': np.quantile(gain_draws, 0.975),
                        'share_of_repeats_with_gain_%': (gain_draws > 0).mean() * 100})
        rows.append(row)
    table = pd.DataFrame(rows)
    table['events'] = len(rows_by_event)
    table['valid_repeats'] = valid_repeats
    return table


def show_answer_numbers(rows, notebook_prefix):
    """Print the key numbers of a notebook as one table and save them as tables/<prefix>_answer_numbers.csv.
    rows: list of (quantity, value, source table). The Markdown answer below the table uses these numbers."""
    table = pd.DataFrame(rows, columns=['quantity', 'value', 'source'])
    table['value'] = [round(v, 4) if isinstance(v, (float, np.floating)) else v for v in table['value']]
    table.to_csv(TABLE_DIR / f'{notebook_prefix}_answer_numbers.csv', index=False)
    with pd.option_context('display.max_colwidth', 90, 'display.width', 220):
        print(table.to_string(index=False))
    return table
