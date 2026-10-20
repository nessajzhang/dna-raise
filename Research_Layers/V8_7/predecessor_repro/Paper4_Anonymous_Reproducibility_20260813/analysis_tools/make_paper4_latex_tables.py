#!/usr/bin/env python3
"""Generate every numerical manuscript-table row from audited CSV outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("% Generated from audited CSV; do not edit by hand.\n" + "\n".join(rows) + "\n", encoding="utf-8")


def pct(value: str, digits: int = 3) -> str:
    return f"{100 * float(value):.{digits}f}"


def build(data: Path, output: Path) -> None:
    policy_names = {
        "fixed_schedule": "Fixed schedule",
        "oracle_true_hazard": "Full-information oracle",
        "posterior_mean": "Posterior mean",
        "posterior_conservative": "Conservative posterior",
    }
    seed = read(data / "paper4_archive_maintenance_seed_summary.csv")
    seed.sort(key=lambda row: (float(row["true_hazard"]), list(policy_names).index(row["policy"])))
    write(
        output / "main_seed_rows.tex",
        [
            f'{float(r["true_hazard"]):.3f} & {policy_names[r["policy"]]} & '
            f'{float(r["mean_refreshes"]):.3f} & {float(r["violation_rate"]):.6f} & '
            f'{float(r["episode_failure_rate"]):.4f} \\\\'
            for r in seed
        ],
    )

    qpcr = read(data / "paper4_dt4dds_qpcr_decay_summary.csv")
    loo = {r["holdout_pool"]: r for r in read(data / "paper4_dt4dds_qpcr_leave_one_pool_out.csv")}
    write(
        output / "main_qpcr_rows.tex",
        [
            f'{r["pool"]} & {float(r["half_life_days"]):.3f} & '
            f'{float(r["half_life_technical_bootstrap_ci95_low_days"]):.3f}--'
            f'{float(r["half_life_technical_bootstrap_ci95_high_days"]):.3f} & '
            f'{float(r["decay_r2_uncentered"]):.3f} & '
            f'{float(loo[r["pool"]]["holdout_rmse_log10"]):.3f} \\\\'
            for r in qpcr
        ],
    )

    sensitivity = read(data / "paper4_policy_sensitivity_summary.csv")
    def select(**criteria: object) -> dict[str, str]:
        matches = [
            row
            for row in sensitivity
            if all(str(row[key]) == str(value) for key, value in criteria.items())
        ]
        if len(matches) != 1:
            raise AssertionError((criteria, len(matches)))
        return matches[0]

    main_specs = [
        ("Broad prior", dict(sweep="design", policy="posterior_conservative", true_hazard="0.006", prior="broad", sentinel_size="300", inspection_interval="5")),
        ("20\\% under-detection, acceleration", dict(sweep="misspecification", policy="posterior_conservative", true_hazard="0.006", hazard_profile="accelerating", observation_factor="0.8")),
        ("Broad prior", dict(sweep="design", policy="posterior_conservative", true_hazard="0.012", prior="broad", sentinel_size="300", inspection_interval="5")),
        ("Low-biased prior", dict(sweep="design", policy="posterior_conservative", true_hazard="0.012", prior="low_biased", sentinel_size="300", inspection_interval="5")),
        ("Ten-year inspection", dict(sweep="design", policy="posterior_conservative", true_hazard="0.012", prior="broad", sentinel_size="300", inspection_interval="10")),
        ("Sentinel panel 100", dict(sweep="design", policy="posterior_conservative", true_hazard="0.012", prior="broad", sentinel_size="100", inspection_interval="5")),
        ("Sentinel panel 1,000", dict(sweep="design", policy="posterior_conservative", true_hazard="0.012", prior="broad", sentinel_size="1000", inspection_interval="5")),
    ]
    main_rows = []
    for label, spec in main_specs:
        r = select(**spec)
        main_rows.append(
            f'{float(r["true_hazard"]):.3f} & {label} & {int(r["sentinel_size"]):,} & '
            f'{int(r["inspection_interval"])} & {float(r["mean_refreshes"]):.3f} & '
            f'{float(r["violation_rate"]):.6f} \\\\'
        )
    write(output / "main_robustness_rows.tex", main_rows)

    finite = read(data / "paper4_finite_horizon_certificate_summary.csv")
    finite_order = [r["scenario"] for r in finite]
    labels = {
        "exact_constant_h0002": "Constant 0.002",
        "exact_constant_h0006": "Constant 0.006",
        "exact_constant_h0012": "Constant 0.012",
        "naive_acceleration_and_underdetection_high_information": "Biased acceleration, naive",
        "enveloped_acceleration_and_underdetection_high_information": "Biased acceleration, tight",
        "enveloped_acceleration_and_underdetection_high_information_coarse": "Biased acceleration, relaxed",
        "envelope_cost_under_constant_hazard_high_information": "Robust constant, tight",
        "envelope_cost_under_constant_hazard_high_information_coarse": "Robust constant, relaxed",
        "ten_year_schedule_gap": "Ten-year cycle-entry gap",
    }
    main_finite, si_finite, si_finite_count_ratio = [], [], []
    for name in finite_order:
        r = next(row for row in finite if row["scenario"] == name)
        episodes = int(r["episodes"])
        fs_ep = round(float(r["episode_false_safe_rate"]) * episodes)
        ci = (
            f'{pct(r["episode_false_safe_rate_ci95_low"], 3)}--'
            f'{pct(r["episode_false_safe_rate_ci95_high"], 3)}'
        )
        fs_text = f'{fs_ep:,}/{episodes:,} ({pct(r["episode_false_safe_rate"], 3)}\\%; {ci}\\%)'
        if fs_ep == 0:
            fs_text = f'0/{episodes:,} (0--{pct(r["episode_false_safe_rate_ci95_high"], 4)}\\%)'
        envelope = "valid" if r["model_envelope_valid"] == "True" else "invalid"
        main_finite.append(
            f'{labels[name]} & {envelope} & {int(r["no_refresh_decisions"]):,} & '
            f'{fs_text} & {float(r["mean_refreshes"]):.3f} & '
            f'{pct(r["episode_violation_rate"], 3)}\\% \\\\'
        )
        si_finite.append(
            f'{labels[name]} & {envelope} & {int(r["no_refresh_decisions"]):,} & '
            f'{int(r["false_safe_decisions"]):,} & {fs_text} & '
            f'{pct(r["episode_look_bound_failure_rate"], 2)}\\% & '
            f'{float(r["mean_refreshes"]):.3f} \\\\'
        )
        if int(r["no_refresh_decisions"]) == 0:
            ratio_text = "undefined (no admitted decisions)"
        else:
            ratio_text = (
                f'{pct(r["decision_count_ratio"], 3)}\\% '
                f'[{pct(r["decision_count_ratio_ci95_low"], 3)}, '
                f'{pct(r["decision_count_ratio_ci95_high"], 3)}]'
            )
        si_finite_count_ratio.append(
            f'{labels[name]} & {int(r["no_refresh_decisions"]):,} & '
            f'{int(r["false_safe_decisions"]):,} & {ratio_text} \\\\'
        )
    write(output / "main_finite_rows.tex", main_finite)
    write(output / "si_finite_rows.tex", si_finite)
    write(output / "si_finite_count_ratio_rows.tex", si_finite_count_ratio)

    periodic = read(data / "paper4_periodic_schedule_frontier_summary.csv")
    write(
        output / "si_periodic_rows.tex",
        [f'{float(r["true_hazard"]):.3f} & {r["minimum_refreshes_among_enumerated_periodic_schedules"]} & {r["largest_interval_at_minimum_refreshes"]} \\\\' for r in periodic],
    )

    design_specs = [
        dict(true_hazard="0.006", prior="broad", sentinel_size="300", inspection_interval="5"),
        dict(true_hazard="0.006", prior="low_biased", sentinel_size="300", inspection_interval="5"),
        dict(true_hazard="0.006", prior="high_biased", sentinel_size="300", inspection_interval="5"),
        dict(true_hazard="0.012", prior="broad", sentinel_size="300", inspection_interval="5"),
        dict(true_hazard="0.012", prior="low_biased", sentinel_size="300", inspection_interval="5"),
        dict(true_hazard="0.012", prior="high_biased", sentinel_size="300", inspection_interval="5"),
        dict(true_hazard="0.012", prior="broad", sentinel_size="300", inspection_interval="2"),
        dict(true_hazard="0.012", prior="broad", sentinel_size="300", inspection_interval="10"),
        dict(true_hazard="0.012", prior="broad", sentinel_size="100", inspection_interval="5"),
        dict(true_hazard="0.012", prior="broad", sentinel_size="1000", inspection_interval="5"),
    ]
    prior_names = {"broad": "broad", "low_biased": "low-biased", "high_biased": "high-biased"}
    rows = []
    for spec in design_specs:
        r = select(sweep="design", policy="posterior_conservative", **spec)
        rows.append(
            f'{float(r["true_hazard"]):.3f} & {prior_names[r["prior"]]} & {int(r["sentinel_size"]):,} & '
            f'{r["inspection_interval"]} & {float(r["mean_refreshes"]):.3f} & '
            f'{float(r["violation_rate"]):.6f} & {float(r["episode_failure_rate"]):.3f} \\\\'
        )
    write(output / "si_design_rows.tex", rows)

    misspec_specs = [
        ("posterior_conservative", "constant", "0.006", "0.8"),
        ("posterior_conservative", "constant", "0.006", "1.0"),
        ("posterior_conservative", "accelerating", "0.006", "0.8"),
        ("posterior_conservative", "accelerating", "0.006", "1.0"),
        ("posterior_mean", "constant", "0.006", "0.8"),
        ("posterior_mean", "accelerating", "0.006", "0.8"),
        ("posterior_conservative", "constant", "0.012", "0.8"),
        ("posterior_conservative", "accelerating", "0.012", "0.8"),
        ("posterior_mean", "constant", "0.012", "1.0"),
        ("posterior_mean", "accelerating", "0.012", "1.0"),
    ]
    rows = []
    for pol, profile, hazard, detection in misspec_specs:
        r = select(sweep="misspecification", policy=pol, true_hazard=hazard, hazard_profile=profile, observation_factor=detection)
        rows.append(
            f'{policy_names[pol]} & {profile} & {float(hazard):.3f} & {float(detection):.1f} & '
            f'{float(r["mean_refreshes"]):.3f} & {float(r["violation_rate"]):.6f} & '
            f'{float(r["episode_failure_rate"]):.3f} \\\\'
        )
    write(output / "si_misspec_rows.tex", rows)

    decision = read(data / "paper4_decision_calibration_summary.csv")
    decision_labels = {
        "calibrated_prior_predictive": "Prior predictive",
        "calibrated_fixed_hazard": "Fixed hazard 0.006",
        "prior_support_exclusion": "Prior exclusion",
        "biased_accelerating_hazard": "Biased acceleration",
        "ten_year_inspection": "Ten-year inspection",
    }
    rows = []
    for r in decision:
        if int(r["no_refresh_decisions"]) == 0:
            rate = "not defined"
        else:
            rate = f'{pct(r["decision_count_ratio"], 4)}\\% ({pct(r["decision_count_ratio_ci95_low"], 4)}--{pct(r["decision_count_ratio_ci95_high"], 4)})'
        rows.append(
            f'{decision_labels[r["scenario"]]} & {int(r["no_refresh_decisions"]):,} & '
            f'{int(r["false_safe_decisions"]):,} & {rate} & {pct(r["episode_false_safe_rate"], 2)}\\% \\\\'
        )
    write(output / "si_decision_rows.tex", rows)

    alpha = read(data / "paper4_alpha_spending_design_summary.csv")
    phase_order = {"design": 0, "evaluation": 1}
    alpha.sort(key=lambda r: (phase_order[r["phase"]], float(r["gamma"])))
    rows = []
    for r in alpha:
        phase = "Design" if r["phase"] == "design" else "Evaluation"
        if r["selected_on_design"] == "1":
            phase += ", selected"
        elif r["phase"] == "evaluation" and float(r["gamma"]) == 0.0:
            phase += ", equal"
        episodes = int(r["episodes"])
        rows.append(
            f'{phase} & ${float(r["gamma"]):.1f}$ & {episodes:,} & '
            f'{float(r["mean_refreshes"]):.3f} & 0/{episodes:,} '
            f'(0--{pct(r["episode_false_safe_rate_ci95_high"], 4)}\\%) \\\\'
        )
    write(output / "si_alpha_rows.tex", rows)

    mission = read(data / "paper4_mission_risk_design.csv")
    mission_labels = {
        "current_per_endpoint_target": "Current per-endpoint target",
        "alpha_0.05_plus_beta_0.05": "$\\alpha=0.05$, $\\beta=0.05$",
        "balanced_total_mission_bound_0.05": "Balanced 5\\% ideal-model bound",
    }
    write(
        output / "si_mission_rows.tex",
        [f'{mission_labels[r["design"]]} & {float(r["false_safe_budget_alpha"]):.3f} & {float(r["residual_random_failure_budget_beta"]):.3f} & {float(r["required_per_endpoint_recoverability"]):.6f} & {float(r["coarse_ideal_model_mission_failure_upper_bound"]):.3f} \\\\' for r in mission],
    )
    cycle = read(data / "paper4_cycle_entry_design.csv")
    write(
        output / "si_cycle_rows.tex",
        [f'{float(r["post_refresh_residual_dropout"]):.2f} & {float(r["hazard_ceiling"]):.3f} & {float(r["acceleration_ceiling"]):.2f} & {float(r["maximum_cycle_entry_interval_years"]):.3f} \\\\' for r in cycle],
    )
    dependence = read(data / "paper4_erasure_dependence_stress.csv")
    rho_label = lambda value: "0" if float(value) == 0 else f'$10^{{{int(round(np_log10(float(value))))}}}$'
    write(
        output / "si_dependence_rows.tex",
        [f'{rho_label(r["intraclass_correlation"])} & {float(r["critical_mean_dropout"]):.6f} & {float(r["critical_dropout_relative_to_independent"]):.4f} \\\\' for r in dependence],
    )

    extended = [
        row
        for row in read(data / "paper4_extended_policy_summary.csv")
        if row["family"] == "common_design_policy"
        and abs(float(row["true_hazard_per_year"]) - 0.009) < 1e-12
    ]
    extended_labels = {
        "fixed_5y": "Fixed 5 y",
        "fixed_10y": "Fixed 10 y",
        "naive_plugin": "Naive plug-in",
        "beta_sentinel_q95": "Beta(1,1) sentinel 95\\%",
        "certificate_independent": "Independent certificate",
        "certificate_robust": "Fixed-ICC beta-binomial",
        "oracle_service_model": "Service-model oracle",
        "posterior_hazard_q95": "Hazard-posterior 95\\% rule",
    }
    extended_order = list(extended_labels)
    extended.sort(key=lambda row: extended_order.index(row["policy"]))
    validity = {"yes": "Yes", "no": "No", "not applicable": "N/A"}
    extended_rows = []
    for row in extended:
        episodes = int(row["episodes"])
        events = round(float(row["episode_unsafe_deferral_rate"]) * episodes)
        if events == 0:
            rate = (
                f'0/{episodes:,} (0--'
                f'{pct(row["episode_unsafe_deferral_rate_ci95_high"], 4)}\\%)'
            )
        else:
            rate = (
                f'{events:,}/{episodes:,} '
                f'({pct(row["episode_unsafe_deferral_rate"], 3)}\\%)'
            )
        event = "Under-target authorization"
        extended_rows.append(
            f'{extended_labels[row["policy"]]} & '
            f'{validity[row["observation_envelope_valid"]]} & '
            f'{validity[row["aging_envelope_valid"]]} & '
            f'{validity[row["service_envelope_valid"]]} & {event} & {rate} & '
            f'{float(row["mean_refresh_attempts"]):.3f} \\\\'
        )
    write(output / "si_common_design_rows.tex", extended_rows)

    observation_dependence = [
        row
        for row in read(data / "paper4_extended_policy_summary.csv")
        if row["family"] == "observation_dependence"
    ]
    observation_labels = {
        "certificate_robust": "Binomial-observation certificate",
        "certificate_observation_robust": "Beta-binomial-observation certificate",
    }
    observation_dependence.sort(
        key=lambda row: (
            float(row["actual_observation_rho"]),
            list(observation_labels).index(row["policy"]),
        )
    )
    observation_rows = []
    for row in observation_dependence:
        episodes = int(row["episodes"])
        events = round(float(row["episode_unsafe_deferral_rate"]) * episodes)
        rate = (
            f'0/{episodes:,} (0--{pct(row["episode_unsafe_deferral_rate_ci95_high"], 4)}\\%)'
            if events == 0
            else f'{events:,}/{episodes:,} ({pct(row["episode_unsafe_deferral_rate"], 3)}\\%)'
        )
        rho = float(row["actual_observation_rho"])
        rho_text = "0" if rho == 0.0 else f'$10^{{{int(round(np_log10(rho)))}}}$'
        observation_rows.append(
            f'{rho_text} & {observation_labels[row["policy"]]} & '
            f'{validity[row["observation_envelope_valid"]]} & {rate} & '
            f'{float(row["mean_refresh_attempts"]):.3f} & '
            f'{float(row["mean_inspection_panels"]):.3f} & '
            f'{float(row["mean_sentinel_trials"]):.0f} \\\\'
        )
    write(output / "si_observation_dependence_rows.tex", observation_rows)

    comparison = read(data / "paper4_policy_comparison_uncertainty.csv")
    comparison_labels = dict(extended_labels)
    comparison_rows = []
    for row in comparison:
        comparison_rows.append(
            f'{comparison_labels[row["policy"]]} & '
            f'{100 * float(row["risk_difference"]):.3f} '
            f'[{100 * float(row["risk_difference_newcombe95_low"]):.3f}, '
            f'{100 * float(row["risk_difference_newcombe95_high"]):.3f}] & '
            f'{float(row["mean_refresh_difference"]):.3f} '
            f'[{float(row["mean_refresh_difference_bootstrap95_low"]):.3f}, '
            f'{float(row["mean_refresh_difference_bootstrap95_high"]):.3f}] & '
            f'{float(row["mean_inspection_panel_difference"]):.3f} '
            f'[{float(row["mean_inspection_panel_difference_bootstrap95_low"]):.3f}, '
            f'{float(row["mean_inspection_panel_difference_bootstrap95_high"]):.3f}] \\\\'
        )
    write(output / "si_policy_difference_rows.tex", comparison_rows)

    screen = read(data / "paper4_exact_screen_pass_diagnostic.csv")
    screen_rows = [
        f'{float(row["true_hazard"]):.3f} & '
        f'{int(row["cycle_age_years"])} & '
        f'{"yes" if row["candidate_next_state_model_safe"] == "True" else "no"} & '
        f'{int(row["y_star"])} & '
        f'{float(row["screen_pass_percent"]):.4f} \\\\'
        for row in screen
    ]
    write(output / "si_screen_pass_rows.tex", screen_rows)

    wrapper = read(data / "paper4_policy_wrapper_summary.csv")
    wrapper_policy_labels = {
        "posterior_mean": "Posterior mean",
        "posterior_conservative": "Posterior Q95",
    }
    wrapper.sort(
        key=lambda row: (
            float(row["true_hazard"]),
            list(wrapper_policy_labels).index(row["policy"]),
        )
    )
    wrapper_rows = []
    wrapper_admission_rows = []
    wrapper_event_rows = []
    wrapper_refresh_rows = []
    for row in wrapper:
        episodes = int(row["episodes"])
        base_events = round(
            float(row["base_episode_under_target_authorization_rate"]) * episodes
        )
        screened_events = round(
            float(row["screened_episode_under_target_authorization_rate"])
            * episodes
        )
        base_event_text = (
            f'{base_events}/{episodes:,} '
            f'({pct(row["base_episode_under_target_authorization_rate"], 2)}\\%)'
        )
        screened_event_text = (
            f'0/{episodes:,} '
            f'(0--{pct(row["screened_episode_under_target_authorization_ci95_high"], 4)}\\%)'
            if screened_events == 0
            else f'{screened_events}/{episodes:,} '
            f'({pct(row["screened_episode_under_target_authorization_rate"], 3)}\\%)'
        )
        wrapper_rows.append(
            f'{float(row["true_hazard"]):.3f} & '
            f'{wrapper_policy_labels[row["policy"]]} & '
            f'{pct(row["screened_candidate_no_refresh_rate"], 2)}\\% & '
            f'{pct(row["screen_pass_given_candidate"], 2)}\\% & '
            f'{base_event_text} & {screened_event_text} & '
            f'{float(row["base_mean_refreshes"]):.3f} & '
            f'{float(row["screened_mean_refreshes"]):.3f} & '
            f'{float(row["paired_mean_refresh_difference"]):.3f} '
            f'[{float(row["paired_mean_refresh_difference_ci95_low"]):.3f}, '
            f'{float(row["paired_mean_refresh_difference_ci95_high"]):.3f}] \\\\'
        )
        wrapper_admission_rows.append(
            f'{float(row["true_hazard"]):.3f} & '
            f'{wrapper_policy_labels[row["policy"]]} & '
            f'{pct(row["screened_candidate_no_refresh_rate"], 2)} & '
            f'{pct(row["screen_pass_given_candidate"], 2)} \\\\'
        )
        wrapper_event_rows.append(
            f'{float(row["true_hazard"]):.3f} & '
            f'{wrapper_policy_labels[row["policy"]]} & '
            f'{base_event_text} & {screened_event_text} & '
            f'{100 * float(row["paired_episode_event_difference"]):.2f} '
            f'[{100 * float(row["paired_episode_event_difference_ci95_low"]):.2f}, '
            f'{100 * float(row["paired_episode_event_difference_ci95_high"]):.2f}] \\\\'
        )
        wrapper_refresh_rows.append(
            f'{float(row["true_hazard"]):.3f} & '
            f'{wrapper_policy_labels[row["policy"]]} & '
            f'{float(row["base_mean_refreshes"]):.3f} & '
            f'{float(row["screened_mean_refreshes"]):.3f} & '
            f'{float(row["paired_mean_refresh_difference"]):.3f} '
            f'[{float(row["paired_mean_refresh_difference_ci95_low"]):.3f}, '
            f'{float(row["paired_mean_refresh_difference_ci95_high"]):.3f}] \\\\'
        )
    write(output / "si_policy_wrapper_rows.tex", wrapper_rows)
    write(output / "si_policy_wrapper_admission_rows.tex", wrapper_admission_rows)
    write(output / "si_policy_wrapper_event_rows.tex", wrapper_event_rows)
    write(output / "si_policy_wrapper_refresh_rows.tex", wrapper_refresh_rows)

    sampling = read(data / "paper4_sampling_law_sensitivity.csv")
    sampling_rows = []
    for row in sampling:
        rho = float(row["beta_binomial_intraclass_correlation"])
        rho_text = "0" if rho == 0.0 else f'{rho:.3f}'
        sampling_rows.append(
            f'{rho_text} & {float(row["p_at_max_noncoverage"]):.5f} & '
            f'{pct(row["max_per_look_noncoverage"], 3)}\\% & '
            f'{pct(row["excess_per_look_noncoverage"], 3)}\\% & '
            f'{float(row["illustrative_delta_samp_upper"]):.4f} & '
            f'{float(row["illustrative_horizon_bound"]):.4f} \\\\'
        )
    write(output / "si_sampling_law_rows.tex", sampling_rows)
    print("PASS_PAPER4_LATEX_TABLE_GENERATION")


def np_log10(value: float) -> float:
    import math
    return math.log10(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers/paper4_adaptive_archive_maintenance/generated_tables",
    )
    args = parser.parse_args()
    build(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
