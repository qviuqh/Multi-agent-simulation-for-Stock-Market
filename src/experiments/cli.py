import argparse
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Any

import pandas as pd
import yaml

from .runner import ExperimentRunner


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_population_config(populations: Dict[str, Dict], population_ref: Any) -> Dict[str, Any]:
    """Resolve population name or inline config into a full population config."""
    if isinstance(population_ref, str):
        if population_ref not in populations:
            raise KeyError(f"Population '{population_ref}' not found in config.")
        return {"name": population_ref, **populations[population_ref]}

    if isinstance(population_ref, dict):
        name = population_ref.get("name", "custom")
        config = {key: value for key, value in population_ref.items() if key != "name"}
        return {"name": name, **config}

    raise TypeError("Population reference must be a name or a dict.")


def _ensure_output_dir(base_dir: str, scenario_name: str) -> Path:
    output_dir = Path(base_dir) / scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_results(df: pd.DataFrame, output_dir: Path, filename: str, save_data: bool) -> Optional[Path]:
    if not save_data:
        return None
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    return output_path


def _parse_seed_values(seed: Optional[int], n_seeds: int) -> Iterable[int]:
    if seed is not None:
        return [seed]
    return range(n_seeds)


def run_population_sweep(
    config: Dict[str, Any],
    scenario: Dict[str, Any],
    seed: Optional[int],
) -> Optional[pd.DataFrame]:
    if not scenario.get("enabled", False):
        return None

    market_config = config["market"]
    agent_configs = config.get("agents", {})
    population_defs = config.get("populations", {})
    experiment_config = config.get("experiment", {})
    output_config = config.get("output", {})

    populations = [
        resolve_population_config(population_defs, pop_name)
        for pop_name in scenario.get("populations", [])
    ]

    runner = ExperimentRunner(market_config, agent_configs)
    results = runner.run_population_sweep(
        populations,
        n_seeds=experiment_config.get("n_seeds", 10),
        n_steps=experiment_config.get("episode_length", 1000),
        seeds=_parse_seed_values(seed, experiment_config.get("n_seeds", 10)),
    )

    output_dir = _ensure_output_dir(output_config.get("base_dir", "./outputs"), "population_sweep")
    _save_results(results, output_dir, "results.csv", output_config.get("save_data", True))
    return results


def run_cost_sweep(
    config: Dict[str, Any],
    scenario: Dict[str, Any],
    seed: Optional[int],
) -> Optional[pd.DataFrame]:
    if not scenario.get("enabled", False):
        return None

    market_config = config["market"]
    agent_configs = config.get("agents", {})
    population_defs = config.get("populations", {})
    experiment_config = config.get("experiment", {})
    output_config = config.get("output", {})

    base_population = resolve_population_config(population_defs, scenario["base_population"])
    results_frames: List[pd.DataFrame] = []

    for cost in scenario.get("transaction_costs", []):
        cost_market_config = {**market_config, "transaction_cost": cost}
        runner = ExperimentRunner(cost_market_config, agent_configs)
        cost_population = {**base_population, "name": f"{base_population['name']}_cost_{cost}"}
        results = runner.run_population_sweep(
            [cost_population],
            n_seeds=experiment_config.get("n_seeds", 10),
            n_steps=experiment_config.get("episode_length", 1000),
            seeds=_parse_seed_values(seed, experiment_config.get("n_seeds", 10)),
        )
        results["transaction_cost"] = cost
        results_frames.append(results)

    if not results_frames:
        return None

    combined = pd.concat(results_frames, ignore_index=True)
    output_dir = _ensure_output_dir(output_config.get("base_dir", "./outputs"), "cost_sweep")
    _save_results(combined, output_dir, "results.csv", output_config.get("save_data", True))
    return combined


def run_rl_impact(
    config: Dict[str, Any],
    scenario: Dict[str, Any],
    seed: Optional[int],
) -> Optional[pd.DataFrame]:
    if not scenario.get("enabled", False):
        return None

    market_config = config["market"]
    agent_configs = config.get("agents", {})
    population_defs = config.get("populations", {})
    experiment_config = config.get("experiment", {})
    output_config = config.get("output", {})

    base_population = resolve_population_config(population_defs, scenario["base_population"])
    populations: List[Dict[str, Any]] = []

    if scenario.get("without_rl", False):
        populations.append({**base_population, "name": f"{base_population['name']}_no_rl"})

    if scenario.get("with_rl", False):
        rl_count = scenario.get("rl_count", 1)
        populations.append({**base_population, "name": f"{base_population['name']}_with_rl", "rl": rl_count})

    if not populations:
        return None

    runner = ExperimentRunner(market_config, agent_configs)
    results = runner.run_population_sweep(
        populations,
        n_seeds=experiment_config.get("n_seeds", 10),
        n_steps=experiment_config.get("episode_length", 1000),
        seeds=_parse_seed_values(seed, experiment_config.get("n_seeds", 10)),
    )

    output_dir = _ensure_output_dir(output_config.get("base_dir", "./outputs"), "rl_impact")
    _save_results(results, output_dir, "results.csv", output_config.get("save_data", True))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment scenarios from YAML config.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Scenario to run (can be specified multiple times). Defaults to all enabled scenarios.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed to run a single deterministic run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    scenarios = config.get("scenarios", {})
    requested = set(args.scenario or [])

    for scenario_name, scenario_config in scenarios.items():
        if requested and scenario_name not in requested:
            continue

        if scenario_name == "population_sweep":
            run_population_sweep(config, scenario_config, args.seed)
        elif scenario_name == "cost_sweep":
            run_cost_sweep(config, scenario_config, args.seed)
        elif scenario_name == "rl_impact":
            run_rl_impact(config, scenario_config, args.seed)


if __name__ == "__main__":
    main()