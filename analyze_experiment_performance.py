"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Analyze experiment performance across training iterations.

Computes dual metrics:
- Present: accuracy among successfully processed examples  
- Full: accuracy against expected total (missing = wrong)

Usage:
    python analyze_experiment_performance.py experiment_dir [--test_size 500] [--agents 3] [--no-plot]

Outputs:
- Terminal: detailed performance comparison
- experiment_summary_structured.json: structured metrics
- summary.png: performance frontier plot  
- agreement_distribution_iteration_N.pdf: agent agreement analysis
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from collections import Counter

from read_debate_performance import (
    load_debate_file,
    analyze_debate_performance,
    detect_dataset_from_filename,  # imported for completeness, may be unused
    parse_answer,
    grade_answer,
)


def extract_trainer_metrics(
    experiment_dir: str, iteration: int, agent_id: str, phase: str = "kto"
) -> Dict[str, float]:
    """Extract metrics (loss, kl, reward, margin) from trainer_state.json for a specific agent."""
    agent_num = agent_id.split("_")[-1] if "_" in agent_id else agent_id
    trainer_state_path = os.path.join(
        experiment_dir, "logs", phase, agent_num, "checkpoint-480", "trainer_state.json"
    )
    if not os.path.exists(trainer_state_path):
        return {}

    try:
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)
        log_history = trainer_state.get("log_history", [])
        if not log_history:
            return {}

        metrics: Dict[str, float] = {}
        # loss & kl
        for key in ["loss", "kl"]:
            initial = log_history[0].get(key)
            final = log_history[-1].get(key)
            if initial is not None or final is not None:
                metrics[f"initial_{key}"] = initial
                metrics[f"final_{key}"] = final

        # reward stats
        reward_keys = {
            "rewards/chosen": "reward_chosen",
            "rewards/rejected": "reward_rejected",
            "rewards/margins": "margin",
        }
        for json_key, metric_key in reward_keys.items():
            initial = log_history[0].get(json_key)
            final = log_history[-1].get(json_key)
            if initial is not None or final is not None:
                metrics[f"initial_{metric_key}"] = initial
                metrics[f"final_{metric_key}"] = final

        return metrics
    except Exception:
        return {}


# -------------------------------------------------------------
# Helpers for standard-format debates
# -------------------------------------------------------------

def compute_agreement_distribution(
    debate_data: Dict[str, Any],
    dataset: str,
    num_agents: int,
    expected_test_size: int = None,
) -> Dict[str, Any]:
    """Compute agreement distributions before (initial) and after (final) the debate.
    
    Unparseable responses are treated as disagreement. Agreement level represents
    how many agents gave parseable responses that agreed on the most popular answer.
    
    Returns agreement levels {0,1,2,3} where:
    - 0: No agents gave parseable responses  
    - 1: Only 1 agent gave a parseable response (or all parseable responses disagree)
    - 2: 2 agents agreed on the most popular parseable answer
    - 3: All 3 agents agreed on the same parseable answer
    """
    agreement_init = {0: 0, 1: 0, 2: 0, 3: 0}
    agreement_final = {0: 0, 1: 0, 2: 0, 3: 0}

    n_processed_init = 0
    n_processed_final = 0
    total_examples = 0

    for key, ex in debate_data.items():
        if key == "metrics" or not isinstance(ex, dict) or ex.get("split") != "test":
            continue

        total_examples += 1
        contexts = ex.get("context", [])
        
        # Collect initial and final answers for each agent
        init_answers = []
        final_answers = []
        for i in range(num_agents):
            if i >= len(contexts):
                init_answers.append(None)
                final_answers.append(None)
                continue
            ctx = contexts[i]
            if not isinstance(ctx, list) or len(ctx) <= 1:
                init_answers.append(None)
                final_answers.append(None)
                continue

            # initial: first assistant message
            first_msg = next((m for m in ctx if m.get("role") == "assistant" and m.get("content")), None)
            if first_msg:
                init_answers.append(parse_answer(first_msg["content"], dataset=dataset))
            else:
                init_answers.append(None)

            # final: last assistant message
            last_msg = ctx[-1]
            if last_msg.get("role") == "assistant" and last_msg.get("content"):
                final_answers.append(parse_answer(last_msg["content"], dataset=dataset))
            else:
                final_answers.append(None)

        # Initial: Count parseable answers and their agreement
        parseable_init = [ans for ans in init_answers if ans is not None]
        if len(parseable_init) == 0:
            agreement_level = 0
        else:
            counts = Counter(parseable_init)
            max_count = max(counts.values()) if counts else 0
            agreement_level = min(max_count, num_agents)  # clamp to [0, num_agents]
        
        agreement_init[agreement_level] += 1
        n_processed_init += 1

        # Final: Count parseable answers and their agreement  
        parseable_final = [ans for ans in final_answers if ans is not None]
        if len(parseable_final) == 0:
            agreement_level = 0
        else:
            counts = Counter(parseable_final)
            max_count = max(counts.values()) if counts else 0
            agreement_level = min(max_count, num_agents)
            
        agreement_final[agreement_level] += 1
        n_processed_final += 1

    # Calculate missing examples (expected vs actually processed)
    missing_examples = (expected_test_size - total_examples) if expected_test_size else 0
    
    # Add missing examples to 0/3 bucket (no parseable responses in final round)
    agreement_init[0] += missing_examples
    agreement_final[0] += missing_examples

    return {
        "initial": {
            "agreement": agreement_init,
            "n_processed": expected_test_size if expected_test_size else n_processed_init,
            "total": sum(agreement_init.values()),
            "unanimous": agreement_init.get(3, 0),
            "no_parseable": agreement_init.get(0, 0),
        },
        "final": {
            "agreement": agreement_final,
            "n_processed": expected_test_size if expected_test_size else n_processed_final,
            "total": sum(agreement_final.values()),
            "unanimous": agreement_final.get(3, 0),
            "no_parseable": agreement_final.get(0, 0),
        },
        "metadata": {
            "total_examples_in_file": total_examples,
            "expected_test_size": expected_test_size,
            "missing_examples": missing_examples,
        },
    }


def plot_agreement_distribution(
    distribution: Dict[str, Any],
    output_path: str = "agreement_distribution.pdf",
    num_agents: int = 3,
):
    """Plot agreement distributions for initial and final rounds.

    X-axis categories: 0/3, 1/3, 2/3, 3/3. Single bar series per subplot showing
    agreement levels (number of agents that agreed on the most popular parseable answer).
    0/3 represents examples where no agents gave parseable responses.
    """
    from matplotlib.ticker import MaxNLocator

    categories = [0, 1, 2, 3]
    labels = [f"{c}/{num_agents}" for c in categories]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Helper to draw a single subplot
    def _draw(ax, title, round_dist):
        agreement_counts = round_dist.get("agreement", {})
        agreement_vals = [agreement_counts.get(c, 0) for c in categories]
        x = np.arange(len(categories))
        width = 0.6
        
        # Use different colors: red for 0/3 (no parseable), blue for others
        colors = ["firebrick" if c == 0 else "steelblue" for c in categories]
        bars = ax.bar(x, agreement_vals, width, color=colors, label="Examples")
        
        # Simplified title - now that missing are included in 0/3, just show the round name
        n_processed = round_dist.get("n_processed", 0)
        title_str = f"{title} (n={n_processed})"
        
        ax.set_title(title_str)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Number of agents in agreement")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # annotate bars
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    _draw(axes[0], "Initial round", distribution.get("initial", {}))
    _draw(axes[1], "Final round", distribution.get("final", {}))

    axes[0].set_ylabel("# Test examples")
    
    # Add legend explaining colors to left chart
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='firebrick', label='No parseable responses (0/3)'),
        Patch(facecolor='steelblue', label='Parseable agent responses (1-3/3)')
    ]
    axes[0].legend(handles=legend_elements, loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def compute_agent_stats_standard(
    debate_data: Dict[str, Any],
    dataset: str,
    expected_total: int,
    num_agents: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (agent_perf_dict, consensus_dict).

    * "present" metrics use the actually *loaded* example counts.
    * "all" metrics use the provided `expected_total` (missing = incorrect).
    """
    agent_correct_final = [0] * num_agents
    agent_resp_final = [0] * num_agents

    agent_correct_init = [0] * num_agents
    agent_resp_init = [0] * num_agents

    total_loaded = 0
    consensus_correct = 0

    for key, ex in debate_data.items():
        if key == "metrics" or not isinstance(ex, dict) or ex.get("split") != "test":
            continue
        total_loaded += 1

        if grade_answer(ex["consensus_answer"], ex["ground_truth"]):
            consensus_correct += 1

        contexts = ex.get("context", [])
        for i in range(num_agents):
            if i >= len(contexts):
                continue
            ctx = contexts[i]
            if not isinstance(ctx, list) or len(ctx) <= 1:
                continue

            # final msg
            final_msg = ctx[-1]
            if final_msg.get("role") == "assistant" and final_msg.get("content"):
                agent_resp_final[i] += 1
                parsed = parse_answer(final_msg["content"], dataset=dataset)
                if grade_answer(parsed, ex["ground_truth"]):
                    agent_correct_final[i] += 1

            # initial msg
            first_msg = next(
                (m for m in ctx if m.get("role") == "assistant" and m.get("content")), None
            )
            if first_msg:
                agent_resp_init[i] += 1
                parsed_first = parse_answer(first_msg["content"], dataset=dataset)
                if grade_answer(parsed_first, ex["ground_truth"]):
                    agent_correct_init[i] += 1

    agent_perf: Dict[str, Any] = {}
    for i in range(num_agents):
        final_present = agent_correct_final[i] / agent_resp_final[i] if agent_resp_final[i] else 0.0
        final_all = agent_correct_final[i] / expected_total if expected_total else 0.0

        init_present = agent_correct_init[i] / agent_resp_init[i] if agent_resp_init[i] else 0.0
        init_all = agent_correct_init[i] / expected_total if expected_total else 0.0

        agent_perf[f"agent_{i}"] = {
            "final_present": final_present,    # R1 accuracy among examples agent responded to
            "final_all": final_all,            # R1 accuracy against expected total (missing = wrong)
            "init_present": init_present,      # R0 accuracy among examples agent responded to  
            "init_all": init_all,              # R0 accuracy against expected total (missing = wrong)
            "final_correct": agent_correct_final[i],   # Number of correct R1 responses
            "final_responses": agent_resp_final[i],    # Number of R1 responses given
            "init_correct": agent_correct_init[i],     # Number of correct R0 responses
            "init_responses": agent_resp_init[i],      # Number of R0 responses given
            "total_loaded": total_loaded,      # Total examples successfully loaded
            "total_expected": expected_total,  # Expected total examples
        }

    consensus = {
        "correct": consensus_correct,          # Number of correct consensus answers
        "loaded": total_loaded,                # Number of examples successfully loaded  
        "expected": expected_total,            # Expected total number of examples
        "acc_loaded": consensus_correct / total_loaded if total_loaded else 0.0,        # PRESENT: consensus accuracy among loaded examples
        "acc_expected": consensus_correct / expected_total if expected_total else 0.0,  # FULL: consensus accuracy against expected total
    }
    return agent_perf, consensus


def load_iteration_debates(experiment_dir: str) -> Dict[int, Dict[str, Any]]:
    """Load all debate files from an experiment directory."""
    debate_dir = os.path.join(experiment_dir, "debate")
    if not os.path.exists(debate_dir):
        raise FileNotFoundError(f"Debate directory not found: {debate_dir}")

    debates: Dict[int, Dict[str, Any]] = {}
    for filename in os.listdir(debate_dir):
        if filename.startswith("debate_iteration_") and filename.endswith(".json"):
            try:
                iteration_num = int(filename.split("_")[-1].replace(".json", ""))
                filepath = os.path.join(debate_dir, filename)
                debates[iteration_num] = load_debate_file(filepath)
            except Exception:
                continue

    if not debates:
        raise FileNotFoundError(f"No debate files found in {debate_dir}")
    return debates


def extract_agent_performance_from_binary(
    debate_data: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Extract individual agent performance from binary agent_answers format."""
    agent_correct = [0, 0, 0]
    agent_responses = [0, 0, 0]
    total_examples_loaded = 0

    for key, value in debate_data.items():
        if key != "metrics" and isinstance(value, dict) and value.get("split") == "test":
            if "agent_answers" in value:
                agent_answers = value["agent_answers"]
                if len(agent_answers) == 3:
                    for i, correct in enumerate(agent_answers):
                        agent_correct[i] += correct
                        # Count if agent actually responded
                        if "context" in value and len(value["context"]) > i:
                            context = value["context"][i]
                            if isinstance(context, list) and len(context) > 1:
                                has_response = any(
                                    msg.get("role") == "assistant" and msg.get("content")
                                    for msg in context
                                )
                                if has_response:
                                    agent_responses[i] += 1
                    total_examples_loaded += 1

    expected_total = 500  # kept for backward compatibility inside this helper
    agent_performance: Dict[str, Dict[str, float]] = {}
    for i in range(3):
        accuracy_with_responses = (
            agent_correct[i] / agent_responses[i] if agent_responses[i] else 0.0
        )
        accuracy_all_500 = agent_correct[i] / expected_total if expected_total else 0.0

        agent_performance[f"agent_{i}"] = {
            "accuracy_with_responses": accuracy_with_responses,
            "accuracy_all_500": accuracy_all_500,
            "correct": float(agent_correct[i]),
            "responses": float(agent_responses[i]),
            "total_loaded": float(total_examples_loaded),
            "total_expected": float(expected_total),
        }

    consensus_correct = sum(
        1
        for k, v in debate_data.items()
        if k != "metrics"
        and isinstance(v, dict)
        and v.get("split") == "test"
        and v.get("consensus_answer") == v.get("ground_truth")
    )
    consensus = {
        "correct": consensus_correct,
        "loaded": total_examples_loaded,
        "expected": expected_total,
        "acc_loaded": consensus_correct / total_examples_loaded if total_examples_loaded else 0.0,
        "acc_expected": consensus_correct / expected_total if expected_total else 0.0,
    }

    return agent_performance, consensus


def analyze_experiment_performance(
    experiment_dir: str, test_size: int = 500, agents: int = 3
) -> Dict[str, Any]:
    """Analyze performance across all experiment iterations.
    
    Computes present metrics (among processed examples) and full metrics 
    (against expected total, treating missing as wrong) for each iteration.
    
    Args:
        experiment_dir: Path to experiment directory
        test_size: Expected test examples for full metrics
        agents: Number of agents
        
    Returns:
        Dict mapping iteration -> {'method': str, 'metrics': Dict}
    """
    print(f"Analyzing experiment: {experiment_dir}")
    debates = load_iteration_debates(experiment_dir)
    iterations = sorted(debates.keys())

    # Detect dataset and test_size from experiment configuration
    config_path = os.path.join(experiment_dir, "config.json")
    dataset = "gsm8k"  # default fallback
    expected_test_size = test_size  # use command line arg as fallback
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                dataset = config.get('dataset', 'gsm8k')
                expected_test_size = config.get('test_size', test_size)
        except Exception:
            pass
    
    print(f"Detected dataset: {dataset}")
    print(f"Expected test size: {expected_test_size}")

    results: Dict[int, Dict[str, Any]] = {}

    for iteration in iterations:
        debate_data = debates[iteration]
        expected_total = test_size  # ensure defined in both try/except paths

        try:
            # Standard analysis returns *loaded-denominator* metrics in `test` and `train`.
            metrics = analyze_debate_performance(debate_data, dataset)

            if not metrics.get("agent_performance"):
                raise ValueError("No agent performance data in standard analysis")

            # augment with full-denominator stats
            agent_perf_full, consensus_full = compute_agent_stats_standard(
                debate_data, dataset, expected_total, agents
            )

            # agreement distribution + plot
            agreement = compute_agreement_distribution(debate_data, dataset, agents, expected_test_size)
            agreement_plot_path = os.path.join(
                experiment_dir, f"agreement_distribution_iteration_{iteration}.pdf"
            )
            try:
                plot_agreement_distribution(agreement, output_path=agreement_plot_path, num_agents=agents)
            except Exception:
                pass

            # trainer metrics (attach per-agent if available)
            for agent_id in agent_perf_full.keys():
                tmetrics = extract_trainer_metrics(experiment_dir, iteration, agent_id)
                if tmetrics:
                    agent_perf_full[agent_id].update(tmetrics)

            # R0 majority vote analysis (present vs full denominators)
            # PRESENT: Only count examples where agents gave parseable R0 responses
            # FULL: Count all expected examples (missing/unparseable treated as wrong)
            r0_correct = 0
            r0_total = 0
            for _, ex in debate_data.items():
                if isinstance(ex, dict) and ex.get("split") == "test":
                    first_answers = []
                    for ctx in ex.get("context", []):
                        first_msg = next((m for m in ctx if m.get("role") == "assistant"), None)
                        if first_msg and first_msg.get("content"):
                            ans = parse_answer(first_msg["content"], dataset)
                            if ans is not None:
                                first_answers.append(ans)
                    if not first_answers:
                        continue  # Skip examples with no parseable R0 responses
                    majority_ans = Counter(first_answers).most_common(1)[0][0]
                    if grade_answer(majority_ans, ex["ground_truth"]):
                        r0_correct += 1
                    r0_total += 1

            # Store both present and full-denominator forms for R0 majority vote
            metrics["r0_majority_accuracy"] = r0_correct / r0_total if r0_total else 0.0  # PRESENT
            metrics["r0_majority_correct_full"] = r0_correct  # Numerator for full-denom calculation
            metrics["r0_majority_total_full"] = expected_total  # Full denominator (missing = wrong)
            metrics["agent_performance_full"] = agent_perf_full
            metrics["consensus_full"] = consensus_full
            metrics["agreement_distribution"] = agreement
            metrics["agreement_plot_path"] = agreement_plot_path

            # IMPORTANT: do **not** overwrite loaded-denominator metrics produced by
            # `analyze_debate_performance`. Those are the "present" figures.
            # We keep full-denominator figures in `consensus_full` and
            # per-agent in `agent_performance_full`.

            results[iteration] = {"method": "standard", "metrics": metrics}

        except Exception:
            # Fallback to binary format (older logs). Use loaded denominators for `test`.
            try:
                test_examples = [
                    v
                    for k, v in debate_data.items()
                    if k != "metrics" and isinstance(v, dict) and v.get("split") == "test"
                ]
                if not test_examples:
                    continue

                loaded_count = len(test_examples)
                consensus_correct = sum(
                    1 for ex in test_examples if ex.get("consensus_answer") == ex.get("ground_truth")
                )
                consensus_accuracy = consensus_correct / loaded_count if loaded_count else 0.0

                agent_performance, consensus = extract_agent_performance_from_binary(debate_data)

                # Weighted-by-responses average agent accuracy (present)
                total_correct = sum(perf.get("correct", 0.0) for perf in agent_performance.values())
                total_responses = sum(perf.get("responses", 0.0) for perf in agent_performance.values())
                avg_agent_accuracy = total_correct / total_responses if total_responses else 0.0

                # agreement distribution + plot
                agreement = compute_agreement_distribution(debate_data, dataset, agents, expected_test_size)
                agreement_plot_path = os.path.join(
                    experiment_dir, f"agreement_distribution_iteration_{iteration}.pdf"
                )
                try:
                    plot_agreement_distribution(agreement, output_path=agreement_plot_path, num_agents=agents)
                except Exception:
                    pass

                # R0 majority (present vs full)
                r0_correct = 0
                r0_total = 0
                for _, ex in debate_data.items():
                    if isinstance(ex, dict) and ex.get("split") == "test":
                        first_answers = []
                        for ctx in ex.get("context", []):
                            first_msg = next((m for m in ctx if m.get("role") == "assistant"), None)
                            if first_msg and first_msg.get("content"):
                                ans = parse_answer(first_msg["content"], dataset)
                                if ans is not None:
                                    first_answers.append(ans)
                        if not first_answers:
                            continue
                        majority_ans = Counter(first_answers).most_common(1)[0][0]
                        if grade_answer(majority_ans, ex.get("ground_truth")):
                            r0_correct += 1
                        r0_total += 1

                results[iteration] = {
                    "method": "binary",
                    "metrics": {
                        "test": {
                            "count": loaded_count,  # loaded denominator
                            "consensus_accuracy": consensus_accuracy,
                            "consensus_correct": consensus_correct,
                            "avg_agent_accuracy": avg_agent_accuracy,
                        },
                        "agent_performance": agent_performance,
                        "consensus": consensus,
                        # carry R0 present & full-like fields for consistency with printing
                        "r0_majority_accuracy": r0_correct / r0_total if r0_total else 0.0,
                        "r0_majority_correct_full": r0_correct,
                        "r0_majority_total_full": expected_total,
                        "agreement_distribution": agreement,
                        "agreement_plot_path": agreement_plot_path,
                    },
                }
            except Exception:
                continue

    return results


def print_performance_comparison(
    results: Dict[str, Any]
):
    """Print detailed performance comparison across iterations.
    
    Shows consensus, individual agent, and overall performance with
    both present metrics (processed examples) and full metrics (expected total).
    
    Args:
        results: Output from analyze_experiment_performance()
    """
    iterations = sorted(results.keys())
    if len(iterations) < 2:
        print("Need at least 2 iterations for comparison")
        return

    print(f"\n{'='*80}")
    print("EXPERIMENT PERFORMANCE COMPARISON")
    print(f"{'='*80}")

    # Data splits
    print("\nDATA SPLITS:")
    for iteration in iterations:
        metrics = results[iteration]['metrics']
        train_count = metrics.get('train', {}).get('count', 0)
        test_count = metrics.get('test', {}).get('count', 0)
        print(f"  Iteration {iteration}: {train_count} train + {test_count} test examples")

    comparison_data = {}
    for iteration in iterations:
        metrics = results[iteration]['metrics']

        test_consensus_acc = metrics['test']['consensus_accuracy']
        r1_present = metrics['test']['avg_agent_accuracy']

        if 'agent_performance_full' in metrics:
            r1_all_vals = [
                pd.get('final_all', pd.get('accuracy_all_500', 0.0))
                for pd in metrics['agent_performance_full'].values()
            ]
            r1_all = float(np.mean(r1_all_vals)) if r1_all_vals else r1_present
        else:
            r1_all = r1_present

        # R0 present/all
        r0_present_vals, r0_all_vals = [], []
        for _aid, _pd in metrics.get('agent_performance_full', {}).items():
            if 'init_present' in _pd:
                r0_present_vals.append(_pd['init_present'])
                r0_all_vals.append(_pd['init_all'])
        if not r0_present_vals:
            for _aid, _pd in metrics.get('agent_performance', {}).items():
                if isinstance(_pd, dict) and 'test_round0_accuracy' in _pd:
                    r0_present_vals.append(_pd['test_round0_accuracy'])
                    r0_all_vals.append(_pd.get('test_round0_accuracy', 0.0))

        r0_present = float(np.mean(r0_present_vals)) if r0_present_vals else 0.0
        r0_all     = float(np.mean(r0_all_vals))     if r0_all_vals     else r0_present

        # Clean agent dict and merge in full-denom extras
        agent_perf_clean = {}
        for agent_id, perf in metrics.get('agent_performance', {}).items():
            if isinstance(perf, dict):
                agent_perf_clean[agent_id] = {
                    k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in perf.items()
                }
            else:
                agent_perf_clean[agent_id] = float(perf)

        for ag, pd in metrics.get('agent_performance_full', {}).items():
            agent_perf_clean.setdefault(ag, {})
            for k, v in pd.items():
                agent_perf_clean[ag][k] = float(v) if isinstance(v, (np.floating, np.integer)) else v

        comparison_data[iteration] = {
            'consensus_accuracy': float(test_consensus_acc),
            'r1_present': r1_present,
            'r1_all': r1_all,
            'r0_present': r0_present,
            'r0_all': r0_all,
            'r0_majority_accuracy': metrics.get('r0_majority_accuracy', 0.0),
            'avg_agent_accuracy': r1_present,
            'avg_agent_accuracy_all': r1_all,
            'agent_performance': agent_perf_clean
        }

    # Consensus performance
    print("\nCONSENSUS PERFORMANCE (TEST SET):")
    for i, iteration in enumerate(iterations):
        metrics = results[iteration]['metrics']
        acc = comparison_data[iteration]['consensus_accuracy']
        correct = metrics['test']['consensus_correct']
        total_loaded = metrics['test']['count']

        # R0 majority present vs full
        r0_present_mv = comparison_data[iteration].get('r0_majority_accuracy', 0.0)
        full_total_for_r0 = (
            metrics.get('r0_majority_total_full')
            or metrics.get('consensus_full', {}).get('expected')
            or 0
        )
        if full_total_for_r0:
            r0_full_mv = metrics.get('r0_majority_correct_full', 0) / full_total_for_r0
        else:
            r0_full_mv = r0_present_mv

        if 'consensus_full' in metrics:
            full = metrics['consensus_full']
            acc_all = full['acc_expected']
            total_exp = full['expected']
            print(
                f"  Iteration {iteration}: "
                f"R1 present={acc*100:.1f}% ({correct}/{total_loaded}) "
                f"| all={acc_all*100:.1f}% ({full['correct']}/{total_exp}) "
                f"| R0 majority present={r0_present_mv*100:.1f}% | all={r0_full_mv*100:.1f}%"
            )
        else:
            # Compute an 'all' view using full_total_for_r0 if available; fall back to present.
            if full_total_for_r0:
                r1_all_est = correct / full_total_for_r0
            else:
                r1_all_est = acc
            print(
                f"  Iteration {iteration}: "
                f"R1 present={acc*100:.1f}% ({correct}/{total_loaded}) "
                f"| all={r1_all_est*100:.1f}% "
                f"| R0 majority present={r0_present_mv*100:.1f}% | all={r0_full_mv*100:.1f}%"
            )

        if i > 0:
            prev_r1 = comparison_data[iterations[i-1]]['consensus_accuracy']
            prev_r0_present = comparison_data[iterations[i-1]].get('r0_majority_accuracy', 0.0)
            print(f"    Δ R1 present={acc - prev_r1:+.3f} | Δ R0 majority present={r0_present_mv - prev_r0_present:+.3f}")

    # Average agent
    print("\nAVERAGE AGENT PERFORMANCE (TEST SET):")
    for i, iteration in enumerate(iterations):
        data_it = comparison_data[iteration]
        print(
            f"  Iteration {iteration}: R1 present={data_it['r1_present']*100:.1f}% | "
            f"R1 all={data_it['r1_all']*100:.1f}% | R0 present={data_it['r0_present']*100:.1f}% | "
            f"R0 all={data_it['r0_all']*100:.1f}%"
        )
        if i > 0:
            prev_it = comparison_data[iterations[i-1]]
            def d(c,p): return c-p
            print(
                f"    Δ R1 present={d(data_it['r1_present'], prev_it['r1_present']):+.3f} | "
                f"R1 all={d(data_it['r1_all'], prev_it['r1_all']):+.3f} | "
                f"R0 present={d(data_it['r0_present'], prev_it['r0_present']):+.3f} | "
                f"R0 all={d(data_it['r0_all'], prev_it['r0_all']):+.3f}"
            )

    # Individual agents
    print("\nINDIVIDUAL AGENT PERFORMANCE:")
    all_agent_ids = sorted({
        ag for iteration in iterations
        for ag in comparison_data[iteration]['agent_performance'].keys()
    })

    metric_keys = [
        ('loss', 'Training Loss'),
        ('kl', 'KL'),
        ('reward_chosen', 'Reward Chosen'),
        ('reward_rejected', 'Reward Rejected'),
        ('margin', 'Margin'),
    ]
    last_metric_line_per_agent = {}

    for agent_id in all_agent_ids:
        print(f"\n  {agent_id.upper()}:")
        prev_perf = None
        for i, iteration in enumerate(iterations):
            perf_data = comparison_data[iteration]['agent_performance'].get(agent_id, {})

            if 'final_present' in perf_data:
                r1p = perf_data.get('final_present', 0.0)
                r1a = perf_data.get('final_all', 0.0)
                r0p = perf_data.get('init_present', 0.0)
                r0a = perf_data.get('init_all', 0.0)
                print(f"    Iteration {iteration} R1  : present={r1p*100:.1f}% | all={r1a*100:.1f}%")
                print(f"    Iteration {iteration} R0  : present={r0p*100:.1f}% | all={r0a*100:.1f}%")
                print(f"      Δ within iter (R1-R0): present={r1p-r0p:+.3f} | all={r1a-r0a:+.3f}")
            elif 'test_round1_accuracy' in perf_data:
                r1 = perf_data.get('test_round1_accuracy', perf_data.get('test_accuracy', 0.0))
                r0 = perf_data.get('test_round0_accuracy', perf_data.get('test_init_accuracy', 0.0))
                print(f"    Iteration {iteration} R1={r1*100:.1f}%  | R0={r0*100:.1f}%")
                print(f"      Δ within iter (R1-R0): present={r1-r0:+.3f}")
            else:
                acc = float(perf_data) if perf_data else 0.0
                print(f"    Iteration {iteration}: {acc:.3f} ({acc*100:.1f}%)")

            # trainer metrics line
            metric_strs = []
            for key, label in metric_keys:
                iv = perf_data.get(f'initial_{key}')
                fv = perf_data.get(f'final_{key}')
                if iv is not None and fv is not None:
                    metric_strs.append(f"{label}: {iv:.4f} → {fv:.4f}")
            metric_line = " | ".join(metric_strs)
            if metric_line and last_metric_line_per_agent.get(agent_id) != metric_line:
                print("      " + metric_line)
                last_metric_line_per_agent[agent_id] = metric_line

            if i > 0 and prev_perf is not None:
                def get_r1(p):
                    if 'final_present' in p: return p['final_present']
                    return p.get('test_round1_accuracy', p.get('test_accuracy', 0.0))
                def get_r0(p):
                    if 'init_present' in p: return p['init_present']
                    return p.get('test_round0_accuracy', p.get('test_init_accuracy', 0.0))
                print(
                    f"      Δ R1 present: {get_r1(perf_data)-get_r1(prev_perf):+.3f} | "
                    f"R0 present: {get_r0(perf_data)-get_r0(prev_perf):+.3f}"
                )
            prev_perf = perf_data

    # Best agent
    print("\nBEST AGENT ANALYSIS:")
    for i, iteration in enumerate(iterations):
        agent_perf = comparison_data[iteration]['agent_performance']
        best_agent, best_acc = None, -1.0
        for ag, pd in agent_perf.items():
            acc = (pd.get('test_round1_accuracy', pd.get('final_present', pd.get('test_accuracy', 0.0)))
                   if isinstance(pd, dict) else float(pd))
            if acc > best_acc:
                best_acc, best_agent = acc, ag

        if best_agent is not None:
            print(f"  Iteration {iteration}: {best_agent} ({best_acc:.3f})")
            if i > 0:
                prev_pd = comparison_data[iterations[i-1]]['agent_performance'].get(best_agent, {})
                prev_acc = (prev_pd.get('test_round1_accuracy',
                                        prev_pd.get('final_present',
                                                    prev_pd.get('test_accuracy', 0.0)))
                            if isinstance(prev_pd, dict) else (float(prev_pd) if prev_pd else 0.0))
                print(f"    {best_agent} improvement: {best_acc - prev_acc:+.3f} ({(best_acc - prev_acc)*100:.1f}%)")

    # Overall improvement
    first_iter = iterations[0]
    last_iter = iterations[-1]
    first_consensus = comparison_data[first_iter]['consensus_accuracy']
    last_consensus = comparison_data[last_iter]['consensus_accuracy']
    first_avg = comparison_data[first_iter]['avg_agent_accuracy']
    last_avg = comparison_data[last_iter]['avg_agent_accuracy']

    print("\nOVERALL IMPROVEMENT SUMMARY (for debates with answers):")
    print(f"  Consensus Accuracy: {first_consensus:.3f} → {last_consensus:.3f} ({last_consensus-first_consensus:+.3f})")
    print(f"  Average Agent Accuracy: {first_avg:.3f} → {last_avg:.3f} ({last_avg-first_avg:+.3f})")
    if last_consensus > first_consensus:
        print(f"  [SUCCESS] Consensus performance improved by {(last_consensus-first_consensus)*100:.1f} percentage points")
    if last_avg > first_avg:
        print(f"  [SUCCESS] Average agent performance improved by {(last_avg-first_avg)*100:.1f} percentage points")

    # -------- FULL-DENOM SUMMARY (dynamic, no hard-coded 500) --------
    if ('r0_majority_correct_full' in results[first_iter]['metrics'] and
        'r0_majority_total_full' in results[first_iter]['metrics']):
        full_total = (
            results[first_iter]['metrics'].get('r0_majority_total_full')
            or results[first_iter]['metrics'].get('consensus_full', {}).get('expected', 500)
        )
        print(f"\nFULL-DENOM SUMMARY (out of {full_total} examples):")

        # R0 majority vote
        r0_first_corr = results[first_iter]['metrics']['r0_majority_correct_full']
        r0_last_corr  = results[last_iter]['metrics']['r0_majority_correct_full']
        r0_total_full = results[first_iter]['metrics'].get('r0_majority_total_full', full_total)
        print(
            f"  R0 Majority Vote: {r0_first_corr}/{r0_total_full} ({r0_first_corr/r0_total_full:.3f}) → "
            f"{r0_last_corr}/{r0_total_full} ({r0_last_corr/r0_total_full:.3f})  "
            f"Δ={(r0_last_corr - r0_first_corr)/r0_total_full:+.3f}"
        )

        # R1 consensus (always use full_total as denominator)
        cons_first_corr = (results[first_iter]['metrics']['consensus_full']['correct']
                           if 'consensus_full' in results[first_iter]['metrics']
                           else results[first_iter]['metrics']['test']['consensus_correct'])
        cons_last_corr  = (results[last_iter]['metrics']['consensus_full']['correct']
                           if 'consensus_full' in results[last_iter]['metrics']
                           else results[last_iter]['metrics']['test']['consensus_correct'])
        cons_total_full = full_total
        print(
            f"  R1 Consensus (final): {cons_first_corr}/{cons_total_full} ({cons_first_corr/cons_total_full:.3f}) → "
            f"{cons_last_corr}/{cons_total_full} ({cons_last_corr/cons_total_full:.3f})  "
            f"Δ={(cons_last_corr - cons_first_corr)/cons_total_full:+.3f}"
        )

        # Avg agent full-denom
        def avg_full(metric_key, iter_idx):
            vals = [pd.get(metric_key, 0.0) for pd in results[iter_idx]['metrics'].get('agent_performance_full', {}).values()]
            return np.mean(vals) if vals else 0.0

        avg_r1_first = avg_full('final_all', first_iter)
        avg_r1_last  = avg_full('final_all', last_iter)
        avg_r0_first = avg_full('init_all', first_iter)
        avg_r0_last  = avg_full('init_all', last_iter)
        print(f"  Avg Agent R1 (final_all): {avg_r1_first*full_total:.0f}/{full_total} ({avg_r1_first:.3f}) → "
              f"{avg_r1_last*full_total:.0f}/{full_total} ({avg_r1_last:.3f})  Δ={avg_r1_last-avg_r1_first:+.3f}")
        print(f"  Avg Agent R0 (init_all) : {avg_r0_first*full_total:.0f}/{full_total} ({avg_r0_first:.3f}) → "
              f"{avg_r0_last*full_total:.0f}/{full_total} ({avg_r0_last:.3f})  Δ={avg_r0_last-avg_r0_first:+.3f}")

        # Best agent final_all
        def best_agent_full(iter_idx):
            best_ag, best_val = None, -1.0
            for ag, pd in results[iter_idx]['metrics'].get('agent_performance_full', {}).items():
                val = pd.get('final_all', 0.0)
                if val > best_val:
                    best_val = val
                    best_ag = ag
            return best_ag, best_val

        best_first_ag, best_first_val = best_agent_full(first_iter)
        best_last_ag,  best_last_val  = best_agent_full(last_iter)
        print(f"  Best Agent R1 (final_all): {best_first_ag} {best_first_val*full_total:.0f}/{full_total} ({best_first_val:.3f}) → "
              f"{best_last_ag} {best_last_val*full_total:.0f}/{full_total} ({best_last_val:.3f})  Δ={best_last_val-best_first_val:+.3f}")
        
        # Print clean summary in orchestrator format
        print(f"\n=========== FULL DENOMINATOR METRICS (out of {full_total} examples) ===========")
        print(f"Initial Debate Train Accuracy: {results[first_iter]['metrics'].get('train', {}).get('consensus_accuracy', 0.0):.4f}")
        print(f"Initial Debate Test Accuracy:  {results[first_iter]['metrics'].get('consensus_full', {}).get('acc_expected', results[first_iter]['metrics'].get('test', {}).get('consensus_accuracy', 0.0)):.4f}")
        print(f"Final Debate Train Accuracy:   {results[last_iter]['metrics'].get('train', {}).get('consensus_accuracy', 0.0):.4f}")
        print(f"Final Debate Test Accuracy:    {results[last_iter]['metrics'].get('consensus_full', {}).get('acc_expected', results[last_iter]['metrics'].get('test', {}).get('consensus_accuracy', 0.0)):.4f}")
        
        # Calculate improvements
        initial_train = results[first_iter]['metrics'].get('train', {}).get('consensus_accuracy', 0.0)
        final_train = results[last_iter]['metrics'].get('train', {}).get('consensus_accuracy', 0.0)
        initial_test = results[first_iter]['metrics'].get('consensus_full', {}).get('acc_expected', results[first_iter]['metrics'].get('test', {}).get('consensus_accuracy', 0.0))
        final_test = results[last_iter]['metrics'].get('consensus_full', {}).get('acc_expected', results[last_iter]['metrics'].get('test', {}).get('consensus_accuracy', 0.0))
        
        train_improvement = final_train - initial_train
        test_improvement = final_test - initial_test
        print(f"\nImprovements:")
        print(f"Train Accuracy: {train_improvement:+.4f}")
        print(f"Test Accuracy:  {test_improvement:+.4f}")
        print("=" * 70)
        
        # Print debate stage analysis
        print(f"\n=========== DEBATE STAGE ANALYSIS ===========")
        print(f"Round 0 Avg | Round 0 MajVot | Round 1 Avg | Round 1 MajVot")
        print("-" * 65)
        
        for iteration in iterations:
            metrics = results[iteration]['metrics']
            
            # Get R0 average (init_all)
            agent_perf_full = metrics.get('agent_performance_full', {})
            r0_vals = [pd.get('init_all', 0.0) for pd in agent_perf_full.values() if isinstance(pd, dict)]
            r0_avg = float(np.mean(r0_vals)) if r0_vals else 0.0
            
            # Get R0 majority vote
            r0_maj = metrics.get('r0_majority_correct_full', 0) / full_total
            
            # Get R1 average (final_all)
            r1_vals = [pd.get('final_all', 0.0) for pd in agent_perf_full.values() if isinstance(pd, dict)]
            r1_avg = float(np.mean(r1_vals)) if r1_vals else 0.0
            
            # Get R1 majority vote (consensus)
            if 'consensus_full' in metrics:
                r1_maj = metrics['consensus_full']['acc_expected']
            else:
                r1_maj = metrics['test']['consensus_correct'] / full_total
            
            print(f"Iteration {iteration}: {r0_avg:.4f} | {r0_maj:.4f} | {r1_avg:.4f} | {r1_maj:.4f}")
        print("=" * 65)
        
def generate_summary_plots(
    results: Dict[str, Any],
    output_path: str = "summary.png"
):
    """Generate performance frontier plot comparing pre/post training.
    
    Plots debate pipeline stages (R0 individual -> R0 majority -> R1 individual -> R1 consensus)
    using full metrics, comparing iteration 0 vs final iteration.
    
    Args:
        results: Output from analyze_experiment_performance()
        output_path: Plot save path
    """
    iterations = sorted(results.keys())
    if len(iterations) < 2:
        print("Need at least two iterations to plot.")
        return

    stages = ["Avg 0-shot Round 0", "Majority Vote R0", "Avg 0-shot Round 1", "Majority Vote R1"]

    def _avg_r0_all(metrics_dict: Dict[str, Any]) -> float:
        apf = metrics_dict.get("agent_performance_full", {})
        vals = [d.get("init_all") for d in apf.values() if isinstance(d, dict) and "init_all" in d]
        if vals:
            return float(np.mean(vals))
        # fallback to present R0 if full stats absent
        ap = metrics_dict.get("agent_performance", {})
        vals = [d.get("test_round0_accuracy") for d in ap.values()
                if isinstance(d, dict) and "test_round0_accuracy" in d]
        return float(np.mean(vals)) if vals else 0.0

    def _avg_r1_all(metrics_dict: Dict[str, Any]) -> float:
        apf = metrics_dict.get("agent_performance_full", {})
        vals = [d.get("final_all") for d in apf.values() if isinstance(d, dict) and "final_all" in d]
        if vals:
            return float(np.mean(vals))
        # fallback to present avg if full stats absent
        return float(metrics_dict.get("test", {}).get("avg_agent_accuracy", 0.0))

    def _vote_r0_full(metrics_dict: Dict[str, Any]) -> float:
        corr = float(metrics_dict.get("r0_majority_correct_full", 0.0))
        total = float(metrics_dict.get("r0_majority_total_full", 0.0))
        if total > 0:
            return corr / total
        return float(metrics_dict.get("r0_majority_accuracy", 0.0))

    def _vote_r1_full(metrics_dict: Dict[str, Any]) -> float:
        cf = metrics_dict.get("consensus_full")
        if isinstance(cf, dict) and "acc_expected" in cf:
            return float(cf["acc_expected"])
        return float(metrics_dict.get("test", {}).get("consensus_accuracy", 0.0))

    it0, it1 = iterations[0], iterations[1]
    m0 = results[it0]["metrics"]
    m1 = results[it1]["metrics"]

    avg_r0_0 = _avg_r0_all(m0)
    vote_r0_0 = _vote_r0_full(m0)
    avg_r1_0 = _avg_r1_all(m0)
    vote_r1_0 = _vote_r1_full(m0)

    avg_r0_1 = _avg_r0_all(m1)
    vote_r0_1 = _vote_r0_full(m1)
    avg_r1_1 = _avg_r1_all(m1)
    vote_r1_1 = _vote_r1_full(m1)

    vals0 = [avg_r0_0, vote_r0_0, avg_r1_0, vote_r1_0]
    vals1 = [avg_r0_1, vote_r0_1, avg_r1_1, vote_r1_1]

    plt.figure(figsize=(8, 5))
    plt.plot(stages, vals0, linestyle=":", marker="o", label=f"Pre-sharpening - Iter {it0}")
    plt.plot(stages, vals1, linestyle="-", marker="o", label=f"Post-sharpening - Iter {it1}")
    plt.title("Performance Frontier (Full-Denom)")
    plt.xlabel("Debate Stage")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved summary plot to {output_path}")


def create_structured_experiment_summary(
    results: Dict[str, Any],
    expected_train_size: int = None,
    expected_test_size: int = None
) -> Dict[str, Any]:
    """Create structured JSON summary with present and full metrics.
    
    Args:
        results: Output from analyze_experiment_performance()
        expected_train_size: Expected train examples (inferred if None)
        expected_test_size: Expected test examples (inferred if None)
        
    Returns:
        Dict with 'full_denominator_metrics', 'present_denominator_metrics', 'iterations'
    """
    iterations = sorted(results.keys())
    if len(iterations) < 2:
        return {}
    
    first_iter = iterations[0]
    last_iter = iterations[-1]
    
    # Get full denominator (expected total) - this is the test size
    full_total = (
        results[first_iter]['metrics'].get('r0_majority_total_full')
        or results[first_iter]['metrics'].get('consensus_full', {}).get('expected', expected_test_size or 500)
    )
    
    # If train size not provided, try to infer from the data
    if expected_train_size is None:
        # Look for train_size in the first iteration's metrics
        train_metrics = results[first_iter]['metrics'].get('train', {})
        if train_metrics.get('count', 0) > 0:
            # Estimate train size based on the ratio of loaded to expected test examples
            loaded_test = results[first_iter]['metrics'].get('test', {}).get('count', 0)
            expected_test = full_total
            if loaded_test > 0 and expected_test > 0:
                # Assume same ratio for train
                loaded_train = train_metrics.get('count', 0)
                expected_train_size = int((loaded_train / loaded_test) * expected_test)
            else:
                expected_train_size = 1500  # fallback
        else:
            expected_train_size = 1500  # fallback
    
    # Extract full-denominator metrics for each iteration
    iteration_metrics = {}
    
    for iteration in iterations:
        metrics = results[iteration]['metrics']
        
        # Get consensus metrics (R1 final)
        if 'consensus_full' in metrics:
            consensus_correct = metrics['consensus_full']['correct']
            consensus_accuracy = metrics['consensus_full']['acc_expected']
        else:
            consensus_correct = metrics['test']['consensus_correct']
            consensus_accuracy = consensus_correct / full_total
        
        # Get R0 majority vote metrics
        r0_correct = metrics.get('r0_majority_correct_full', 0)
        r0_accuracy = r0_correct / full_total
        
        # Get average agent R0 and R1 metrics
        agent_perf_full = metrics.get('agent_performance_full', {})
        
        # Average R0 (initial) accuracy
        r0_vals = [pd.get('init_all', 0.0) for pd in agent_perf_full.values() if isinstance(pd, dict)]
        avg_r0_accuracy = float(np.mean(r0_vals)) if r0_vals else 0.0
        
        # Average R1 (final) accuracy  
        r1_vals = [pd.get('final_all', 0.0) for pd in agent_perf_full.values() if isinstance(pd, dict)]
        avg_r1_accuracy = float(np.mean(r1_vals)) if r1_vals else 0.0
        
        iteration_metrics[iteration] = {
            'round_0_avg_0shot_accuracy': avg_r0_accuracy,
            'round_0_majority_vote_accuracy': r0_accuracy,
            'round_1_avg_0shot_accuracy': avg_r1_accuracy,
            'round_1_majority_vote_accuracy': consensus_accuracy,
            'round_0_avg_0shot_correct': int(avg_r0_accuracy * full_total),
            'round_0_majority_vote_correct': r0_correct,
            'round_1_avg_0shot_correct': int(avg_r1_accuracy * full_total),
            'round_1_majority_vote_correct': consensus_correct,
            'total_examples': full_total
        }
    
    # Calculate full-denominator train accuracies
    def get_full_train_accuracy(metrics, expected_size):
        train_metrics = metrics.get('train', {})
        correct = train_metrics.get('consensus_correct', 0)
        return correct / expected_size if expected_size > 0 else 0.0
    
    # Create the structured summary
    summary = {
        'full_denominator_metrics': {
            'total_expected_examples': full_total,
            'initial_debate_train_accuracy': get_full_train_accuracy(results[first_iter]['metrics'], expected_train_size),
            'initial_debate_test_accuracy': results[first_iter]['metrics'].get('consensus_full', {}).get('acc_expected', results[first_iter]['metrics'].get('test', {}).get('consensus_accuracy', 0.0)),
            'final_debate_train_accuracy': get_full_train_accuracy(results[last_iter]['metrics'], expected_train_size),
            'final_debate_test_accuracy': results[last_iter]['metrics'].get('consensus_full', {}).get('acc_expected', results[last_iter]['metrics'].get('test', {}).get('consensus_accuracy', 0.0))
        },
        'present_denominator_metrics': {
            'total_processed_examples': results[first_iter]['metrics'].get('test', {}).get('count', 0),
            'initial_debate_train_accuracy': results[first_iter]['metrics'].get('train', {}).get('consensus_accuracy', 0.0),
            'initial_debate_test_accuracy': results[first_iter]['metrics'].get('test', {}).get('consensus_accuracy', 0.0),
            'final_debate_train_accuracy': results[last_iter]['metrics'].get('train', {}).get('consensus_accuracy', 0.0),
            'final_debate_test_accuracy': results[last_iter]['metrics'].get('test', {}).get('consensus_accuracy', 0.0)
        },
        'iterations': iteration_metrics
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment performance across iterations")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment directory")
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        default=True,
        help="Disable summary visual (enabled by default)"
    )
    args = parser.parse_args()

    try:
        results = analyze_experiment_performance(args.experiment_dir, test_size=args.test_size, agents=args.agents)
        print_performance_comparison(results)
        
        # Create structured summary JSON
        structured_summary = create_structured_experiment_summary(results, expected_test_size=args.test_size)
        if structured_summary:
            summary_path = os.path.join(args.experiment_dir, "experiment_summary_structured.json")
            with open(summary_path, 'w') as f:
                json.dump(structured_summary, f, indent=2)
            print(f"Structured experiment summary saved to {summary_path}")
        
        if args.plot:
            generate_summary_plots(results)
    except Exception as e:
        print(f"Error analyzing experiment: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())