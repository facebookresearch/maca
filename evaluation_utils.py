#!/usr/bin/env python3
"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Evaluation utilities for analyzing model performance, completeness, and consistency.

This module contains utility functions extracted from analysis backup files for:
- Answer extraction and completeness analysis
- Self-consistency sampling and computation
- Pass@k curve computation and averaging
- Statistical confidence intervals
"""

import json
import re
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


# ===== Answer Extraction and Completeness Analysis =====

def is_likely_truncated(response: str) -> bool:
    """
    Check if a response appears to be truncated.

    Args:
        response: The model response text to check

    Returns:
        True if the response appears truncated, False otherwise
    """
    response = response.strip()

    # Common truncation indicators
    if response.endswith('...'):
        return True

    # Check if response is long but doesn't end with typical completion markers
    if len(response) > 500:
        last_char = response[-1] if response else ''
        last_word = response.split()[-1] if response.split() else ''

        # Typical completion endings
        good_endings = ['.', ')', '}', ']', '!', '?']
        if last_char not in good_endings:
            # Check if it ends mid-sentence
            if not response.endswith('\\boxed{') and not last_word.endswith(':'):
                return True

    # Check for incomplete boxed answers
    if '\\boxed{' in response and '}' not in response[response.rfind('\\boxed{'):]:
        return True

    # Check for incomplete sentences at the end
    last_sentence = response.split('.')[-1].strip()
    if len(last_sentence) > 20 and not any(last_sentence.endswith(p) for p in ['.', ')', '}', '!', '?']):
        return True

    return False


def extract_answer_even_if_truncated(
    response: str,
    is_mc: bool = False
) -> Optional[str]:
    """
    Try to extract answer even from truncated responses.

    Args:
        response: The model response text
        is_mc: Whether this is a multiple choice question

    Returns:
        Extracted answer string if found, None otherwise
    """
    if is_mc:
        # Multiple choice extraction
        patterns = [
            r'\\boxed\{([A-Ea-e])\}',
            r'[Tt]he (?:correct )?answer is:?\s*([A-Ea-e])',
            r'[Aa]nswer:?\s*\(?([A-Ea-e])\)?',
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).upper()
    else:
        # Numeric extraction
        # Look for partial boxed answers
        if '\\boxed{' in response:
            boxed_start = response.rfind('\\boxed{')
            partial = response[boxed_start + 7:]
            # Extract whatever is there
            match = re.match(r'([^}]+)', partial)
            if match:
                return match.group(1).strip()

        # Look for "answer is" patterns even if incomplete
        patterns = [
            r'[Tt]he (?:final )?answer is:?\s*([0-9,./-]+)',
            r'[Aa]nswer:?\s*([0-9,./-]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                return matches[-1].strip()

    return None


def is_likely_truncated_fixed(response: str) -> bool:
    """
    Fixed version of truncation detection with adjusted thresholds.

    Args:
        response: The model response text to check

    Returns:
        True if the response appears truncated, False otherwise
    """
    response = response.strip()

    # Common truncation indicators
    if response.endswith('...'):
        return True

    # Check if response is long but doesn't end with typical completion markers
    if len(response) > 500:  # Fixed: lowered from 800
        last_char = response[-1] if response else ''
        last_word = response.split()[-1] if response.split() else ''

        # Typical completion endings
        good_endings = ['.', ')', '}', ']', '!', '?']
        if last_char not in good_endings:
            # Check if it ends mid-sentence
            if not response.endswith('\\boxed{') and not last_word.endswith(':'):
                return True

    # Check for incomplete boxed answers
    if '\\boxed{' in response and '}' not in response[response.rfind('\\boxed{'):]:
        return True

    # Check for incomplete sentences at the end
    last_sentence = response.split('.')[-1].strip()
    if len(last_sentence) > 20 and not any(last_sentence.endswith(p) for p in ['.', ')', '}', '!', '?']):  # Fixed: lowered from 50
        return True

    return False


def extract_answer_strict(
    response: str,
    dataset: str
) -> Optional[str]:
    """
    Use the existing strict parser.

    Note: This function requires the parse_answer function from parser.py

    Args:
        response: The model response text
        dataset: The dataset name for context

    Returns:
        Parsed answer if successful, None otherwise
    """
    try:
        from parser import parse_answer
        return parse_answer(response, dataset)
    except ImportError:
        # Fallback if parser is not available
        return None


def extract_answer_relaxed(
    response: str,
    dataset: str
) -> Optional[str]:
    """
    Relaxed answer extraction that accepts various formats.

    Args:
        response: The model response text
        dataset: The dataset name for determining format

    Returns:
        Extracted answer if found, None otherwise
    """
    # First try strict if available
    strict_answer = extract_answer_strict(response, dataset)
    if strict_answer:
        return strict_answer

    # Determine if multiple choice
    is_mc = dataset in ['csqa', 'gpqa', 'mathqa']

    if is_mc:
        # Multiple choice patterns
        patterns = [
            r'\\boxed\{([A-E])\}',
            r'[Tt]he (?:correct )?answer is:?\s*([A-E])',
            r'[Aa]nswer:?\s*\(?([A-E])\)?',
            r'[Ss]o,?\s*(?:the )?(?:correct )?answer is:?\s*\(?([A-E])\)?',
            r'[Tt]herefore,?\s*(?:the )?(?:correct )?answer is:?\s*\(?([A-E])\)?',
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).upper()

        # Last resort: find the last single letter option
        all_mentions = re.findall(r'\b([A-E])\)', response)
        if all_mentions:
            return all_mentions[-1]
    else:
        # Numeric patterns
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'[Tt]he (?:final )?answer is:?\s*([0-9,./-]+)',
            r'[Aa]nswer:?\s*([0-9,./-]+)',
            r'= ([0-9,./-]+)$',  # End of line
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                return matches[-1].strip()

    return None


# ===== Self-Consistency and Sampling Analysis =====

def normalize_answer(val: Optional[str]) -> str:
    """
    Normalize an answer string for comparison.

    Args:
        val: The answer value to normalize

    Returns:
        Normalized answer string
    """
    if val is None:
        return ""
    s = str(val).strip().lower()
    # Strip common wrappers like \boxed{...}
    if s.startswith("\\boxed{") and s.endswith("}"):
        s = s[len("\\boxed{"):-1].strip()
    return s


def load_sampling_consistency(
    samples_path: Path,
    n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sampling consistency over t for a JSONL cache.

    For each prompt and each t in [1..min(n, len(parsed))], compute max vote share among the first t
    parsed answers (after normalization). Then average across prompts for each t.

    Args:
        samples_path: Path to JSONL file containing sample data
        n: Maximum t to consider

    Returns:
        Tuple of (t, mean_consistency, counts) arrays where:
        - t: array [T] of t values
        - mean_consistency: array [T] with mean of max_vote_share per t
        - counts: array [T] with number of prompts contributing at each t
    """
    per_prompt_curves: List[np.ndarray] = []

    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            parsed: List[Optional[str]] = rec.get("parsed", [])
            if not parsed:
                continue
            # Keep None values as disagreements, normalize non-None values
            norm = [normalize_answer(a) if a is not None else None for a in parsed]
            if not norm:
                continue
            L = min(n, len(norm))
            curve = np.zeros(L, dtype=float)
            for i in range(1, L + 1):
                # Count non-None responses only for majority calculation
                valid_responses = [a for a in norm[:i] if a is not None]
                if not valid_responses:
                    curve[i - 1] = 0.0  # All None responses = 0 consistency
                    continue
                counts = Counter(valid_responses)
                max_votes = max(counts.values()) if counts else 0
                # Denominator is total responses (including None), numerator is max votes among valid
                curve[i - 1] = float(max_votes) / float(i) if i > 0 else 0.0
            per_prompt_curves.append(curve)

    if not per_prompt_curves:
        return np.array([]), np.array([]), np.array([])

    max_t = max(len(c) for c in per_prompt_curves)
    sums = np.zeros(max_t, dtype=float)
    counts = np.zeros(max_t, dtype=int)
    for c in per_prompt_curves:
        L = len(c)
        sums[:L] += c
        counts[:L] += 1
    with np.errstate(invalid="ignore"):
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    t = np.arange(1, max_t + 1)
    return t, means, counts


def load_mv_from_jsonl(
    samples_path: Path,
    n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MV@t by voting over final parsed answers among first t samples.
    Ties (no unique mode) count as incorrect.

    Args:
        samples_path: Path to JSONL file containing sample data
        n: Maximum number of samples to consider

    Returns:
        Tuple of (t, mv_avg, totals) arrays where:
        - t: array [T] of t values
        - mv_avg: array [T] with majority vote accuracy per t
        - totals: array [T] with number of examples contributing at each t
    """
    mv_curves: List[np.ndarray] = []

    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            parsed: List[str] = rec.get("parsed", [])
            gt: str = rec.get("ground_truth", "")
            if not parsed:
                continue
            gt_norm = normalize_answer(gt)
            mv_binary: List[float] = []
            for t in range(1, min(n, len(parsed)) + 1):
                answers_t = [normalize_answer(a) for a in parsed[:t] if a is not None]
                if not answers_t:
                    mv_binary.append(0.0)
                    continue
                counts = Counter(answers_t)
                most_common = counts.most_common()
                if len(most_common) == 1 or (len(most_common) > 1 and most_common[0][1] > most_common[1][1]):
                    modal_answer = most_common[0][0]
                    mv_binary.append(1.0 if modal_answer == gt_norm else 0.0)
                else:
                    # tie: no unique mode → treat as incorrect
                    mv_binary.append(0.0)
            if mv_binary:
                mv_curves.append(np.array(mv_binary, dtype=float))

    if not mv_curves:
        return np.array([]), np.array([]), np.array([])

    max_t = max(len(c) for c in mv_curves)
    successes = np.zeros(max_t, dtype=float)
    totals = np.zeros(max_t, dtype=int)
    for c in mv_curves:
        L = len(c)
        successes[:L] += c
        totals[:L] += 1
    with np.errstate(invalid="ignore"):
        mv_avg = np.divide(successes, totals, out=np.zeros_like(successes, dtype=float), where=totals>0)
    t = np.arange(1, max_t + 1)
    return t, mv_avg, totals


# ===== Statistical Utilities =====

def wilson_ci(
    p: np.ndarray,
    n: np.ndarray,
    z: float = 1.96
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Wilson confidence interval for binomial proportions.

    Args:
        p: Array of observed proportions
        n: Array of sample sizes
        z: Z-score for confidence level (1.96 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) arrays
    """
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = 1 + (z**2)/n
        center = (p + (z**2)/(2*n)) / denom
        margin = (z * np.sqrt((p*(1-p)/n) + (z**2)/(4*(n**2)))) / denom
        lower = center - margin
        upper = center + margin
    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)
    return lower, upper


# ===== Pass@k Curve Computation =====

def compute_pass_curve(
    sampled_texts: List[str],
    ground_truth: str,
    dataset: str
) -> List[int]:
    """
    Compute cumulative "any correct so far" for t=1..n (0/1 per t).

    Note: This function requires parse_answer and grade_answer from parser.py

    Args:
        sampled_texts: List of generated text responses
        ground_truth: The correct answer
        dataset: Dataset name for parsing context

    Returns:
        List of 0/1 values indicating if any response up to position t was correct
    """
    try:
        from parser import parse_answer, grade_answer
    except ImportError:
        # Return dummy data if parser not available
        return [0] * len(sampled_texts)

    correctness = []
    seen_correct = False
    for txt in sampled_texts:
        parsed = parse_answer(txt, dataset=dataset)
        is_correct = grade_answer(parsed, ground_truth)
        seen_correct = seen_correct or is_correct
        correctness.append(1 if seen_correct else 0)
    return correctness


def average_curves(
    curves: List[List[int]],
    n: int
) -> np.ndarray:
    """
    Average multiple pass@k curves.

    Args:
        curves: List of curves, each being a list of 0/1 values
        n: Length to truncate/pad curves to

    Returns:
        Numpy array of averaged curve values
    """
    if not curves:
        return np.zeros(n)
    arr = np.array([c[:n] for c in curves], dtype=float)
    return arr.mean(axis=0)


def batch_sample_completions(
    agent,
    prompt: str,
    dataset: str,
    n: int,
    temperature: float,
    top_p: float
) -> List[str]:
    """
    Sample multiple completions from an agent in batch.

    Note: This function requires specific Agent class and helper functions

    Args:
        agent: The model agent to sample from
        prompt: The input prompt
        dataset: Dataset name for formatting
        n: Number of samples to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        List of generated text responses
    """
    try:
        import asyncio
        from debate import construct_question_prompt

        context = {"role": "user", "content": construct_question_prompt(prompt, dataset)}
        device = agent.get_device()

        # Greedy: generate once then replicate to avoid redundant compute
        if temperature == 0.0:
            outs = asyncio.run(agent.batch_generate(
                contexts_list=[[context]],
                device=device,
                temperature=0.0,
                top_p=top_p,
            ))
            text = outs[0]["choices"][0]["message"]["content"]
            return [text for _ in range(n)]

        # Sampling: generate n samples in one batch
        contexts = [[context] for _ in range(n)]
        outs = asyncio.run(agent.batch_generate(
            contexts_list=contexts,
            device=device,
            temperature=temperature,
            top_p=top_p,
        ))
        texts = [o["choices"][0]["message"]["content"] for o in outs]
        return texts
    except ImportError:
        # Return dummy data if dependencies not available
        return ["dummy response"] * n


def select_prompts(
    test_examples: List[Dict],
    k: int,
    seed: int
) -> List[Dict]:
    """
    Select k prompts randomly from test examples.

    Args:
        test_examples: List of test examples (dicts with questions/answers)
        k: Number of prompts to select (<=0 for all)
        seed: Random seed for reproducibility

    Returns:
        List of selected examples
    """
    import random
    rng = random.Random(seed)
    if k <= 0 or k >= len(test_examples):
        return list(test_examples)
    return rng.sample(test_examples, k)


# ===== Utility Functions =====

def parse_meta_from_name(name: str) -> Tuple[Optional[str], bool, Optional[float]]:
    """
    Parse metadata from filename.

    Args:
        name: Filename to parse

    Returns:
        Tuple of (which_model, is_quantized, temperature)
    """
    import os
    base = os.path.basename(name)
    core = base.replace(".jsonl", "").replace(".csv", "")
    tokens = core.split("_")
    which: Optional[str] = None
    temp: Optional[float] = None
    quant = False
    for t in tokens:
        if t in ("base", "post"):
            which = t
        elif t.startswith("temp"):
            try:
                temp = float(t.replace("temp", ""))
            except Exception:
                temp = None
        elif t in ("qb", "qp", "quant", "quantized"):
            quant = True
    return which, quant, temp


def label_for(
    which: str,
    quantized: bool,
    temp: float,
    metric: str
) -> str:
    """
    Generate label string for plots.

    Args:
        which: Model type ("base" or "post")
        quantized: Whether model is quantized
        temp: Temperature value
        metric: Metric name

    Returns:
        Formatted label string
    """
    side = "Post" if which == "post" else "Base"
    quant = "4-bit" if quantized else "full"
    mode = "greedy" if float(temp) == 0.0 else f"temp={temp}"
    return f"{side} {quant} – {mode} – {metric}"