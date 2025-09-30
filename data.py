"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""Dataset loading and processing for multi-agent debate training."""

import os

import json
import random
import numpy as np
from glob import glob
from typing import List, Dict, Any
from parser import parse_answer, grade_answer
from datasets import Dataset, load_dataset

from debate import construct_message, majority_vote

def get_data(
    dataset_name: str = 'gsm8k',
    dataset_path: str = 'data',
    train_size: int = 500,
    test_size: int = 500,
    seed: int = 0
) -> List[Dict[str, Any]]:
    split_path = f"{dataset_path}/splits/{dataset_name}"
    if os.path.exists(f'{split_path}/{seed}_{train_size}_{test_size}.json'):
        with open(f'{split_path}/{seed}_{train_size}_{test_size}.json', "r") as f:
            return json.load(f)

    data = []
    cache_dir = f"{dataset_path}/raw"

    if dataset_name == 'gsm8k':
        raw_data = load_dataset(path='openai/gsm8k',
                                name='main', cache_dir=cache_dir)
        train_subset = raw_data['train'].shuffle(seed=seed)[:train_size]
        test_subset = raw_data['test'].shuffle(seed=seed)[:test_size]

        data.extend([{'question': q, 'split': 'train',
                      'answer_cot': train_subset['answer'][idx],
                      'answer': train_subset['answer'][idx].split('\n####')[-1].strip(), }
                     for idx, q in enumerate(train_subset['question'])])

        data.extend([{'question': q, 'split': 'test',
                      'answer_cot': test_subset['answer'][idx],
                      'answer': test_subset['answer'][idx].split('\n####')[-1].strip(), }
                     for idx, q in enumerate(test_subset['question'])])


    elif dataset_name == 'gpqa':
        choice_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        raw_data = load_dataset(
            path='Wanfq/gpqa', name='gpqa_main', cache_dir=cache_dir)
        # GPQA only has train split, so we'll use it for both train and test
        # Users can adjust train_size and test_size as needed
        all_data = raw_data['train'].shuffle(seed=seed)
        train_subset = all_data[:train_size]
        test_subset = all_data[train_size:train_size + test_size]

        # Create a local random generator for consistent shuffling within each subset
        rng = random.Random(seed)

        # Process training data with randomized answer positions
        for idx, q in enumerate(train_subset['Question']):
            # Create list of all choices (correct + 3 incorrect)
            choices = [
                train_subset['Correct Answer'][idx],
                train_subset['Incorrect Answer 1'][idx], 
                train_subset['Incorrect Answer 2'][idx],
                train_subset['Incorrect Answer 3'][idx]
            ]
            
            # Shuffle the choices while tracking correct answer position
            choice_order = list(range(4))
            rng.shuffle(choice_order)
            shuffled_choices = [choices[i] for i in choice_order]
            
            # Find where the correct answer ended up (it was originally at index 0)
            correct_position = choice_order.index(0)
            correct_letter = choice_map[correct_position]
            
            data.append({
                'question': (
                    f"{q}"
                    f" \\n A) {shuffled_choices[0]}"
                    f" \\n B) {shuffled_choices[1]}"
                    f" \\n C) {shuffled_choices[2]}"
                    f" \\n D) {shuffled_choices[3]}"
                ),
                'split': 'train', 
                'answer_cot': train_subset['Explanation'][idx],
                'answer': correct_letter
            })

        # Process test data with randomized answer positions  
        for idx, q in enumerate(test_subset['Question']):
            # Create list of all choices (correct + 3 incorrect)
            choices = [
                test_subset['Correct Answer'][idx],
                test_subset['Incorrect Answer 1'][idx],
                test_subset['Incorrect Answer 2'][idx], 
                test_subset['Incorrect Answer 3'][idx]
            ]
            
            # Shuffle the choices while tracking correct answer position
            choice_order = list(range(4))
            rng.shuffle(choice_order)
            shuffled_choices = [choices[i] for i in choice_order]
            
            # Find where the correct answer ended up (it was originally at index 0)
            correct_position = choice_order.index(0)
            correct_letter = choice_map[correct_position]
            
            data.append({
                'question': (
                    f"{q}"
                    f" \\n A) {shuffled_choices[0]}"
                    f" \\n B) {shuffled_choices[1]}"
                    f" \\n C) {shuffled_choices[2]}"
                    f" \\n D) {shuffled_choices[3]}"
                ),
                'split': 'test',
                'answer_cot': test_subset['Explanation'][idx],
                'answer': correct_letter
            })

    elif dataset_name == 'arithmatic':
        for idx in range(train_size + test_size):
            a, b, c, d, e, f = np.random.randint(0, 30, size=6)
            answer = a + b * c + d - e * f
            data.append({
                'question': f"What is the result of {a}+{b}*{c}+{d}-{e}*{f}?",
                'answer_cot': '',
                'answer': f'{answer}',
                'split': 'test' if idx >= train_size else 'train'})

    elif dataset_name == 'math':
        def get_hard_problems(jsons, data_size):
            random.shuffle(jsons)
            hard_problems = []

            for json_file in jsons:
                raw_data = json.load(open(json_file, "r"))
                if ('1' in raw_data['level']) or ('2' in raw_data['level']) or ('3' in raw_data['level']):
                    hard_problems.append(raw_data)
            random.shuffle(hard_problems)
            random_subset = hard_problems[:data_size]
            return random_subset

        train_subset = get_hard_problems(
            sorted(glob(f"{cache_dir}/MATH/train/*/*.json")), train_size)
        test_subset = get_hard_problems(
            sorted(glob(f"{cache_dir}/MATH/test/*/*.json")), test_size)

        data.extend([{
            'question': example['problem'],
            'answer_cot': example['solution'],
            'answer': parse_answer(example['solution']),
            'split': 'train'
        } for example in train_subset])

        data.extend([{
            'question': example['problem'],
            'answer_cot': example['solution'],
            'answer': parse_answer(example['solution']),
            'split': 'test'
        } for example in test_subset])

    elif dataset_name == 'aime2024':
        raw_data = load_dataset(path = "Maxwell-Jia/AIME_2024", cache_dir = cache_dir)
        train_subset = raw_data['train'].shuffle(seed=seed)[:train_size]
        test_subset = raw_data['train'].shuffle(seed=seed)[:train_size]

        data.extend([{'question': q, 'split': 'train',
                      'answer_cot': train_subset['Solution'][idx],
                      'answer': str(train_subset['Answer'][idx]), }
                     for idx, q in enumerate(train_subset['Problem'])])

        data.extend([{'question': q, 'split': 'test',
                      'answer_cot': test_subset['Solution'][idx],
                      'answer': str(test_subset['Answer'][idx]), }
                     for idx, q in enumerate(test_subset['Problem'])])
    
    elif dataset_name == 'amc':
        raw_data = load_dataset(path = 'knoveleng/AMC-23', cache_dir = cache_dir)
        #dataset contains only 40 records
        #so same will be used for train and test, or only for test with model trained on another dataset
        train_subset = raw_data['train'].shuffle(seed=seed)[:train_size]
        test_subset = raw_data['train'].shuffle(seed=seed)[:train_size]

        data.extend([{'question': q, 'split': 'train',
                      'answer_cot': '',
                      'answer': train_subset['answer'][idx], }
                     for idx, q in enumerate(train_subset['problem'])])

        data.extend([{'question': q, 'split': 'test',
                      'answer_cot': '',
                      'answer': test_subset['answer'][idx], }
                     for idx, q in enumerate(test_subset['problem'])])

    elif dataset_name == 'aime_amc':
        raw_data_aime = load_dataset(path = "Maxwell-Jia/AIME_2024", cache_dir = cache_dir)
        raw_data_amc = load_dataset(path = 'knoveleng/AMC-23', cache_dir = cache_dir)

        train_subset = raw_data_aime['train'].shuffle(seed=seed)[:train_size]
        test_subset = raw_data_amc['train'].shuffle(seed=seed)[:test_size]

        data.extend([{'question': q, 'split': 'train',
                      'answer_cot': train_subset['Solution'][idx],
                      'answer': str(train_subset['Answer'][idx]), }
                     for idx, q in enumerate(train_subset['Problem'])])
        
        data.extend([{'question': q, 'split': 'test',
                      'answer_cot': '',
                      'answer': test_subset['answer'][idx], }
                     for idx, q in enumerate(test_subset['problem'])])
        
    elif dataset_name == 'amc_aime':
        raw_data_amc = load_dataset(path = 'knoveleng/AMC-23', cache_dir = cache_dir)
        raw_data_aime = load_dataset(path = "Maxwell-Jia/AIME_2024", cache_dir = cache_dir)

        train_subset = raw_data_amc['train'].shuffle(seed=seed)[:train_size]
        test_subset = raw_data_aime['train'].shuffle(seed=seed)[:test_size]

        data.extend([{'question': q, 'split': 'train',
                      'answer_cot': '',
                      'answer': train_subset['answer'][idx], }
                     for idx, q in enumerate(train_subset['problem'])])

        data.extend([{'question': q, 'split': 'test',
                      'answer_cot': test_subset['Solution'][idx],
                      'answer': str(test_subset['Answer'][idx]), }
                     for idx, q in enumerate(test_subset['Problem'])])

    elif dataset_name == 'svamp':
        raw_data = load_dataset(path='ChilleD/SVAMP', cache_dir=cache_dir)
        # SVAMP has explicit train (700) and test (300) splits
        train_subset = raw_data['train'].shuffle(seed=seed).select(range(train_size))
        test_subset = raw_data['test'].shuffle(seed=seed).select(range(test_size))

        # Process training data
        for example in train_subset:
            data.append({
                'question': example['question_concat'],
                'split': 'train',
                'answer_cot': '',
                'answer': str(example['Answer'])
            })

        # Process test data
        for example in test_subset:
            data.append({
                'question': example['question_concat'],
                'split': 'test',
                'answer_cot': '',
                'answer': str(example['Answer'])
            })

    elif dataset_name == 'mathqa':
        raw_data = load_dataset(path='allenai/math_qa', cache_dir=cache_dir, trust_remote_code=True)
        # MathQA has train, validation, and test splits
        train_subset = raw_data['train'].shuffle(seed=seed).select(range(train_size))
        test_subset = raw_data['test'].shuffle(seed=seed).select(range(test_size))

        # Process training data
        for example in train_subset:
            data.append({
                'question': f"{example['Problem']}\n{example['options']}",
                'split': 'train',
                'answer_cot': example['Rationale'],
                'answer': example['correct']
            })

        # Process test data
        for example in test_subset:
            data.append({
                'question': f"{example['Problem']}\n{example['options']}",
                'split': 'test',
                'answer_cot': example['Rationale'],
                'answer': example['correct']
            })

    elif dataset_name == 'csqa':
        choice_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        raw_data = load_dataset(
            path='tau/commonsense_qa', cache_dir=cache_dir)
        train_subset = raw_data['train'].shuffle(seed=seed)[:train_size]
        test_subset = raw_data['validation'].shuffle(seed=seed)[:test_size]

        data.extend([{'question': (
            f"{q}"
            f" \n A) {train_subset['choices'][idx]['text'][0]}"
            f" \n B) {train_subset['choices'][idx]['text'][1]}"
            f" \n C) {train_subset['choices'][idx]['text'][2]}"
            f" \n D) {train_subset['choices'][idx]['text'][3]}"
            f" \n E) {train_subset['choices'][idx]['text'][4]}"),
            'split': 'train', 'answer_cot': '',
            'answer': train_subset['answerKey'][idx], }
            for idx, q in enumerate(train_subset['question'])])

        data.extend([{'question': (
            f"{q}"
            f" \n A) {test_subset['choices'][idx]['text'][0]}"
            f" \n B) {test_subset['choices'][idx]['text'][1]}"
            f" \n C) {test_subset['choices'][idx]['text'][2]}"
            f" \n D) {test_subset['choices'][idx]['text'][3]}"
            f" \n E) {test_subset['choices'][idx]['text'][4]}"),
            'split': 'test', 'answer_cot': '',
            'answer': test_subset['answerKey'][idx], }
            for idx, q in enumerate(test_subset['question'])])



    else:
        raise ValueError("Invalid dataset name")

    os.makedirs(split_path, exist_ok=True)
    with open(f'{split_path}/{seed}_{train_size}_{test_size}.json', "w") as f:
        json.dump(data, f)

    return data


def prepare_finetune_data(
    debate: Dict[str, Any],
    agent_idx: int
) -> List[Dict[str, Any]]:
    data = []
    for k, v in debate.items():
        if k == 'metrics' or v['split'] == 'test' or v['agent_answers'][agent_idx] == 0:
            continue
        else:
            # Critic data --> Full trace
            data.append({
                "messages": v['context'][agent_idx]
            })
    return data


def prepare_post_train_data(
    debate: Dict[str, Any],
    agent_idx: int,
    include_debate_context: bool = False,
    dataset: str = 'gsm8k',
    diversity_prompt: bool = False,
    use_initial_responses: bool = False
) -> Dataset:
    data = []
    for k, v in debate.items():
        if k == 'metrics' or v['split'] == 'test':
            continue
        
        # Determine the target answer for the reward signal
        if use_initial_responses:
            # Compute initial round consensus from initial responses
            from debate import majority_vote
            from parser import parse_answer
            
            initial_answers = []
            for agent_context in v['context']:
                if len(agent_context) > 1:  # Ensure we have initial response
                    initial_response = agent_context[1]['content']  # Index 1 = first assistant response
                    parsed_answer = parse_answer(initial_response, dataset)
                    if parsed_answer is not None:
                        initial_answers.append(parsed_answer)
            
            if initial_answers:
                target_answer = majority_vote(initial_answers)
                # print(f"Debug: Computed initial round consensus: {target_answer} from {len(initial_answers)} initial responses")
            else:
                target_answer = None
        else:
            # Use final round consensus (default behavior)
            target_answer = v.get('consensus_answer')
        
            
        # Skip if there's no target answer to train on
        if not target_answer:
            continue
            
        # Determine which round to use for responses
        response_idx = 1 if use_initial_responses else -1
        
        if include_debate_context:
            # Get other agents' contexts and limit to 5
            other_agents = [context for idx, context in enumerate(v['context']) if idx != agent_idx]
            random.shuffle(other_agents)
            other_agents = other_agents[:5]  # Limit to 5 other agents like in debate
            
            # Use the same message construction as in debate
            message = construct_message(
                agents=other_agents,
                question=v['context'][0][0]['content'],
                dataset=dataset,  # Use the passed dataset parameter
                summary=None,
                idx=response_idx,
                diversity_prompt=diversity_prompt  # Use the passed diversity_prompt parameter
            )
            # Format messages into a single string
            prompt = f"{v['context'][0][0]['content']}\n\n{message['content']}"
        else:
            prompt = v['context'][0][0]['content']  # Just the original question
            
        # Guard: ensure context has enough messages for the requested index
        if use_initial_responses:
            # For initial responses, need at least 2 messages (user + assistant)
            completion = v['context'][agent_idx][response_idx]['content'] if len(v['context'][agent_idx]) > 1 else ''
            agent_responses = [response[response_idx]['content'] if len(response) > 1 else '' for response in v['context']]
        else:
            # For final responses, use -1 index (always safe if context exists)
            completion = v['context'][agent_idx][response_idx]['content'] if v['context'][agent_idx] else ''
            agent_responses = [response[response_idx]['content'] if response else '' for response in v['context']]
        
        item = {
            'prompt': prompt,
            'completion': completion,
            'majority_answer': target_answer,
            'ground_truth': v['ground_truth'],
            'agent_responses': agent_responses,
            'example_id': k
        }
        data.append(item)
    return Dataset.from_list(data)

# =====================================================================================
#  KTO DATA PREPARATION
# =====================================================================================

def prepare_kto_data(
    debate: dict,
    *,
    include_debate_context: bool = False,
    pref_debate_context: bool = False,
    dataset: str = 'gsm8k',
    diversity_prompt: bool = False,
    use_majority_vote: bool = True,
    use_initial_responses: bool = False,
) -> Dataset:
    """Prepare an *un-paired* preference dataset for TRL's ``KTOTrainer``.

    The output schema matches the default expected by KTO when using
    unpaired data::

        {"prompt": <str>, "completion": <str>, "label": <bool>}

    • ``label=True``  → desirable / preferred (i.e. correct w.r.t the oracle)
    • ``label=False`` → undesirable / non-preferred (i.e. incorrect)

    One record is produced **per agent** for every debate example that has at
    least one correct and one incorrect response.  Examples where *all*
    agents are correct or *all* agents are wrong are still retained (KTO can
    handle class imbalance; users can later re-weight via
    ``desirable_weight`` / ``undesirable_weight``).
    """

    from datasets import Dataset

    data = []

    # Iterate over all debate entries
    for example_id, v in debate.items():
        # Skip metrics and test-set examples – KTO is a training-only phase
        if example_id == 'metrics' or v.get('split') == 'test':
            continue

        # Decide oracle answer: use final-round consensus by default
        oracle = v.get('consensus_answer') if use_majority_vote else v.get('ground_truth')


        # Clean consensus answer for MathQA and CSQA datasets (remove letter prefix if present)
        if dataset in ['mathqa', 'csqa'] and oracle and isinstance(oracle, str):
            if ') ' in oracle:
                oracle = oracle.split(') ')[0] + ')'
            if oracle.endswith(')'):
                oracle = oracle[:-1]
        
        # If we still have no oracle, skip this example entirely
        if not oracle:
            continue

        # ------------------------------------------------------------------
        # Prompt construction  ------------------------------------------------
        # 1. pref_debate_context  → common prompt for ALL agents based on the
        #    assistant messages from the previous round (idx = len-3).
        # 2. include_debate_context → legacy behaviour (peer feedback from the
        #    *final* round, varies per agent).
        # 3. neither flag → just the original question.
        # ------------------------------------------------------------------

        shared_prompt_pref = None  # will be set if needed

        if pref_debate_context:
            # Need at least two rounds: sequence length must be ≥4
            if len(v['context'][0]) < 4:
                # Not enough history to build previous-round context; skip example
                continue

            second_last_idx = len(v['context'][0]) - 3  # assistant from prev round

            message = construct_message(
                agents=v['context'],            # all agents
                question=v['context'][0][0]['content'],
                dataset=dataset,
                summary=None,
                idx=second_last_idx,
                diversity_prompt=diversity_prompt,
            )

            shared_prompt_pref = f"{v['context'][0][0]['content']}\n\n{message['content']}"

        elif include_debate_context:
            # Existing behaviour: last-round peer feedback without own response
            # We'll build a richer prompt that includes (up to) five other
            # agents' messages, exactly the same way construct_message() is
            # used in debate.py and prepare_post_train_data.

            # Gather other agents' contexts once so we can reuse per agent.
            all_agent_ctxs = v['context']

            # Pre-compute question text (first user message)
            question_prompt = all_agent_ctxs[0][0]['content']

            # For each agent we'll dynamically create a prompt that excludes
            # that agent's own response in the *feedback* section but keeps
            # the common question.
        else:
            # Simple prompt: original question only (first message)
            base_prompt = v['context'][0][0]['content']

        # ------------------------------------------------------------------
        # Iterate over agents and create records
        # ------------------------------------------------------------------
        # Determine which round to use for responses (final-only for KTO)
        response_idx = -1
        
        for agent_idx, agent_ctx in enumerate(v['context']):
            # Always use final responses for KTO labels
            reply = agent_ctx[response_idx]['content'] if agent_ctx else ''

            # Determine correctness vs oracle
            parsed_reply = parse_answer(reply, dataset)
            is_correct = grade_answer(parsed_reply, oracle)

            # Build the final prompt for this record
            if pref_debate_context:
                prompt = shared_prompt_pref
            elif include_debate_context:
                # Existing behaviour: last-round peer feedback (excluding current agent)
                other_agents = [c for idx, c in enumerate(v['context']) if idx != agent_idx]
                random.shuffle(other_agents)
                other_agents = other_agents[:5]

                message = construct_message(
                    agents=other_agents,
                    question=question_prompt,
                    dataset=dataset,
                    summary=None,
                    idx=response_idx,
                    diversity_prompt=diversity_prompt,
                )

                prompt = f"{question_prompt}\n\n{message['content']}"
            else:
                prompt = base_prompt

            data.append({
                'prompt': prompt,
                'completion': reply,
                'label': bool(is_correct),
                'example_id': example_id,
                'agent_idx': agent_idx,
            })

    return Dataset.from_list(data)

def prepare_dpo_data(
    debate: dict,
    *,
    include_debate_context: bool = False,
    pref_debate_context: bool = False,
    dataset: str = 'gsm8k',
    diversity_prompt: bool = False,
    use_majority_vote: bool = True,
    use_initial_responses: bool = False,
) -> Dataset:
    """Prepare *paired* preference data for TRL's ``DPOTrainer``.

    Each output record has the canonical DPO schema::

        {"prompt": <str>, "chosen": <str>, "rejected": <str>}

    A record is produced for **every** (chosen, rejected) pair where *chosen*
    is correct with respect to the oracle answer and *rejected* is incorrect.
    If an example has *k* correct and *m* incorrect answers we create
    ``k × m`` pairs.  Examples where all agents are correct **or** all are
    wrong are skipped because no preference signal can be formed.

    Parameters
    ----------
    debate : dict
        Debate data structure generated by ``debate.py``.
    include_debate_context : bool, default=False
        Whether to prepend up-to five other agents' messages (excluding the
        chosen/rejected ones) to the question – identical to
        ``prepare_kto_data`` / ``prepare_post_train_data`` behaviour.
    dataset : str, default='gsm8k'
        Downstream dataset identifier used by ``parse_answer`` / ``grade_answer``.
    diversity_prompt : bool, default=False
        Passed through to ``construct_message`` for diversity prompting.
    use_majority_vote : bool, default=True
        If *True* the oracle answer is the consensus/majority vote, otherwise
        the ground-truth label is used.
    """

    from datasets import Dataset

    data = []

    # ----------------------------------------------------------------------
    # Iterate over debate examples
    # ----------------------------------------------------------------------
    for example_id, v in debate.items():
        # Skip global metrics dict and held-out test examples
        if example_id == 'metrics' or v.get('split') == 'test':
            continue

        # Decide oracle answer
        if use_initial_responses and use_majority_vote:
            # For initial responses, compute initial round majority vote
            initial_responses = []
            for agent_ctx in v['context']:
                if agent_ctx and len(agent_ctx) > 1:
                    initial_response = agent_ctx[1]['content']
                    try:
                        parsed_answer = parse_answer(initial_response, dataset)
                        initial_responses.append(parsed_answer)
                    except Exception:
                        continue  # Skip responses that can't be parsed
            
            if len(initial_responses) >= 2:
                # Filter out None values before computing majority vote
                valid_responses = [resp for resp in initial_responses if resp is not None]
                if len(valid_responses) >= 2:
                    oracle = majority_vote(valid_responses)
                else:
                    # No valid responses - skip this example
                    continue
            else:
                # Fall back to stored consensus or ground truth
                oracle = v.get('consensus_answer') if use_majority_vote else v.get('ground_truth')
        else:
            # Use stored consensus answer (from final round)
            oracle = v.get('consensus_answer') if use_majority_vote else v.get('ground_truth')
            
        if not oracle:
            # Cannot judge correctness without an oracle – skip
            continue

        # MathQA and CSQA-specific: clean consensus-style oracle labels like "c) 410" → "c"
        if dataset in ['mathqa', 'csqa'] and isinstance(oracle, str):
            if ') ' in oracle:
                oracle = oracle.split(') ')[0] + ')'
            if oracle.endswith(')'):
                oracle = oracle[:-1]

        # Pre-compute question text
        question_prompt = v['context'][0][0]['content']

        shared_prompt_pref = None

        if pref_debate_context:
            if len(v['context'][0]) < 4:
                continue
            second_last_idx = len(v['context'][0]) - 3
            message = construct_message(
                agents=v['context'],
                question=question_prompt,
                dataset=dataset,
                summary=None,
                idx=second_last_idx,
                diversity_prompt=diversity_prompt,
            )
            shared_prompt_pref = f"{question_prompt}\n\n{message['content']}"

        if not include_debate_context and shared_prompt_pref is None:
            base_prompt = question_prompt

        # ------------------------------------------------------------------
        # Partition agents into correct / incorrect buckets
        # ------------------------------------------------------------------
        # Determine which round to use for responses
        response_idx = 1 if use_initial_responses else -1
        
        correct_agents = []   # list[(idx, reply)]
        incorrect_agents = [] # list[(idx, reply)]

        for agent_idx, agent_ctx in enumerate(v['context']):
            # Guard: ensure context has enough messages for the requested index
            if use_initial_responses:
                # For initial responses, need at least 2 messages (user + assistant)
                reply = agent_ctx[response_idx]['content'] if agent_ctx and len(agent_ctx) > 1 else ''
            else:
                # For final responses, use -1 index (always safe if context exists)
                reply = agent_ctx[response_idx]['content'] if agent_ctx else ''

            # Classify correctness (empty replies => incorrect)
            is_correct = False
            if reply:
                try:
                    is_correct = grade_answer(parse_answer(reply, dataset), oracle)
                except Exception:
                    # If parsing/grading fails treat as incorrect to avoid crashes
                    is_correct = False

            bucket = correct_agents if is_correct else incorrect_agents
            bucket.append((agent_idx, reply))

        # Need at least one agent on *both* sides to create preference pairs
        if not correct_agents or not incorrect_agents:
            continue

        # ------------------------------------------------------------------
        # Create all (chosen, rejected) pairs  (k × m pairs per example)
        # ------------------------------------------------------------------
        for chosen_idx, chosen_reply in correct_agents:
            for rejected_idx, rejected_reply in incorrect_agents:
                # Build prompt (optionally with debate context)
                if pref_debate_context:
                    prompt = shared_prompt_pref
                elif include_debate_context:
                    other_agents = [ctx for idx, ctx in enumerate(v['context']) if idx not in (chosen_idx, rejected_idx)]
                    random.shuffle(other_agents)
                    other_agents = other_agents[:5]

                    message = construct_message(
                        agents=other_agents,
                        question=question_prompt,
                        dataset=dataset,
                        summary=None,
                        idx=response_idx,
                        diversity_prompt=diversity_prompt,
                    )
                    prompt = f"{question_prompt}\n\n{message['content']}"
                else:
                    prompt = base_prompt

                data.append({
                    'prompt': prompt,
                    'chosen': chosen_reply,
                    'rejected': rejected_reply,
                    # Auxiliary fields (useful for analysis/debugging)
                    'example_id': example_id,
                    'agent_idx_chosen': chosen_idx,
                    'agent_idx_rejected': rejected_idx,
                })

    return Dataset.from_list(data)

if __name__ == "__main__":
    # for dataset_name in ['gsm8k', 'arithmatic', 'math']:
    #     for i in range(11):
    #         dataset = get_data(dataset_name=dataset_name,
    #                            dataset_path='data', seed=i)
    #         print(dataset_name, i, len([q for q in dataset if q['split'] == 'train']), len(
    #             [q for q in dataset if q['split'] == 'test']))
    
    dataset = get_data(dataset_name = 'amc_aime', dataset_path = 'data', seed = 0, train_size = 40, test_size = 30)
    print('amc_aime', len([q for q in dataset if q['split'] == 'train']), len(
                [q for q in dataset if q['split'] == 'test']))

