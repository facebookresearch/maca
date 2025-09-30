"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""Main entry point for multi-agent debate training."""

import torch
from args import parse_args
import torch.multiprocessing as mp
from orchestrator import MultiAgentOrchestrator

if __name__ == '__main__':
    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
    config = vars(parse_args())       
    orchestrator = MultiAgentOrchestrator(config)
    orchestrator.run_training()