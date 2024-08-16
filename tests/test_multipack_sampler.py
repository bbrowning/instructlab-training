# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the top-level multipack_sampler module.
"""

# Standard
from unittest.mock import patch
import os
import unittest

# First Party
from instructlab.training.multipack_sampler import (
    find_packing_max_batch_len_and_grad_accum,
)
from instructlab.training.token_dataset import setup_dataset

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


class TestPackingMaxBatchLen(unittest.TestCase):
    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"})
    def test_max_batch_len_not_longer_than_all_data(self):
        # small_process_data.jsonl has 403 total tokens and
        # avg_sample_len of 57.57
        num_gpus = 1
        effective_batch_size = 1000  # something > 403
        max_batch_len_per_gpu = 10000
        is_padding = True
        dataset = setup_dataset(
            os.path.join(TEST_DATA_DIR, "small_processed_data.jsonl")
        )
        avg_sample_len = dataset.get_lengths().mean()
        pad_id = 32001  # a random realistic looking token id
        seed = 42  # a random seed

        packing_max_batch_len, grad_accum = find_packing_max_batch_len_and_grad_accum(
            num_gpus,
            avg_sample_len,
            effective_batch_size,
            max_batch_len_per_gpu,
            is_padding,
            dataset,
            pad_id,
            seed,
        )
        assert packing_max_batch_len <= dataset.get_lengths().sum()
        assert grad_accum == 1

    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"})
    def test_falls_back_to_distributed_when_batch_too_small(self):
        num_gpus = 1
        effective_batch_size = 10
        max_batch_len_per_gpu = 50
        is_padding = True
        dataset = setup_dataset(
            os.path.join(TEST_DATA_DIR, "small_processed_data.jsonl")
        )
        avg_sample_len = dataset.get_lengths().mean()
        pad_id = 32001  # a random realistic looking token id
        seed = 42  # a random seed

        with self.assertRaises(RuntimeError):
            find_packing_max_batch_len_and_grad_accum(
                num_gpus,
                avg_sample_len,
                effective_batch_size,
                max_batch_len_per_gpu,
                is_padding,
                dataset,
                pad_id,
                seed,
            )
