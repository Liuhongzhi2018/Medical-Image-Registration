# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest
from unittest import skipUnless

import numpy as np
import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.selfattention import SABlock
from monai.networks.layers.factories import RelPosEmbedding
from monai.utils import optional_import

einops, has_einops = optional_import("einops")

TEST_CASE_SABLOCK = []
for dropout_rate in np.linspace(0, 1, 4):
    for hidden_size in [360, 480, 600, 768]:
        for num_heads in [4, 6, 8, 12]:
            for rel_pos_embedding in [None, RelPosEmbedding.DECOMPOSED]:
                for input_size in [(16, 32), (8, 8, 8)]:
                    test_case = [
                        {
                            "hidden_size": hidden_size,
                            "num_heads": num_heads,
                            "dropout_rate": dropout_rate,
                            "rel_pos_embedding": rel_pos_embedding,
                            "input_size": input_size,
                        },
                        (2, 512, hidden_size),
                        (2, 512, hidden_size),
                    ]
                    TEST_CASE_SABLOCK.append(test_case)


class TestResBlock(unittest.TestCase):

    @parameterized.expand(TEST_CASE_SABLOCK)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, input_param, input_shape, expected_shape):
        net = SABlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            SABlock(hidden_size=128, num_heads=12, dropout_rate=6.0)

        with self.assertRaises(ValueError):
            SABlock(hidden_size=620, num_heads=8, dropout_rate=0.4)

    def test_attention_dim_not_multiple_of_heads(self):
        with self.assertRaises(ValueError):
            SABlock(hidden_size=128, num_heads=3, dropout_rate=0.1)

    @skipUnless(has_einops, "Requires einops")
    def test_inner_dim_different(self):
        SABlock(hidden_size=128, num_heads=4, dropout_rate=0.1, dim_head=30)

    def test_causal_no_sequence_length(self):
        with self.assertRaises(ValueError):
            SABlock(hidden_size=128, num_heads=4, dropout_rate=0.1, causal=True)

    @skipUnless(has_einops, "Requires einops")
    def test_causal(self):
        block = SABlock(hidden_size=128, num_heads=1, dropout_rate=0.1, causal=True, sequence_length=16, save_attn=True)
        input_shape = (1, 16, 128)
        block(torch.randn(input_shape))
        # check upper triangular part of the attention matrix is zero
        assert torch.triu(block.att_mat, diagonal=1).sum() == 0

    @skipUnless(has_einops, "Requires einops")
    def test_access_attn_matrix(self):
        # input format
        hidden_size = 128
        num_heads = 2
        dropout_rate = 0
        input_shape = (2, 256, hidden_size)

        # be  not able to access the matrix
        no_matrix_acess_blk = SABlock(hidden_size=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate)
        no_matrix_acess_blk(torch.randn(input_shape))
        assert isinstance(no_matrix_acess_blk.att_mat, torch.Tensor)
        # no of elements is zero
        assert no_matrix_acess_blk.att_mat.nelement() == 0

        # be able to acess the attention matrix
        matrix_acess_blk = SABlock(
            hidden_size=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate, save_attn=True
        )
        matrix_acess_blk(torch.randn(input_shape))
        assert matrix_acess_blk.att_mat.shape == (input_shape[0], input_shape[0], input_shape[1], input_shape[1])

    def test_number_of_parameters(self):

        def count_sablock_params(*args, **kwargs):
            """Count the number of parameters in a SABlock."""
            sablock = SABlock(*args, **kwargs)
            return sum([x.numel() for x in sablock.parameters() if x.requires_grad])

        hidden_size = 128
        num_heads = 8
        default_dim_head = hidden_size // num_heads

        # Default dim_head is hidden_size // num_heads
        nparams_default = count_sablock_params(hidden_size=hidden_size, num_heads=num_heads)
        nparams_like_default = count_sablock_params(
            hidden_size=hidden_size, num_heads=num_heads, dim_head=default_dim_head
        )
        self.assertEqual(nparams_default, nparams_like_default)

        # Increasing dim_head should increase the number of parameters
        nparams_custom_large = count_sablock_params(
            hidden_size=hidden_size, num_heads=num_heads, dim_head=default_dim_head * 2
        )
        self.assertGreater(nparams_custom_large, nparams_default)

        # Decreasing dim_head should decrease the number of parameters
        nparams_custom_small = count_sablock_params(
            hidden_size=hidden_size, num_heads=num_heads, dim_head=default_dim_head // 2
        )
        self.assertGreater(nparams_default, nparams_custom_small)

        # Increasing the number of heads with the default behaviour should not change the number of params.
        nparams_default_more_heads = count_sablock_params(hidden_size=hidden_size, num_heads=num_heads * 2)
        self.assertEqual(nparams_default, nparams_default_more_heads)


if __name__ == "__main__":
    unittest.main()
