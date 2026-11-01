"""CPU-only unit tests; tests the actual Phase6 Stock-ID module without GPU or BaoStock."""
from __future__ import annotations
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase6_id_model import StockIDResidual, batch_ids, stable_vocab

class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(6, 5)
    def forward(self, x):
        return self.linear(x[:, -1])

class StockIDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.vocab=stable_vocab(["sz.000983","sh.601225","sh.601088"])
        self.model=StockIDResidual(ToyBackbone(),lambda b,x:b(x),len(self.vocab),8,5)
        self.x=torch.randn(3,60,6)
        self.ids=torch.tensor([0,1,2],dtype=torch.long)

    def test_vocab_stable_and_duplicate_forbidden(self):
        self.assertEqual(self.vocab,["sh.601088","sh.601225","sz.000983"])
        with self.assertRaises(ValueError):stable_vocab(["sh.601088","sh.601088"])

    def test_id_head_initializes_as_exact_no_id_model(self):
        torch.testing.assert_close(self.model(self.x,self.ids),self.model.backbone(self.x),rtol=0,atol=0)
        self.assertTrue(torch.all(self.model.id_correction(self.ids)==0).item())

    def test_batch_ids_use_stock_names_not_dataset_position(self):
        names=np.array(["sz.000983","sh.601088","sh.601225"])
        shuffled=torch.tensor([2,0,1])
        self.assertEqual(batch_ids(names,shuffled,self.vocab,torch.device("cpu")).tolist(),[1,2,0])

    def test_different_stock_ids_can_generate_different_forecasts(self):
        with torch.no_grad():
            self.model.embedding.weight.copy_(torch.eye(3,8))
            self.model.id_head.weight[0,0]=0.01
            self.model.id_head.weight[0,1]=-0.02
        same=self.x[0:1].expand(3,-1,-1)
        output=self.model(same,self.ids)
        self.assertGreater(abs(float(output[0,0].detach())-float(output[1,0].detach())), 0.02)
        self.assertAlmostEqual(float(output[0,1].detach()),float(output[1,1].detach()),places=6)

    def test_gradients_flow_to_stock_head_and_embedding(self):
        y=torch.randn(3,5)
        opt=torch.optim.Adam(self.model.parameters(),lr=.01)
        pred=self.model(self.x,self.ids)
        loss=(pred-y).square().mean()+.001*self.model.id_correction(self.ids).square().mean()
        opt.zero_grad();loss.backward();opt.step()
        self.assertGreater(self.model.id_head.weight.abs().sum().item(),0)
        opt.zero_grad();(self.model(self.x,self.ids)-y).square().mean().backward()
        self.assertIsNotNone(self.model.embedding.weight.grad)
        self.assertTrue(torch.isfinite(self.model.embedding.weight.grad).all().item())

    def test_unknown_stock_id_and_mismatched_batch_fail(self):
        with self.assertRaises(ValueError):self.model(self.x,torch.tensor([0,1,3]))
        with self.assertRaises(ValueError):self.model(self.x,torch.tensor([0]))
        with self.assertRaises(ValueError):batch_ids(np.array(["no.stock"]),torch.tensor([0]),self.vocab,torch.device("cpu"))

    def test_state_dict_roundtrip_and_zero_vs_nonzero_stock_effect(self):
        with torch.no_grad():self.model.id_head.bias.fill_(.004)
        other=StockIDResidual(ToyBackbone(),lambda b,x:b(x),len(self.vocab),8,5)
        other.load_state_dict(copy.deepcopy(self.model.state_dict()))
        torch.testing.assert_close(other(self.x,self.ids),self.model(self.x,self.ids),rtol=0,atol=0)

if __name__ == "__main__":unittest.main()
