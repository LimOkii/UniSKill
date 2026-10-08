from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from uniskill.training.operation_support import operation_support_loss


@unittest.skipIf(torch is None, "torch is not installed")
class OperationSupportLossTest(unittest.TestCase):
    def test_zero_when_every_legal_action_is_above_floor(self):
        logits = torch.zeros(2, 3, requires_grad=True)
        valid = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
        loss, probabilities, violations = operation_support_loss(
            operation_logits=logits,
            valid_actions=valid,
            min_probability=0.1,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(violations, torch.zeros_like(violations)))
        self.assertAlmostEqual(probabilities[0].sum().item(), 1.0)
        self.assertEqual(probabilities[1, 2].item(), 0.0)

    def test_gradient_raises_a_low_probability_legal_action(self):
        logits = torch.tensor([[4.0, 0.0, -4.0]], requires_grad=True)
        valid = torch.ones_like(logits, dtype=torch.bool)
        loss, probabilities, violations = operation_support_loss(
            operation_logits=logits,
            valid_actions=valid,
            min_probability=0.1,
        )
        self.assertGreater(loss.item(), 0.0)
        self.assertLess(probabilities[0, 2].item(), 0.1)
        self.assertGreater(violations[0, 2].item(), 0.0)
        loss.backward()
        self.assertLess(logits.grad[0, 2].item(), 0.0)
        self.assertGreater(logits.grad[0, 0].item(), 0.0)

    def test_illegal_update_is_excluded_from_two_action_context(self):
        logits = torch.tensor([[0.0, 0.0, 20.0]], requires_grad=True)
        valid = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        loss, probabilities, _ = operation_support_loss(
            operation_logits=logits,
            valid_actions=valid,
            min_probability=0.1,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(probabilities.tolist(), [[0.5, 0.5, 0.0]])


if __name__ == "__main__":
    unittest.main()
