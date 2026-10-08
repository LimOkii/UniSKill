from __future__ import annotations

import unittest

from uniskill.environments.webshop_actions import canonicalize_webshop_action_response


class WebshopStrictActionTest(unittest.TestCase):
    def test_two_action_blocks_are_not_overall_valid(self):
        canonical = canonicalize_webshop_action_response(
            raw_response=(
                "<think>Go back and search again.</think>"
                "<action>click[back to search]</action>"
                "<action>search[machine washable pocket tee]</action>"
            ),
            projected_action="click[back to search]",
            format_valid=True,
            admissible_actions=["click[back to search]"],
        )

        self.assertTrue(canonical.payload_parsed)
        self.assertTrue(canonical.admissible)
        self.assertFalse(canonical.strict_format_valid)
        self.assertFalse(canonical.overall_valid)


if __name__ == "__main__":
    unittest.main()
