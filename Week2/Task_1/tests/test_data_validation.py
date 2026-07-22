"""
Unit Tests for Data Validation
"""

import unittest
import pandas as pd

from src.data.validate_data import DataValidator


class TestDataValidator(unittest.TestCase):

    def setUp(self):

        self.df = pd.DataFrame({

            "A": [1, 2, 3],

            "B": [4, 5, 6]

        })

    def test_dataframe_not_empty(self):

        self.assertFalse(self.df.empty)

    def test_shape(self):

        self.assertEqual(
            self.df.shape,
            (3, 2)
        )

    def test_validator_creation(self):

        validator = DataValidator(self.df)

        self.assertIsNotNone(validator)


if __name__ == "__main__":

    unittest.main()
    