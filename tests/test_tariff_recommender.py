import unittest

from src.tariff_recommender import recommend_tariffs


class RecommendTariffsTests(unittest.TestCase):
    def test_filters_by_profile_and_consumption_and_returns_at_most_three(self) -> None:
        tariffs = recommend_tariffs("Privatkunde", 3200, "Bremen")

        self.assertEqual(
            [tariff["name"] for tariff in tariffs],
            ["swb Strom basis", "swb Strom von hier", "swb Strom spar L"],
        )
        self.assertTrue(all(tariff["region"] == "Bremen" for tariff in tariffs))
        self.assertTrue(
            all("Privatperson" in tariff["customer_types"] for tariff in tariffs)
        )

    def test_business_results_are_limited_to_three(self) -> None:
        tariffs = recommend_tariffs("Geschäftskunde", 50_000, "Bremen")

        self.assertEqual(
            [tariff["name"] for tariff in tariffs],
            ["swb Strom Business", "swb Strom dynamisch", "swb Strom Ersatzversorgung"],
        )

    def test_unknown_consumption_still_filters_by_type_and_region(self) -> None:
        tariffs = recommend_tariffs("Privatkunde", "Keine Angabe", "Weyhe")

        self.assertEqual(len(tariffs), 3)
        self.assertTrue(all(tariff["region"] == "Weyhe" for tariff in tariffs))
        self.assertTrue(
            all("Privatperson" in tariff["customer_types"] for tariff in tariffs)
        )

    def test_missing_customer_type_or_region_returns_no_recommendations(self) -> None:
        self.assertEqual(recommend_tariffs(None, 3200, "Bremen"), [])
        self.assertEqual(recommend_tariffs("Privatkunde", 3200, None), [])

    def test_nonpositive_consumption_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recommend_tariffs("Privatkunde", 0, "Bremen")


if __name__ == "__main__":
    unittest.main()
