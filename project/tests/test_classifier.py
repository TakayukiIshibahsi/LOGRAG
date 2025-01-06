import unittest
from models.classifer import Classifier

class TestClassifier(unittest.TestCase):
    def setUp(self):
        """
        テストのセットアップ処理
        """
        self.classifier = Classifier()  # 必要に応じて変更
        self.sample_input = [
            "As Michael Kaleko kept running into people who were getting older and having more vision problems, he realized he could do something about it.",
            "TheDeal.com - The U.K. mobile giant wants to find a way to disentagle the Czech wireless and fixed-line businesses",
            "Jay Haas joined Stewart Cink as the two captain's picks for a U.S. team that will try to regain the cup from Europe next month."
        ]
        self.expected_output = [  # ダミーの期待されるラベル
            "4",
            "4",
            "2"
        ]

    def test_classification(self):
        """
        正常な分類を確認
        """
        predictions = self.classifier.predict(self.sample_input)
        self.assertEqual(len(predictions), len(self.expected_output), "出力数が期待と一致しません")
        
        for pred, expected in zip(predictions, self.expected_output):
            self.assertEqual(pred, expected, f"予測結果が期待と一致しません: {pred} != {expected}")

    def test_empty_input(self):
        """
        空の入力に対する動作を確認
        """
        predictions = self.classifier.predict([])
        self.assertEqual(predictions, [], "空の入力に対して出力が空ではありません")

    def test_invalid_input(self):
        """
        不正な入力に対する例外処理を確認
        """
        with self.assertRaises(ValueError, msg="不正な入力に対して例外が発生しません"):
            self.classifier.predict(None)

    def test_large_input(self):
        """
        大規模な入力に対する動作を確認
        """
        large_input = ["Sample sentence"] * 1000  # 1000件の入力
        predictions = self.classifier.predict(large_input)
        self.assertEqual(len(predictions), len(large_input), "大規模入力の出力数が一致しません")

if __name__ == "__main__":
    unittest.main()
