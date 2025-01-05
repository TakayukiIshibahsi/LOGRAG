import unittest
from models.classifier import Classifier

class TestClassifier(unittest.TestCase):
    def setUp(self):
        """
        テストのセットアップ処理
        """
        self.classifier = Classifier()  # 必要に応じて変更
        self.sample_input = [
            "This is a news article about technology.",
            "Today's weather is sunny and warm.",
            "The stock market saw a significant increase."
        ]
        self.expected_output = [  # ダミーの期待されるラベル
            "technology",
            "weather",
            "finance"
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
