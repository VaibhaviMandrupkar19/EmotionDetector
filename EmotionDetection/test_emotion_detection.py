import unittest
from unittest.mock import patch

from EmotionDetection.emotion_detection import emotion_detector


class TestDominantEmotions(unittest.TestCase):

    def mock_emotion(self, dominant_emotion):
        emotions = {
            "anger": 0.01,
            "disgust": 0.02,
            "fear": 0.03,
            "joy": 0.04,
            "sadness": 0.05
        }

        emotions[dominant_emotion] = 0.90

        mock_response = type(
            "MockResponse",
            (),
            {
                "status_code": 200,
                "text": '{"emotionPredictions":[{"emotion":'
                        + str(emotions).replace("'", '"')
                        + '}]}',
                "raise_for_status": lambda self: None
            }
        )()

        return mock_response

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy(self, mock_post):
        mock_post.return_value = self.mock_emotion("joy")
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger(self, mock_post):
        mock_post.return_value = self.mock_emotion("anger")
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_disgust(self, mock_post):
        mock_post.return_value = self.mock_emotion("disgust")
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result["dominant_emotion"], "disgust")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness(self, mock_post):
        mock_post.return_value = self.mock_emotion("sadness")
        result = emotion_detector("I am so sad about this")
        self.assertEqual(result["dominant_emotion"], "sadness")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_fear(self, mock_post):
        mock_post.return_value = self.mock_emotion("fear")
        result = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()
