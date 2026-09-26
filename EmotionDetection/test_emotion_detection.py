from unittest.mock import patch

from EmotionDetection.emotion_detection import emotion_detector


def test_blank_input():
    result = emotion_detector("")
    assert result["anger"] is None
    assert result["disgust"] is None
    assert result["fear"] is None
    assert result["joy"] is None
    assert result["sadness"] is None
    assert result["dominant_emotion"] is None


def test_none_input():
    result = emotion_detector(None)
    assert result["dominant_emotion"] is None


@patch("EmotionDetection.emotion_detection.requests.post")
def test_emotion_output(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.text = (
        '{"emotionPredictions":[{"emotion":{'
        '"anger":0.01,'
        '"disgust":0.02,'
        '"fear":0.03,'
        '"joy":0.90,'
        '"sadness":0.04'
        '}}]}'
    )

    result = emotion_detector("I love this new technology.")

    assert result["anger"] == 0.01
    assert result["disgust"] == 0.02
    assert result["fear"] == 0.03
    assert result["joy"] == 0.90
    assert result["sadness"] == 0.04
    assert result["dominant_emotion"] == "joy"
