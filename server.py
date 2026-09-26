"""Flask web application for emotion detection."""
from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector


app = Flask(__name__)


@app.route("/")
def index():
    """Display the emotion detector interface."""
    return render_template("index.html")


@app.route("/emotionDetector", methods=["GET"])
def emotion_detector_route():
    """Analyze the text submitted by the user."""

    text_to_analyze = request.args.get("textToAnalyze")

    if text_to_analyze is None or text_to_analyze.strip() == "":
        return (
            "Please enter some text to analyze.",
            400
        )

    response = emotion_detector(text_to_analyze)

    if response["dominant_emotion"] is None:
        return (
            "Unable to analyze the text. "
            "The Watson NLP service is currently unavailable.",
            503
        )

    return (
        "For the given statement, the system response is: "
        f"{response}",
        200
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
