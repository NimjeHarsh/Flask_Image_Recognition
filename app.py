
"""THE Flask application for hand-sign digit recognition."""

from flask import Flask, render_template, request
from model import preprocess_img, predict_result

app = Flask(__name__)


@app.route("/")
def main():
    """Display the home page."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Process an uploaded image and display its predicted digit."""
    if "file" not in request.files:
        return render_template(
            "result.html", err="File cannot be processed."
        )

    try:
        image = preprocess_img(request.files["file"].stream)
        prediction = predict_result(image)
        return render_template(
            "result.html", predictions=str(prediction)
        )
    except (ValueError, OSError, TypeError) as error:
        app.logger.warning("Image processing failed: %s", error)
        return render_template(
            "result.html", err="File cannot be processed."
        )


if __name__ == "__main__":
    app.run(port=9000, debug=True)
