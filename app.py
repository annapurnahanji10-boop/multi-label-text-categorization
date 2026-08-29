from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


# ==========================================
# LOAD TRAINED MODELS
# ==========================================

model = joblib.load("models/text_model.pkl")

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

mlb = joblib.load(
    "models/label_binarizer.pkl"
)


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        predictions=[],
        text=""
    )


# ==========================================
# PREDICT
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    text = request.form.get(
        "text",
        ""
    ).strip()

    predictions = []

    if text:

        # Clean text
        clean_text = text.lower()

        clean_text = " ".join(
            clean_text.split()
        )

        # TF-IDF
        text_vector = vectorizer.transform(
            [clean_text]
        )

        # Prediction probabilities
        probabilities = model.predict_proba(
            text_vector
        )[0]

        labels = mlb.classes_

        # Create prediction results
        for label, probability in zip(
            labels,
            probabilities
        ):

            if probability >= 0.50:

                predictions.append({
                    "label": label,
                    "confidence": round(
                        probability * 100,
                        2
                    )
                })

        # Sort highest confidence first
        predictions.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

    return render_template(
        "index.html",
        predictions=predictions,
        text=text
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )