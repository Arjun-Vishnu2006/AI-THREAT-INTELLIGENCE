import joblib

path = "artifacts/real_url_model/url_phishing_model.joblib"
model = joblib.load(path)

urls = [
    "https://www.google.com",
    "https://www.wikipedia.org",
    "http://dbs.vote-friend.com/sg?ref=anything",
    "https://www.halisupportservice.com/Login.php",
]

for url in urls:
    prediction = int(model.predict([url])[0])
    probability = model.predict_proba([url])[0][1]
    label = "PHISHING" if prediction == 1 else "LEGITIMATE"
    print(f"\nURL: {url}")
    print(f"Prediction: {label}")
    print(f"Phishing probability: {probability:.2%}")
