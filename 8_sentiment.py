import os
import requests
import json
import pandas as pd


end_date = pd.Timestamp.today()
end_date = pd.to_datetime(end_date).date()
file_path = "data/META_sentiment.csv"
# prompt = ("Please give me an assessment of Meta shares (ISIN: US30303M1027) based on current news you think are relevant on a scale of 1-100 (1 very poor, 100 very good). Please return only a single int number as your answer.")
prompt = ("Please give me an assessment of Meta shares (ISIN: US30303M1027) based on current news within the past 7 days,"
          " using at least 6 reputable financial sources. Use sentiment analysis and any major events impacting the stock."
          " Rate on a scale of 1-100 (1 very poor, 100 very good). Return only a single int as the answer.")
# IMPORTANT: secrets.json needs to be generated before running this script, cf. README!
with open("secrets.json") as f:
    secrets = json.load(f)
gemini_score = 0
perplexity_score = 0

# Crawl news score from Google Gemini, available models:
# gemini-2.5-pro
# gemini-2.5-flash
# gemini-2.5-flash-lite
gemini_models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

# Try different Gemini models as better models are mostly not available
# Select best model first, then break loop according to availability
for gemini_model in gemini_models:
    # Send an API request to the Gemini API
    print(f"Send API request to the Gemini API using model {gemini_model}...")
    gemini_response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": secrets["GEMINI_API_KEY"]
        },
        json={
            "contents": [
            {
                "parts": [
                    {"text": f"{prompt}"}
                ]
            }
            ]
        }
    )

    print("Try to parse API request from the Gemini API...")
    if gemini_response.status_code == 200:
        try:
            gemini_score = gemini_response.json()["candidates"][0]["content"]["parts"][0]["text"].split()[-1]
            break
        except Exception as e:
            print(f"Gemini model {gemini_model} currently not available.")
            print("Error was:", repr(e))
    else:
        print(f"Gemini model {gemini_model} currently not available.")
        print(f"Error was: {gemini_response.text}")

# Crawl news score from Perplexity, available models:
# sonar - Fast, straightforward answers
# sonar-pro - Complex research with more sources
# sonar-reasoning - Enhanced reasoning capabilities
# sonar-pro-reasoning - Advanced reasoning with deeper research
# sonar-deep-research - Comprehensive research-intensive tasks
perplexity_model = "sonar-pro"

# Send an API request to the Perplexity API
print("Send API request to the Perplexity API...")
perplexity_response = requests.post(
    "https://api.perplexity.ai/chat/completions",
    headers={
        "Authorization": f"Bearer {secrets["PERPLEXITY_API_KEY"]}",
        "Content-Type": "application/json"
    },
    json={
        "model": f"{perplexity_model}",
        "messages": [
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ]
    }
)

print("Try to parse API request from the Perplexity API...")
try:
    perplexity_score = int(perplexity_response.json()["choices"][0]["message"]["content"])
except Exception as e:
    print("Perplexity score currently not available.")
    print("Error was:", repr(e))

new_row = {"DATE": f"{end_date}", "PERPLEXITY_SCORE": perplexity_score, "PERPLEXITY_MODEL": perplexity_model, "GEMINI_SCORE": gemini_score, "GEMINI_MODEL": gemini_model}
df = pd.DataFrame([new_row])
print(f"Save data to {file_path}")
if os.path.isfile(file_path):
    df.to_csv(file_path, mode="a", header=False, index=False)
else:
    df.to_csv(file_path, mode="w", header=True, index=False)
