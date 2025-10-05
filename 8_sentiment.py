import os
import requests
import json
import pandas as pd
import time


END_DATE = pd.Timestamp.today()
END_DATE = pd.to_datetime(END_DATE).date()
MAX_RETRIES = 5
FILE_PATH = "data/META_sentiment.csv"
# PROMPT = ("Please give me an assessment of Meta shares (ISIN: US30303M1027) based on current news you think are relevant on a scale of 1-100 (1 very poor, 100 very good). Please return only a single int number as your answer.")
PROMPT = ("Please give me an assessment of Meta shares (ISIN: US30303M1027) based on current news within the past 7 days,"
          " using at least 6 reputable financial sources. Use sentiment analysis and any major events impacting the stock."
          " Rate on a scale of 1-100 (1 very poor, 100 very good). Return only a single int as the answer.")
# IMPORTANT: secrets.json needs to be generated before running this script, cf. README!
with open("secrets.json") as f:
    secrets = json.load(f)
gemini_score = 0
perplexity_score = 0
chatgpt_score = 0

# Crawl news score from available Google Gemini models
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
                    {"text": f"{PROMPT}"}
                ]
            }
            ]
        },
        timeout=30
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
perplexity_counter = 0
perplexity_score = None

# Send an API request to the Perplexity API
# Try 5 times until giving up
while perplexity_counter < MAX_RETRIES:
    try:
        print(f"Sending API request to Perplexity API (attempt {perplexity_counter + 1}/{max_retries})...")
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
                        "content": f"{PROMPT}"
                    }
                ]
            },
            timeout=30
        )

        # Check HTTP status
        perplexity_response.raise_for_status()

        # Parse response
        perplexity_score = int(perplexity_response.json()["choices"][0]["message"]["content"])
        print(f"Successfully received score: {perplexity_score}")
        break

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {repr(e)}")
        perplexity_counter += 1

    except (KeyError, ValueError, TypeError) as e:
        print(f"Failed to parse response: {repr(e)}")
        perplexity_counter += 1

    # Exponential backoff (wait longer between retries)
    if perplexity_counter < max_retries:
        wait_time = 2 ** perplexity_counter  # 1s, 2s, 4s, 8s, 16s
        print(f"Waiting {wait_time}s before retry...")
        time.sleep(wait_time)

if perplexity_score is None:
    print(f"Failed to get Perplexity score after {max_retries} attempts")
    # Handle failure case (use default, raise exception, etc.)

# Send an API request to the ChatGPT API
# print("Send API request to the ChatGPT API...")
# chatgpt_model = "gpt-5"
# chatgpt_response = requests.post(
#     "https://api.openai.com/v1/responses",
#     headers={
#         "Authorization": f"Bearer {secrets["OPENAI_API_KEY"]}",
#         "Content-Type": "application/json"
#     },
#     json={
#         "model": chatgpt_model,
#         "input": PROMPT
#     },
#     timeout=30
# )
#
# print("Try to parse API request from the ChatGPT API...")
# try:
#     chatgpt_score = int(chatgpt_response.json()["choices"][0]["message"]["content"])  # NEEDS TO BE MODIFIED; NOT TESTED!
# except Exception as e:
#     print("ChatGPT score currently not available.")
#     print("Error was:", repr(e))

new_row = {
    "DATE": f"{END_DATE}",
    "PERPLEXITY_SCORE": perplexity_score, "PERPLEXITY_MODEL": perplexity_model,
    "GEMINI_SCORE": gemini_score, "GEMINI_MODEL": gemini_model,
    # "CHATGPT_SCORE": chatgpt_score, "CHATGPT_MODEL": chatgpt_model
}
df = pd.DataFrame([new_row])
print(f"Save data to {FILE_PATH}")
if os.path.isfile(FILE_PATH):
    df.to_csv(FILE_PATH, mode="a", header=False, index=False)
else:
    df.to_csv(FILE_PATH, mode="w", header=True, index=False)
