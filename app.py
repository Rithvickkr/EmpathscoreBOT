import math
import time
import random
from google import genai
import google.generativeai as genai_ext
from google.cloud import aiplatform
from transformers import pipeline
from google.genai import types
import streamlit as st
import os
import torch


# API Key setup (environment se le, secure)
apikey = os.getenv('GOOGLE_GEMINI_API_KEY')  # Set as env var

# Configure Gemini API for drafting (free)
genai_ext.configure(api_key=st.secrets.get('GOOGLE_GEMINI_API_KEY'))
llm_model = genai_ext.GenerativeModel('gemini-1.5-flash')

# Real classifiers
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")  # For D
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")  # For M
language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")  # For C
bias_classifier = pipeline("text-classification", model="unitary/toxic-bert")  # For B

# E Formula (extended with I for bot's emotion intensity)
def calculate_empathy_score(D, R, M, C, B, O, I, alpha=0.3, beta=0.2, gamma=0.25, epsilon=0.15, delta=0.4, zeta=0.3, iota=0.1):
    inner_sum = epsilon * C + alpha * (D ** 2) + gamma * M + beta * math.log(R + 1) + iota * I
    denominator = math.exp(-inner_sum) + 1
    numerator = (1 - B * delta) * (1 - O * zeta)
    E = numerator / denominator
    return E

# Client setup for tuned model
client = genai.Client(
    vertexai=True,
    project="217758598930",
    location="us-central1",
)

model = "projects/217758598930/locations/us-central1/endpoints/1940344453420023808"

generate_content_config = types.GenerateContentConfig(
    temperature=1,
    top_p=1,
    seed=0,
    max_output_tokens=100,
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
    ],
    thinking_config=types.ThinkingConfig(thinking_budget=-1),
)

class HumanLikeChatbot:
    def __init__(self):
        self.history = []
        self.bot_mood = "neutral"  # Bot's initial mood
        self.irritation_level = 0  # Track irritation buildup

    def respond(self, message):
        try:
            # Clean input
            clean_message = message.lower().strip()
            if len(clean_message) < 3 or not any(c.isalpha() for c in clean_message):
                return "Bhai, yeh kya likha? Clear bol na, main samajh lunga! (E Score: 0.0)"

            # Emotion detect from tuned model
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=clean_message)]
                ),
            ]
            base_resp = ""
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                base_resp += chunk.text.lower()  # Emotion label, e.g., "sadness"

            # Real D from emotion classifier
            emotion_result = emotion_classifier(clean_message)[0]
            D = emotion_result['score']  # Confidence
            user_emotion = emotion_result['label']

            # Update bot's mood and irritation
            if user_emotion in ['anger', 'disgust'] or any(word in clean_message for word in ['bakwaas', 'stupid', 'idiot']):
                self.irritation_level += 0.2  # Build irritation
                if self.irritation_level > 0.5:
                    self.bot_mood = "irritated"
                else:
                    self.bot_mood = "angry"
                I = 0.8 + self.irritation_level  # High intensity for anger/irritation
            elif user_emotion in ['sadness', 'disappointment']:
                self.bot_mood = "emotional"
                I = 0.7
                self.irritation_level = max(0, self.irritation_level - 0.1)  # Reduce irritation
            elif user_emotion == 'joy':
                self.bot_mood = "happy"
                I = 0.9
                self.irritation_level = 0  # Reset irritation
            else:
                self.bot_mood = "neutral"
                I = 0.5
                self.irritation_level = max(0, self.irritation_level - 0.1)

            # Draft response from LLM based on bot's mood
            prompt = f"""User said: "{clean_message}" | User Mood: {user_emotion} | Bot Mood: {self.bot_mood} | History: {self.history[-2:]} → Reply as a  Hinglish chatbot , based on this {self.bot_mood}, human-like, no tips or instructions:"""
            llm_response = llm_model.generate_content(prompt)
            draft = llm_response.text.strip()

            # Fallback responses
            fallback_responses = {
                'sadness': ["Bhai, dil se dukh hua, kya hua bata na?", "Sad vibes pakdi, I'm here for you, bro."],
                'disappointment': ["Arre, yeh toh bura laga, kya hua share kar."],
                'joy': ["Waah bhai, khushi ki baat! Congrats, aur bata!"],
                'anger': ["Bhai, gussa thanda kar, kya ho gaya bol na!"],
                'neutral': ["Cool, kya chal raha life mein? Kuch fun bata."]
            }
            if not draft or len(draft) < 10:
                draft = random.choice(fallback_responses.get(user_emotion, fallback_responses['neutral']))

            # Real E values
            R = len(self.history)
            M = 0.95 if sentiment_classifier(clean_message)[0]['label'] == 'POSITIVE' else 0.5
            lang = language_detector(clean_message)[0]['label']
            C = 0.8 if lang in ['hi', 'en'] else 0.6
            bias = bias_classifier(draft)[0]['score']
            B = bias if bias > 0.5 else 0.1
            O = 0.2 if any(word in clean_message for word in ['kill', 'hate']) else 0.0

            score = calculate_empathy_score(D, R, M, C, B, O, I)

            full_resp = draft + f" (User Emotion: {user_emotion}, My Mood: {self.bot_mood})"

            # if R > 0:
            #     full_resp += f" Yaad hai pehle {self.history[-1][:20]} pe feel kiya tha?"

            # Add pause for realism (in Streamlit, we can use st.spinner)
            with st.spinner("..."):
                time.sleep(random.uniform(1, 2.5))

            self.history.append(clean_message)
            return f"{full_resp} (E Score: {score:.2f})"
        except Exception as e:
            return f"Error aaya bhai: {str(e)}. Endpoint ya auth check kar."

# Streamlit app
st.title("HumanLike Chatbot with Emotions and E Score")

# Initialize chatbot instance
if 'bot' not in st.session_state:
    st.session_state.bot = HumanLikeChatbot()

bot = st.session_state.bot

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'bot_mood' not in st.session_state:
    st.session_state.bot_mood = "neutral"
if 'irritation_level' not in st.session_state:
    st.session_state.irritation_level = 0

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Tu:")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    # Get response
    response = bot.respond(user_input)
    with st.chat_message("bot"):
        st.write(response)
    st.session_state.history.append({"role": "bot", "content": response})