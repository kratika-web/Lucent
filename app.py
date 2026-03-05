!pip install transformers groq matplotlib numpy pandas gradio torch -q

from transformers import pipeline
from groq import Groq
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import gradio as gr
import json
import os
import re
import time
from datetime import datetime
from transformers import pipeline

#--------importing model-----------
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=-1
)

#-------------define compute_distress()----------------
emotion_weights = {
    "joy": 1.0,
    "surprise": 2.0,
    "neutral": 2.5,
    "disgust": 3.5,
    "anger": 4.0,
    "fear": 4.5,
    "sadness": 5.0
}

def compute_distress(text):
    results = emotion_model(text)[0]  # get inner list

    weighted_score = 0
    dominant_emotion = None
    max_prob = 0

    for r in results:
        label = r["label"]
        prob = r["score"]

        weighted_score += emotion_weights[label] * prob

        if prob > max_prob:
            max_prob = prob
            dominant_emotion = label

    final_score = round(weighted_score, 2)

    return {
        "score": final_score,
        "dominant_emotion": dominant_emotion
    }
  
#--------------------define intake questions with dimensions----------------
INTAKE_QUESTIONS = [
    {
        "dimension": "Sleep_quality",
        "text": "Let's start with something simple — how's your sleep been lately? Are you waking up feeling rested, or more like you could sleep for a week? 😄",
    },
    {
        "dimension": "Academic_stress",
        "text": "Got it! Now tell me — how are things going with your studies or work? Are you feeling on top of things, or does it sometimes feel like there's a mountain of tasks staring you down? 📚",
    },
    {
        "dimension": "Social_anxiety",
        "text": "Nice, thanks for sharing! Quick one — when you're around people like friends, classmates, or in group situations, how do you usually feel? Energized and comfortable, or more drained and nervous? 🤝",
    },
    {
        "dimension": "Emotional_regulation",
        "text": "You're doing great! When something stressful or upsetting happens, what do you usually do? Do you handle it calmly, or does it sometimes feel like your emotions take over? 🌊",
    },
    {
        "dimension": "Self_view",
        "text": "Last one, I promise! 🌟 How do you feel about yourself day to day? Do you generally feel good about who you are, or do self-doubts creep in more than you'd like?",
    }
]

#---------------------define intake class----------------------
class Intake:
    def __init__(self):
        self.index = 0
        self.answers = []
        self.scores = {}
        self.completed = False

    def get_current_question(self):
        if self.index < len(INTAKE_QUESTIONS):
            return INTAKE_QUESTIONS[self.index]["text"]
        return None

    def submit_answer(self, text):
        if self.completed:
            return None

        # 1. Compute distress
        result = compute_distress(text)

        # 2. Identify dimension
        dimension = INTAKE_QUESTIONS[self.index]["dimension"]

        # 3. Store full answer data
        self.answers.append({
            "dimension": dimension,
            "text": text,
            "score": result["score"],
            "emotion": result["dominant_emotion"]
        })

        # 4. Store only score separately
        self.scores[dimension] = result["score"]

        # 5. Move to next question
        self.index += 1

        # 6. Check if finished
        if self.index >= len(INTAKE_QUESTIONS):
            self.completed = True
            return "DONE"

        return self.get_current_question()
      
#-----------------radar chart generation---------------
import matplotlib.pyplot as plt
import numpy as np

def generate_radar(scores, username="User"):
    labels = list(scores.keys())
    values = list(scores.values())

    # Close the polygon
    values += values[:1]

    # Create angle positions
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    # Create plot
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0,5)

    plt.title(f"{username}'s Mental Health Profile\n")
    
    import io
    from PIL import Image

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)

    return img

#---------------------save_session()--------------------
import json
import os
from datetime import datetime

def save_session(username, scores, answers):

    filename = f"{username.lower()}_data.json"

    new_session = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "scores": scores,
        "answers": answers
    }

    data = {"username": username, "sessions": []}

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            # ensure sessions key exists
            if "sessions" not in data:
                data["sessions"] = []

        except:
            data = {"username": username, "sessions": []}

    data["sessions"].append(new_session)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

  #--------------------load_session()---------------
  def load_sessions(username):
    filename = f"{username.lower()}_data.json"

    if not os.path.exists(filename):
        return None

    with open(filename, "r") as f:
        data = json.load(f)

    return data.get("sessions", [])

#----------------get_last_session_summary----------
def get_last_session_summary(username):
    sessions = load_sessions(username)

    if not sessions:
        return None

    last_session = sessions[-1]

    scores = last_session["scores"]

    # Find highest distress dimension
    highest_dimension = max(scores, key=scores.get)
    highest_score = scores[highest_dimension]

    # Find dominant emotion across intake answers
    emotions = [a["emotion"] for a in last_session["answers"]]
    dominant_emotion = max(set(emotions), key=emotions.count)

    return {
        "session_count": len(sessions),
        "dominant_emotion": dominant_emotion,
        "highest_dimension": highest_dimension,
        "highest_score": highest_score
    }
#------------------collect_user_history--------------
def collect_user_history(username):

    sessions = load_sessions(username)

    if not sessions:
        return None

    sessions = sessions[-5:]

    all_scores = []
    all_answers = []
    all_conversations = []

    for s in sessions:
        all_scores.append(s["scores"])

        for a in s["answers"]:
            all_answers.append(a["text"])

        if "conversations" in s:
            for c in s["conversations"]:
                all_conversations.append(c["text"])

    return {
        "session_count": len(sessions),
        "scores": all_scores,
        "answers": all_answers,
        "conversations": all_conversations
    }
def compute_distress_trend(scores):

    avg_scores = []

    for session_scores in scores:
        avg = sum(session_scores.values()) / len(session_scores)
        avg_scores.append(avg)

    if len(avg_scores) < 2:
        return "not enough data"

    first = avg_scores[0]
    last = avg_scores[-1]

    if last < first - 0.3:
        return "improving"
    elif last > first + 0.3:
        return "worsening"
    else:
        return "stable"

  #----------------weekly_insight_generator--------
  def generate_weekly_insight(username):

    history = collect_user_history(username)

    if not history:
        return "Not enough data yet to generate insight."

    answers_text = " ".join(history["answers"][-20:])
    conversation_text = " ".join(history["conversations"][-20:]) if history["conversations"] else "No conversations yet."

    trend = compute_distress_trend(history["scores"])

    prompt = f"""
A student has been talking with a mental health companion.

Number of sessions: {history['session_count']}
Overall distress trend: {trend}

Things they shared in intake:
{answers_text}

Things they talked about in conversations:
{conversation_text}

Write exactly 3 sentences; not more than 30 words per sentence:

1. Notice and define a pattern in their struggles.
2. Recognize something they handled well and acknowledge it.
3. Suggest one small concrete action/fun activity/help for next week.

Speak warmly, politely and directly to them.
Do not diagnose.
"""

    system = "You are Lucent, a thoughtful mental health companion."

    return call_llm(system, prompt)

#-----------------import groq--------------
from groq import Groq

client = Groq(api_key="YOUR_GROQ_API_KEY")

def call_llm(system_prompt, user_message):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

def determine_mode(scores):
    avg = sum(scores.values()) / len(scores)

    if avg >= 4.0:
        return "high"
    elif avg >= 3.0:
        return "moderate"
    else:
        return "stable"

#---------------------Companion class-----------------
class Companion:
def __init__(self, username, scores):
      self.username = username
      self.scores = scores
      self.mode = determine_mode(scores)
      self.history = []
      self.memory = get_last_session_summary(username)

def build_system_prompt(self):
      base = (
          "You are Lucent, a warm mental health companion. "
          "Never diagnose. Never give medical advice. "
          "Respond in exactly 2 sentences. "
      )

      if self.memory:
        base += (
            "The user has spoken with you before. "
            f"They previously struggled most with {self.memory.get('highest_dimension')}. "
            "Acknowledge continuity naturally, without sounding analytical."
            )
      if self.mode == "high":
          base += (
              "Validate first. Then give ONE simple grounding suggestion. "
              "Keep tone calm and steady."
          )
      elif self.mode == "moderate":
          base += (
              "Validate briefly. Then ask EXACTLY ONE short reflective question. "
              "Do not combine multiple questions."
          )
      else:
          base += (
              "Be supportive and light. Encourage growth with ONE thoughtful question."
          )

      return base

  def respond(self, message):
      real_time = compute_distress(message)["score"]

      if real_time >= 4.0:
          self.mode = "high"
      elif real_time >= 3.0:
          self.mode = "moderate"
      else:
          self.mode = "stable"

      system_prompt = self.build_system_prompt()

      response = call_llm(system_prompt, message)

      self.history.append({
          "user": message,
          "bot": response,
          "distress": real_time,
          "mode": self.mode
      })

      return response

#---------------Gradio UI-----------------
import gradio as gr

state = {
    "phase":"name",
    "username":None,
    "intake":None,
    "bot":None
}

# ---------------- CHAT ----------------
def show_chat():
    return gr.update(visible=True), gr.update(visible=False)

def show_profile():
    return gr.update(visible=False), gr.update(visible=True)

def chat(msg,history):

    if not msg.strip():
        return "",history,None

    try:

        # ask name
        if state["phase"]=="name":

            username = msg.strip().title()
            state["username"] = username

            sessions = load_sessions(username) or []

            history.append((msg,None))

            if sessions:

                last_session = sessions[-1]

                state["phase"]="chat"
                state["bot"] = Companion(username, last_session["scores"])

                summary = get_last_session_summary(username)

                history[-1] = (
                    msg,
                    f"Welcome back {username} ✨\n\n"
                    f"Last time you shared that {summary['highest_dimension']} was weighing on you."
                    )

                return "",history,None


            # new user
            else:

                state["phase"]="intake"
                state["intake"]=Intake()

                q = state["intake"].get_current_question()

                history[-1]=(msg,
                f"Nice to meet you {username} ✨\n\nTo understand you a little better, I’ll ask a few gentle questions. There are no right or wrong answers.\n\n{q}")
                return "",history,None


        # intake phase
        elif state["phase"] == "intake":

          intake = state["intake"]

          result = intake.submit_answer(msg)

          history.append((msg, None))

          # Intake finished
          if result == "DONE":

            chart = generate_radar(intake.scores, state["username"])

            save_session(state["username"], intake.scores, intake.answers)

            state["phase"] = "chat"
            state["bot"] = Companion(state["username"], intake.scores)

            history[-1] = (
                msg,
                "Your profile is ready ✨\n\n"
                "Check the Profile tab.\n\n"
                "What would you like to talk about today?"
                )
            return "", history, chart

          # Intake still continuing → ask next question
          history[-1] = (msg, result)

          return "", history, None


        # chat phase
        else:

          if msg.lower() == "done":

            history.append((
                msg,
                "I'm glad you shared today. Take care and come back anytime ✨"
                ))

            return "", history, None

          bot = state["bot"]

          response = bot.respond(msg)

          history.append((msg, response))

          return "", history, None
    except Exception as e:
      import traceback
      history.append((msg, f"⚠ ERROR: {traceback.format_exc()}"))
      return "", history, None

# ---------------- INSIGHT ----------------

def generate_insight():

    try:
        return generate_weekly_insight(state["username"])
    except:
        return "Insight not available yet."



# ---------------- CSS ----------------

css = """

footer {
display: none !important;
}

.sidebar{
max-width:90px;
}

.sidebar button{
width:50px !important;
height:50px !important;
border-radius:50% !important;
font-size:20px !important;
margin-bottom:14px;
}

.helpline{
position:fixed;
bottom:10px;
left:0;
width:100%;
text-align:center;
font-size:12px;
color:#6B7280;
background:transparent;
z-index:100;
}
"""


# ---------------- UI ----------------

with gr.Blocks(css=css, theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="indigo",
        neutral_hue="gray"
    )) as demo:

    gr.HTML("""
    <div style="display:flex;flex-direction:column;margin-bottom:10px;">
      <div style="font-size:28px;font-weight:700;color:#4F46E5;">
        ✨ Lucent
        </div>
        <div style="font-size:13px;color:#6B7280;">
          A gentle space to talk, reflect, and feel heard.
          </div>
        </div>
        """)

    with gr.Row():

        # sidebar
        with gr.Column(scale=0.6,elem_classes="sidebar"):

            chat_btn = gr.Button("💬", variant="secondary")
            profile_btn = gr.Button("📊", variant="secondary")


        # main area
        with gr.Column(scale=6):

            # chat
            with gr.Column(visible=True) as chat_panel:

                chatbot = gr.Chatbot(
                    value=[[None,
                            "Hi, I'm Lucent ✨\n\nBefore we begin, what should I call you?"]],
                    show_label=False,
                    height=450,
                    bubble_full_width=False
                    )

                msg = gr.Textbox(
                    placeholder="Share what's on your mind...",
                    show_label=False
                )

                send = gr.Button("Send ✨", variant="primary")


            # profile
            with gr.Column(visible=False) as profile_panel:

                with gr.Row():

                    with gr.Column():

                        gr.Markdown("### Mental Profile")

                        radar = gr.Image(
                            show_label=False,
                            interactive=False
                        )

                    with gr.Column():

                        gr.Markdown("### Weekly Insight")

                        insight_box = gr.Textbox(lines=10)

                        insight_btn = gr.Button("Generate Insight")


    gr.HTML(
        "<div class='helpline'>If in crisis, call iCALL helpline: 9152987821</div>"
    )

    chat_btn.click(show_chat,outputs=[chat_panel,profile_panel])
    profile_btn.click(show_profile,outputs=[chat_panel,profile_panel])

    send.click(chat,inputs=[msg,chatbot],outputs=[msg,chatbot,radar])
    msg.submit(chat,inputs=[msg,chatbot],outputs=[msg,chatbot,radar])

    insight_btn.click(generate_insight,outputs=[insight_box])


demo.launch()
