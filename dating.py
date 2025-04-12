import random
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define trait categories and keywords
TRAIT_CATEGORIES = ["Interests", "Communication Style", "Life Preferences"]
TRAITS = {
    "Interests": ["music", "poetry", "art", "hiking", "tech", "adventure", "cooking", "reading", "photography"],
    "Communication Style": ["funny", "sarcastic", "empathetic", "deep thinker", "introvert", "extrovert", "emotional", "calm"],
    "Life Preferences": ["values silence", "talkative", "loves nature", "dislikes small talk", "spiritual", "active", "mindful"]
}

POSITIVE_TONE = ["funny", "empathetic", "loves", "calm", "adventure", "poetry", "spiritual", "active", "deep thinker"]
EMOTIONAL_TONE = ["emotional", "introvert", "values silence", "dislikes small talk", "reading", "calm", "thoughtful"]

# Utility functions
def extract_traits(description):
    traits = {cat: [] for cat in TRAIT_CATEGORIES}
    words = re.findall(r'\b\w[\w\s\-]*\w\b', description.lower())
    for cat in TRAIT_CATEGORIES:
        for trait in TRAITS[cat]:
            if any(trait in word for word in words):
                traits[cat].append(trait)
    return traits

def flatten_traits(traits_dict):
    return sum(traits_dict.values(), [])

def get_embedding(text):
    return model.encode([text])[0]

def calculate_similarity(emb1, emb2):
    return round(cosine_similarity([emb1], [emb2])[0][0] * 100, 2)

def category_score(traits1, traits2):
    return {
        cat: round((len(set(traits1[cat]) & set(traits2[cat])) / max(1, len(set(traits1[cat]) | set(traits2[cat])))) * 100, 2)
        for cat in TRAIT_CATEGORIES
    }

def twin_emotion(traits):
    flat = flatten_traits(traits)
    if sum(t in POSITIVE_TONE for t in flat) >= 3:
        return "😄 upbeat and positive"
    elif any(t in EMOTIONAL_TONE for t in flat):
        return "😊 reflective and deep"
    return "😎 chill and balanced"

def draw_donut_chart(scores, user1, user2):
    colors = ['#FF6F61', '#6B5B95', '#88B04B']
    fig = go.Figure()
    for idx, (cat, score) in enumerate(scores.items()):
        fig.add_trace(go.Pie(
            values=[score, 100 - score],
            labels=[f"{cat} Match", ""],
            hole=0.6,
            marker_colors=[colors[idx], '#f0f0f0'],
            textinfo='label+percent',
            domain={'x': [idx * 0.33, (idx + 1) * 0.33]}
        ))
    fig.update_layout(
        title_text=f"💞 Category-wise Compatibility between {user1} & {user2}",
        annotations=[dict(text=cat, x=0.16 + idx*0.34, y=0.5, showarrow=False, font_size=14)
                     for idx, cat in enumerate(["Interests", "Comm.", "Life"])],
        showlegend=False,
        height=400
    )
    return fig

def ai_twin_reply(name1, name2, traits1, traits2):
    mood1, mood2 = twin_emotion(traits1), twin_emotion(traits2)
    common_traits = set(flatten_traits(traits1)) & set(flatten_traits(traits2))

    reply1 = f"Hey {name2}, I'm {name1}'s AI twin. I'm feeling {mood1} today! "
    reply2 = f"Hey {name1}, I'm {name2}'s AI twin. I'm feeling {mood2} today! "

    if common_traits:
        shared = ', '.join(list(common_traits)[:2])
        reply1 += f"Since you both enjoy {shared}, I bet you'd have some awesome stories to share!"
        reply2 += f"Wow, you two both enjoy {shared}, that's a cool connection!"
    else:
        reply1 += "It looks like you two are quite unique in your own ways! Let's explore that."
        reply2 += "It's exciting that you both have such different vibes. Can’t wait to learn more!"

    return reply1, reply2

def suggest_icebreaker(common_traits):
    if common_traits:
        trait = random.choice(list(common_traits))
        return f"Since you both connect over **{trait}**, ask: *'What’s a moment in your life where {trait} played a key role?'*"
    return random.choice([
        "Try: *'If you could have dinner with anyone, living or dead, who would it be and why?'*",
        "Try: *'If you could instantly acquire any skill, what would it be and why?'*",
        "Try: *'What’s a fun or quirky habit you have that not many people know about?'*",
    ])

def generate_name():
    return random.choice(["Echo", "Nova", "Lumi", "Orion", "Kai", "Zara", "Riven", "Skye", "Aeris", "Theo"])

def run_ai_dating_simulation():
    st.set_page_config(page_title="AI Twin Dating Matchmaker", layout="wide")
    st.title("💬 AI Twin Dating Simulation")
    st.write("✨ Let's discover some soulful connections through AI.")

    num_users = st.number_input("Enter number of users:", min_value=2, max_value=10, value=2)
    users_data = []

    for i in range(num_users):
        with st.expander(f"👤 Enter Details for User {i+1}"):
            name = st.text_input("Name:", key=f"name_{i}")
            desc = st.text_area("Describe yourself:", key=f"desc_{i}")
            gender = st.selectbox("Gender:", options=["Male", "Female"], key=f"gender_{i}")
            if name and desc:
                users_data.append((name, desc, gender))

    st.markdown("---")
    results = []

    if len(users_data) >= 2:
        st.subheader("🔄 Matchmaking Results")
        for i in range(len(users_data)):
            for j in range(i + 1, len(users_data)):
                u1, d1, g1 = users_data[i]
                u2, d2, g2 = users_data[j]

                if g1 != g2:
                    traits1 = extract_traits(d1)
                    traits2 = extract_traits(d2)

                    st.info(f"### 🔍 Matching: {u1} ❤️ {u2}")
                    for cat in TRAIT_CATEGORIES:
                        st.markdown(f"**{u1}'s {cat}:** `{', '.join(traits1[cat]) if traits1[cat] else 'None'}`")
                        st.markdown(f"**{u2}'s {cat}:** `{', '.join(traits2[cat]) if traits2[cat] else 'None'}`")

                    r1, r2 = ai_twin_reply(u1, u2, traits1, traits2)
                    st.success(f"🧠 {generate_name()} (like {u1}): {r1}")
                    st.success(f"🧠 {generate_name()} (like {u2}): {r2}")

                    common = set(flatten_traits(traits1)) & set(flatten_traits(traits2))
                    if common:
                        st.markdown(f"🎯 **Common Traits:** `{', '.join(common)}`")
                    else:
                        st.warning("🎯 No common traits, but uniqueness sparks curiosity!")

                    st.markdown("💬 **Icebreaker Suggestion:**")
                    st.markdown(suggest_icebreaker(common))

                    sim = calculate_similarity(get_embedding(d1), get_embedding(d2))
                    scores = category_score(traits1, traits2)
                    st.plotly_chart(draw_donut_chart(scores, u1, u2), use_container_width=True)
                    results.append((u1, u2, sim))

                    st.markdown("<hr style='border: 1px solid #bbb;'>", unsafe_allow_html=True)

        if results:
            df = pd.DataFrame(results, columns=["User1", "User2", "Similarity"])
            st.subheader("📊 Overall Matching Visualisation")
            fig = go.Figure(data=[
                go.Bar(
                    x=df["User1"] + " & " + df["User2"],
                    y=df["Similarity"],
                    marker_color='lightsalmon',
                    text=df["Similarity"],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="All Match Similarities (%)",
                xaxis_title="User Pairs",
                yaxis_title="Similarity",
                yaxis_range=[0, 100],
                height=500
            )
            st.plotly_chart(fig)

            # 💖 Final Match Recommendations
            st.subheader("💖 Final AI Dating Suggestions")

            matched = set()
            sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
            recommendations = []

            for u1, u2, score in sorted_results:
                if u1 not in matched and u2 not in matched:
                    matched.add(u1)
                    matched.add(u2)
                    recommendations.append((u1, u2, score))

            if recommendations:
                for u1, u2, score in recommendations:
                    st.success(f"💌 Match Made: **{u1}** should date **{u2}** (Compatibility: {score}%)")
            else:
                st.warning("😅 Not enough compatible pairs to recommend unique matches.")

if __name__ == "__main__":
    run_ai_dating_simulation()
