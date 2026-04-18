import os
import re
import sys
import json
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
 
# For reading PDF and DOCX resumes
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[WARNING] PyMuPDF not installed. PDF reading disabled.")
    print("          Run: pip install PyMuPDF")
 
try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("[WARNING] python-docx not installed. DOCX reading disabled.")
    print("          Run: pip install python-docx")
 
 
# ═══════════════════════════════════════════════════════════
#  ✏️  USER SETTINGS — Edit these before running (F5)
# ═══════════════════════════════════════════════════════════
 
# Path to a resume file (.pdf, .docx, or .txt)
# Leave as None to use the built-in DEMO resume text below
RESUME_PATH = "C:/Users/SOUPTIK/SBasu_Resume.pdf"
 
# Set to True to show a radar chart of personality scores
SHOW_CHART = True
 
# Set to True to save the results as a JSON file
SAVE_RESULTS = True
OUTPUT_JSON = "personality_results.json"
 
 
# ═══════════════════════════════════════════════════════════
#  Demo Resume (used when RESUME_PATH = None)
# ═══════════════════════════════════════════════════════════
 
DEMO_RESUME = """
John Smith
Software Engineer | Team Lead
 
SUMMARY
Passionate and creative software engineer with 8 years of experience leading
cross-functional teams. I thrive in collaborative environments and enjoy
mentoring junior developers. Known for my empathetic communication style and
ability to adapt quickly to new technologies and challenges.
 
EXPERIENCE
Senior Software Engineer — TechCorp (2019–Present)
- Led a team of 10 engineers to deliver a cloud migration project on time
- Organized weekly knowledge-sharing sessions to foster team growth
- Introduced agile ceremonies that improved sprint velocity by 30%
- Collaborated with product managers and designers to refine user experience
 
Software Engineer — StartupXYZ (2016–2019)
- Developed RESTful APIs and microservices using Python and Django
- Participated actively in code reviews and pair programming sessions
- Volunteered to mentor 3 intern developers during summer programs
 
EDUCATION
B.Sc. Computer Science — State University (2016)
Graduated with Distinction | GPA: 3.9
 
SKILLS
Python, Django, JavaScript, React, Docker, Kubernetes, SQL, Git,
Agile/Scrum, Technical Writing, Public Speaking, Team Leadership
 
ACHIEVEMENTS
- Speaker at PyCon 2022: "Building Empathetic APIs"
- Open source contributor (500+ GitHub stars)
- Organized annual hackathon for 200+ participants
 
INTERESTS
Reading research papers, painting, hiking, teaching coding to underprivileged youth
"""
 
 
# ═══════════════════════════════════════════════════════════
#  Personality Trait Keyword Lexicon (Big Five)
# ═══════════════════════════════════════════════════════════
 
TRAIT_KEYWORDS = {
    "Openness": [
        "creative", "innovative", "curious", "imaginative", "artistic",
        "inventive", "explore", "research", "design", "vision", "idea",
        "experiment", "novel", "diverse", "philosophy", "culture", "art",
        "literature", "music", "poetry", "travel", "learn", "discovery",
        "abstract", "theoretical", "intellectual", "open-minded", "flexible",
        "painting", "writing", "reading", "hackathon", "speaker", "publish"
    ],
    "Conscientiousness": [
        "organized", "detail", "punctual", "responsible", "reliable",
        "systematic", "structured", "plan", "schedule", "deadline", "goal",
        "achieve", "accurate", "thorough", "efficient", "discipline",
        "methodical", "diligent", "committed", "consistent", "GPA",
        "distinction", "certified", "compliance", "quality", "audit",
        "documentation", "process", "on time", "deliver", "milestone"
    ],
    "Extraversion": [
        "leadership", "team", "collaborate", "communicate", "present",
        "social", "network", "public", "mentor", "coach", "motivate",
        "engage", "enthusiastic", "outgoing", "energetic", "conference",
        "speaker", "event", "organize", "volunteer", "community", "group",
        "workshop", "seminar", "meeting", "client", "stakeholder", "lead"
    ],
    "Agreeableness": [
        "empathetic", "supportive", "helpful", "cooperative", "friendly",
        "patient", "kind", "compassionate", "nurture", "assist", "trust",
        "respect", "harmony", "inclusive", "volunteer", "charity", "mentor",
        "listen", "understand", "feedback", "care", "youth", "underprivileged",
        "community", "share", "collaborate", "peer", "pair"
    ],
    "Emotional Stability": [
        "calm", "stable", "resilient", "adaptable", "consistent", "composed",
        "stress", "pressure", "challenge", "overcome", "recovery", "balance",
        "mindful", "focus", "reliable", "steady", "confident", "manage",
        "handle", "cope", "maintain", "solution", "problem-solving", "crisis",
        "deliver", "deadlines", "agile", "pivot", "flexible", "persevere"
    ]
}
 
 
# ═══════════════════════════════════════════════════════════
#  Step 1 — Read Resume Text
# ═══════════════════════════════════════════════════════════
 
def read_resume(path):
    """Read and return plain text from a .pdf, .docx, or .txt file."""
    ext = os.path.splitext(path)[1].lower()
 
    if ext == ".pdf":
        if not PDF_SUPPORT:
            sys.exit("[ERROR] PyMuPDF required for PDF. Run: pip install PyMuPDF")
        text = ""
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        return text
 
    elif ext == ".docx":
        if not DOCX_SUPPORT:
            sys.exit("[ERROR] python-docx required for DOCX. Run: pip install python-docx")
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
 
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
 
    else:
        sys.exit(f"[ERROR] Unsupported file type: {ext}. Use .pdf, .docx, or .txt")
 
 
# ═══════════════════════════════════════════════════════════
#  Step 2 — Preprocess Text
# ═══════════════════════════════════════════════════════════
 
def download_nltk_data():
    """Silently ensure required NLTK datasets are available."""
    for pkg in ["stopwords", "punkt", "wordnet"]:
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
        except LookupError:
            print(f"[INFO] Downloading NLTK data: {pkg}")
            nltk.download(pkg, quiet=True)
 
def preprocess(text):
    """Lowercase, remove noise, tokenize, lemmatize, remove stopwords."""
    download_nltk_data()
 
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
 
    text = text.lower()
    text = re.sub(r"[^a-z\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
 
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
 
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return tokens, " ".join(tokens)
 
 
# ═══════════════════════════════════════════════════════════
#  Step 3 — Score Personality Traits
# ═══════════════════════════════════════════════════════════
 
def score_traits(tokens, clean_text):
    """
    Score each Big Five trait using two methods combined:
      1. Keyword frequency matching against the lexicon
      2. TF-IDF weighted keyword scoring
    Returns a dict of raw scores per trait.
    """
    token_set = tokens  # list (allows duplicates for frequency count)
 
    # --- Method 1: Keyword frequency ---
    freq_scores = {}
    for trait, keywords in TRAIT_KEYWORDS.items():
        count = sum(token_set.count(kw) for kw in keywords)
        freq_scores[trait] = count
 
    # --- Method 2: TF-IDF similarity ---
    # Build a pseudo-document per trait from its keywords
    trait_docs = [" ".join(kws) for kws in TRAIT_KEYWORDS.values()]
    trait_names = list(TRAIT_KEYWORDS.keys())
 
    corpus = [clean_text] + trait_docs
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
 
    resume_vec = tfidf_matrix[0]
    tfidf_scores = {}
    for i, trait in enumerate(trait_names):
        trait_vec = tfidf_matrix[i + 1]
        # Cosine similarity
        dot = (resume_vec * trait_vec.T).toarray()[0][0]
        norm_r = np.linalg.norm(resume_vec.toarray())
        norm_t = np.linalg.norm(trait_vec.toarray())
        sim = dot / (norm_r * norm_t + 1e-9)
        tfidf_scores[trait] = sim
 
    # --- Combine: 60% frequency + 40% TF-IDF ---
    raw_scores = {}
    for trait in trait_names:
        raw_scores[trait] = (
            0.6 * freq_scores.get(trait, 0) +
            0.4 * tfidf_scores.get(trait, 0) * 100  # scale tfidf to similar range
        )
 
    return raw_scores
 
 
def normalize_scores(raw_scores):
    """Normalize scores to a 0–100 scale."""
    values = np.array(list(raw_scores.values())).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(20, 100))  # min 20 to avoid zero scores
    scaled = scaler.fit_transform(values).flatten()
    return {trait: round(float(score), 1) for trait, score in
            zip(raw_scores.keys(), scaled)}
 
 
# ═══════════════════════════════════════════════════════════
#  Step 4 — Interpret Results
# ═══════════════════════════════════════════════════════════
 
TRAIT_DESCRIPTIONS = {
    "Openness": {
        "high":   "Highly imaginative and curious. Embraces new ideas, creative problem-solving, and diverse experiences.",
        "medium": "Moderately open. Balances conventional approaches with occasional creative exploration.",
        "low":    "Prefers routine and structure. Focuses on practical, proven methods over novel ideas."
    },
    "Conscientiousness": {
        "high":   "Highly organized, detail-oriented, and dependable. Sets clear goals and follows through.",
        "medium": "Generally responsible but may occasionally miss details under pressure.",
        "low":    "More spontaneous and flexible. May struggle with strict deadlines or rigid structures."
    },
    "Extraversion": {
        "high":   "Energetic, sociable, and assertive. Thrives in team environments and leadership roles.",
        "medium": "Comfortable in both social and independent settings. Adaptable communicator.",
        "low":    "Prefers working independently. Thoughtful and reserved; strength in deep focused work."
    },
    "Agreeableness": {
        "high":   "Compassionate, cooperative, and empathetic. Prioritizes harmony and helping others.",
        "medium": "Generally cooperative with some independent assertiveness when needed.",
        "low":    "Direct and competitive. Prioritizes task outcomes over group consensus."
    },
    "Emotional Stability": {
        "high":   "Calm and resilient under pressure. Handles setbacks with composure and consistency.",
        "medium": "Generally stable with occasional stress responses in high-pressure situations.",
        "low":    "More sensitive to stress. May need supportive environments to perform at best."
    }
}
 
ROLE_SUGGESTIONS = {
    "Openness":            ["UX Designer", "Product Manager", "Research Scientist", "Creative Director"],
    "Conscientiousness":   ["Project Manager", "Data Analyst", "Quality Assurance", "Accountant"],
    "Extraversion":        ["Sales Manager", "HR Manager", "Team Lead", "Business Development"],
    "Agreeableness":       ["Customer Success", "Social Worker", "Teacher", "Counselor"],
    "Emotional Stability": ["Crisis Manager", "Healthcare Worker", "Operations Lead", "Support Engineer"]
}
 
def interpret(scores):
    """Return descriptive interpretation for each trait score."""
    interpretations = {}
    for trait, score in scores.items():
        if score >= 70:
            level = "high"
        elif score >= 45:
            level = "medium"
        else:
            level = "low"
        interpretations[trait] = {
            "score": score,
            "level": level.capitalize(),
            "description": TRAIT_DESCRIPTIONS[trait][level]
        }
    return interpretations
 
def suggest_roles(scores, top_n=3):
    """Suggest job roles based on the top-scoring traits."""
    sorted_traits = sorted(scores, key=scores.get, reverse=True)[:top_n]
    roles = []
    for trait in sorted_traits:
        roles.extend(ROLE_SUGGESTIONS[trait])
    # Deduplicate while preserving order
    seen = set()
    unique_roles = []
    for r in roles:
        if r not in seen:
            seen.add(r)
            unique_roles.append(r)
    return unique_roles, sorted_traits
 
 
# ═══════════════════════════════════════════════════════════
#  Step 5 — Visualize (Radar Chart)
# ═══════════════════════════════════════════════════════════
 
def plot_radar(scores, candidate_name="Candidate"):
    """Draw a radar / spider chart for the Big Five scores."""
    traits = list(scores.keys())
    values = [scores[t] for t in traits]
 
    # Close the radar loop
    traits_loop  = traits + [traits[0]]
    values_loop  = values + [values[0]]
 
    angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
    angles += angles[:1]
 
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
 
    # Style
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
 
    ax.plot(angles, values_loop, color="#00e5ff", linewidth=2.5, linestyle="solid")
    ax.fill(angles, values_loop, color="#00e5ff", alpha=0.25)
 
    # Grid
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"],
                       color="#888888", fontsize=8)
    ax.yaxis.grid(True, color="#30363d", linestyle="--", linewidth=0.8)
    ax.xaxis.grid(True, color="#30363d", linestyle="--", linewidth=0.8)
    ax.spines["polar"].set_color("#30363d")
 
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [f"{t}\n({scores[t]:.0f})" for t in traits],
        color="#e6edf3", fontsize=11, fontweight="bold"
    )
 
    ax.set_title(f"Big Five Personality Profile\n{candidate_name}",
                 color="#00e5ff", fontsize=14, fontweight="bold", pad=20)
 
    plt.tight_layout()
    plt.savefig("personality_radar.png", dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("[INFO] Radar chart saved to: personality_radar.png")
    plt.show()
 
 
# ═══════════════════════════════════════════════════════════
#  Step 6 — Print Report
# ═══════════════════════════════════════════════════════════
 
def print_report(interpretations, suggested_roles, dominant_traits):
    """Print a formatted personality report to the Spyder console."""
    divider = "═" * 60
 
    print(f"\n{divider}")
    print("   PERSONALITY PREDICTION REPORT — Big Five Analysis")
    print(divider)
 
    for trait, info in interpretations.items():
        bar_filled = int(info["score"] / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"\n  {trait:<22} [{bar}] {info['score']:.1f}/100  ({info['level']})")
        print(f"  → {info['description']}")
 
    print(f"\n{divider}")
    print("  DOMINANT TRAITS:")
    for t in dominant_traits:
        print(f"    ✔ {t}  ({interpretations[t]['score']:.1f})")
 
    print(f"\n{divider}")
    print("  SUGGESTED ROLES BASED ON PERSONALITY:")
    for i, role in enumerate(suggested_roles, 1):
        print(f"    {i:>2}. {role}")
 
    print(f"\n{divider}\n")
 
 
# ═══════════════════════════════════════════════════════════
#  MAIN — Press F5 in Spyder to run
# ═══════════════════════════════════════════════════════════
 
print("\n[INFO] Personality Prediction System — Starting...\n")
 
# --- Load resume text ---
if RESUME_PATH:
    if not os.path.exists(RESUME_PATH):
        sys.exit(f"[ERROR] File not found: {RESUME_PATH}")
    print(f"[INFO] Reading resume from: {RESUME_PATH}")
    resume_text = read_resume(RESUME_PATH)
else:
    print("[INFO] No RESUME_PATH set. Using built-in demo resume.")
    resume_text = DEMO_RESUME
 
print(f"[INFO] Resume loaded ({len(resume_text)} characters).\n")
 
# --- Preprocess ---
print("[INFO] Preprocessing text...")
tokens, clean_text = preprocess(resume_text)
print(f"[INFO] Extracted {len(tokens)} meaningful tokens.\n")
 
# --- Score traits ---
print("[INFO] Scoring personality traits...")
raw_scores   = score_traits(tokens, clean_text)
final_scores = normalize_scores(raw_scores)
 
# --- Interpret ---
interpretations = interpret(final_scores)
suggested_roles, dominant_traits = suggest_roles(final_scores)
 
# --- Print report ---
print_report(interpretations, suggested_roles, dominant_traits)
 
# --- Save JSON ---
if SAVE_RESULTS:
    result_data = {
        "scores": final_scores,
        "interpretations": interpretations,
        "dominant_traits": dominant_traits,
        "suggested_roles": suggested_roles
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)
    print(f"[INFO] Results saved to: {OUTPUT_JSON}")
 
# --- Radar chart ---
if SHOW_CHART:
    plot_radar(final_scores, candidate_name="Resume Analysis")
 
print("\n[INFO] Analysis complete.")