import streamlit as st
import tempfile
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from collections import Counter


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def count_word_family(tokens, word_family):
    return sum(1 for token in tokens if token in word_family)


def normalize_series(series):
    if len(series) == 0 or series.max() == series.min():
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def tokenize_text(text):
    return re.findall(r"\b[\w']+\b", text.lower())


def average_word_length(tokens):
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)


def lexical_diversity(tokens):
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def repetition_score(tokens):
    if not tokens:
        return 0
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated


def clean_transcript_text(text):
    # Remove bracketed stage directions like [Music], [Applause]
    text = re.sub(r"\[.*?\]", " ", text)

    # Remove timestamps like 9:30 or 10:56
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)

    # Remove phrases like "10 minutes, 56 seconds"
    text = re.sub(r"\b\d+\s+minutes?,\s+\d+\s+seconds?\b", " ", text, flags=re.IGNORECASE)

    # Remove lone numbers that often come from transcript timing junk
    text = re.sub(r"\b\d+\b", " ", text)

    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def sentence_count(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def average_sentence_length(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0

    sentence_lengths = [len(tokenize_text(sentence)) for sentence in sentences]
    return sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0


def punctuation_count(text, punctuation_mark):
    return text.count(punctuation_mark)


def count_word_family(tokens, word_family):
    return sum(1 for token in tokens if token in word_family)


def pronoun_density(tokens, lex):
    pronouns = (
        lex["FIRST_PERSON"]
        | lex["SECOND_PERSON"]
        | lex["THIRD_PERSON"]
        | lex["WE"]
        | lex["THEY"]
    )

    if not tokens:
        return 0.0

    pronoun_count = sum(1 for token in tokens if token in pronouns)
    return pronoun_count / len(tokens)


def long_word_count(tokens, min_length=7):
    return sum(1 for token in tokens if len(token) >= min_length)
    # Remove bracketed stage directions like [Music], [Applause]
    text = re.sub(r"\[.*?\]", " ", text)

    # Remove timestamps like 9:30 or 10:56
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)

    # Remove phrases like "10 minutes, 56 seconds"
    text = re.sub(r"\b\d+\s+minutes?,\s+\d+\s+seconds?\b", " ", text, flags=re.IGNORECASE)

    # Remove lone numbers that often come from transcript timing junk
    text = re.sub(r"\b\d+\b", " ", text)

    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ----------------------------
# RHETORICAL WORD LISTS
# ----------------------------
# ----------------------------
# LINGUISTIC WORD LISTS
# ----------------------------
# ----------------------------------
# LANGUAGE LEXICONS
# ----------------------------------

LANGUAGE_LEXICONS = {

    "English": {
        # ----------------
        # RHETORICAL
        # ----------------
        "WE": {"we", "us", "our", "ours"},
        "THEY": {"they", "them", "their", "theirs"},
        "NATIONALIST": {"nation", "country", "america", "american", "people", "citizens", "homeland"},
        "CRISIS": {"crisis", "threat", "danger", "war", "violence", "crime", "disaster", "emergency"},

        # ----------------
        # LINGUISTIC
        # ----------------
        "NEGATION": {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "cannot", "cant", "won't", "dont", "isnt", "arent", "wasnt", "werent"},
        "MODAL": {"must", "will", "should", "can", "could", "may", "might", "shall", "need", "needs", "needed"},
        "CONJUNCTION": {"and", "but", "or", "so", "yet", "for", "nor"},
        "SUBORDINATION": {"because", "although", "though", "while", "when", "if", "since", "unless", "after", "before", "until", "whereas"},
        "CERTAINTY": {"always", "never", "must", "certainly", "clearly", "obviously", "undoubtedly", "definitely"},

        "FIRST_PERSON": {"i", "me", "my", "mine", "we", "us", "our", "ours"},
        "SECOND_PERSON": {"you", "your", "yours"},
        "THIRD_PERSON": {"he", "she", "they", "them", "their", "theirs"}
    },

    "Italian": {
        # ----------------
        # RHETORICAL
        # ----------------
        "WE": {"noi", "nostro", "nostri", "nostra"},
        "THEY": {"loro"},
        "NATIONALIST": {"nazione", "paese", "italia", "italiano", "popolo", "cittadini", "patria"},
        "CRISIS": {"crisi", "minaccia", "pericolo", "guerra", "violenza", "crimine", "emergenza"},

        # ----------------
        # LINGUISTIC
        # ----------------
        "NEGATION": {"non", "mai", "nessuno", "niente", "neanche", "nemmeno"},
        "MODAL": {"deve", "devono", "dovrebbe", "posso", "può", "possono", "potrebbe", "voglio", "vuole"},
        "CONJUNCTION": {"e", "ma", "o", "quindi", "però", "né"},
        "SUBORDINATION": {"perché", "anche se", "mentre", "quando", "se", "poiché", "dopo", "prima", "finché"},
        "CERTAINTY": {"sempre", "mai", "certamente", "chiaramente", "ovviamente", "sicuramente"},

        "FIRST_PERSON": {"io", "me", "mio", "noi", "nostro"},
        "SECOND_PERSON": {"tu", "voi", "tuo", "vostro"},
        "THIRD_PERSON": {"lui", "lei", "loro"}
    }
}


# ----------------------------
# APP HEADER
# ----------------------------
st.title("RhetoricTrace")
st.write("Upload a speech and begin analysis.")
st.subheader("Language Selection")

language = st.selectbox(
    "Select language for analysis",
    ["English", "Italian"]
)

# ----------------------------
# TRANSCRIPT UPLOAD
# ----------------------------
st.subheader("Upload Transcript (Optional)")
transcript_file = st.file_uploader("Upload transcript (.txt)", type=["txt"])

transcript_text = None
if transcript_file is not None:
    raw_transcript_text = transcript_file.read().decode("utf-8")
    transcript_text = clean_transcript_text(raw_transcript_text)
    st.success("Transcript loaded and cleaned")


# ----------------------------
# AUDIO UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")
    st.audio(uploaded_file)

    # Save uploaded file temporarily
    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load audio
    y, sr = librosa.load(tmp_path, sr=None)

    # ----------------------------
    # ENERGY + SMOOTHING
    # ----------------------------
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)

    window_size = 50
    rms_smooth = np.convolve(rms, np.ones(window_size) / window_size, mode="same")

    st.subheader("Smoothed Speech Energy Over Time")
    fig, ax = plt.subplots()
    ax.plot(times, rms_smooth)
    ax.set_title("Speech Energy Over Time")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Smoothed Energy (RMS)")
    st.pyplot(fig)

    # ----------------------------
    # PAUSE DETECTION
    # ----------------------------
    st.subheader("Pause Detection")

    pause_threshold = 0.02
    is_pause = rms_smooth < pause_threshold

    pause_segments = []
    start_idx = None

    for i, paused in enumerate(is_pause):
        if paused and start_idx is None:
            start_idx = i
        elif not paused and start_idx is not None:
            end_idx = i - 1
            start_time = times[start_idx]
            end_time = times[end_idx]
            duration = end_time - start_time

            if duration >= 0.2:
                pause_segments.append(
                    {
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "duration_seconds": float(duration),
                    }
                )
            start_idx = None

    if start_idx is not None:
        end_idx = len(times) - 1
        start_time = times[start_idx]
        end_time = times[end_idx]
        duration = end_time - start_time
        if duration >= 0.2:
            pause_segments.append(
                {
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "duration_seconds": float(duration),
                }
            )

    pause_df = pd.DataFrame(pause_segments)

    total_pauses = len(pause_df)
    total_pause_time = pause_df["duration_seconds"].sum() if not pause_df.empty else 0
    mean_pause = pause_df["duration_seconds"].mean() if not pause_df.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total pauses", total_pauses)
    col2.metric("Total pause time (s)", round(float(total_pause_time), 2))
    col3.metric("Average pause length (s)", round(float(mean_pause), 2))

    st.subheader("Speech Energy with Pauses Highlighted")
    fig2, ax2 = plt.subplots()
    ax2.plot(times, rms_smooth, label="Smoothed RMS")

    for pause in pause_segments:
        ax2.axvspan(
            pause["start_time"],
            pause["end_time"],
            alpha=0.15,
            color="gray"
        )

    ax2.set_title("Speech Energy with Detected Pauses")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Smoothed Energy (RMS)")
    st.pyplot(fig2)

    # ----------------------------
    # 5-SECOND SEGMENT ANALYSIS
    # ----------------------------
    st.subheader("5-Second Segment Analysis")

    segment_length = 5.0
    speech_duration = times[-1]
    segment_rows = []

    segment_starts = np.arange(0, speech_duration, segment_length)

    # Build rough transcript chunks
    transcript_chunks = []

    if transcript_text:
        words = transcript_text.split()
        words_per_segment = int(len(words) / len(segment_starts)) if len(segment_starts) > 0 else 0

        for i in range(len(segment_starts)):
            start = i * words_per_segment
            end = start + words_per_segment
            chunk = " ".join(words[start:end])
            transcript_chunks.append(chunk)

    for i, seg_start in enumerate(segment_starts):
        seg_end = min(seg_start + segment_length, speech_duration)

        mask = (times >= seg_start) & (times < seg_end)
        seg_rms = rms_smooth[mask]

        if len(seg_rms) == 0:
            continue

        mean_energy = float(np.mean(seg_rms))
        max_energy = float(np.max(seg_rms))

        seg_pause_count = 0
        seg_pause_time = 0.0

        if not pause_df.empty:
            for _, pause in pause_df.iterrows():
                p_start = pause["start_time"]
                p_end = pause["end_time"]

                overlap_start = max(seg_start, p_start)
                overlap_end = min(seg_end, p_end)

                if overlap_end > overlap_start:
                    seg_pause_count += 1
                    seg_pause_time += (overlap_end - overlap_start)

        segment_text = transcript_chunks[i] if transcript_text and i < len(transcript_chunks) else ""
        padded_text = f" {segment_text.lower()} "
        tokens = tokenize_text(segment_text)
        lex = LANGUAGE_LEXICONS[language]
        
        # ----------------------------
        # RHETORICAL FEATURES
        # ----------------------------
        we_count = count_word_family(tokens, lex["WE"])
        they_count = count_word_family(tokens, lex["THEY"])
        nationalist_count = count_word_family(tokens, lex["NATIONALIST"])
        crisis_count = count_word_family(tokens, lex["CRISIS"])
        antagonism_score = they_count + crisis_count
        
        # ----------------------------
        # LINGUISTIC FEATURES
        # ----------------------------
        tokens = tokenize_text(segment_text)
        word_count = len(tokens)
        avg_len = average_word_length(tokens)
        lex_div = lexical_diversity(tokens)
        negation_count = count_word_family(tokens, lex["NEGATION"])
        modal_count = count_word_family(tokens, lex["MODAL"])
        repetition = repetition_score(tokens)
        sent_count = sentence_count(segment_text)
        avg_sent_len = average_sentence_length(segment_text)
        question_count = punctuation_count(segment_text, "?")
        exclamation_count = punctuation_count(segment_text, "!")
        conjunction_count = count_word_family(tokens, lex["CONJUNCTION"])
        subordination_count = count_word_family(tokens, lex["SUBORDINATION"])
        certainty_count = count_word_family(tokens, lex["CERTAINTY"])

        i_count = count_word_family(tokens, lex["FIRST_PERSON"])
        you_count = count_word_family(tokens, lex["SECOND_PERSON"])
        third_person_count = count_word_family(tokens, lex["THIRD_PERSON"])

        pronoun_density_score = pronoun_density(tokens, lex)
        long_words = long_word_count(tokens)
        segment_rows.append(
            {
                "segment_start": round(seg_start, 2),
                "segment_end": round(seg_end, 2),
                "mean_energy": round(mean_energy, 4),
                "max_energy": round(max_energy, 4),
                "pause_count": int(seg_pause_count),
                "pause_time_seconds": round(seg_pause_time, 2),
                "text": segment_text,
                "we_count": we_count,
                "they_count": they_count,
                "nationalist_count": nationalist_count,
                "crisis_count": crisis_count,
                "antagonism_score": antagonism_score,
                "word_count": word_count,
                "avg_word_length": round(avg_len, 2),
                "lexical_diversity": round(lex_div, 4),
                "negation_count": negation_count,
                "modal_count": modal_count,
                "repetition_score": repetition,
                "sentence_count": sent_count,
                "avg_sentence_length": round(avg_sent_len, 2),
                "question_count": question_count,
                "exclamation_count": exclamation_count,
                "conjunction_count": conjunction_count,
                "subordination_count": subordination_count,
                "certainty_count": certainty_count,
                "i_count": i_count,
                "you_count": you_count,
                "third_person_count": third_person_count,
                "pronoun_density": round(pronoun_density_score, 4),
                "long_word_count": long_words
            }
        )

    segment_df = pd.DataFrame(segment_rows)

    # ----------------------------
    # NORMALIZED SCORES
    # ----------------------------
    segment_df["energy_norm"] = normalize_series(segment_df["mean_energy"])
    segment_df["antagonism_norm"] = normalize_series(segment_df["antagonism_score"])
    segment_df["nationalist_norm"] = normalize_series(segment_df["nationalist_count"])
    segment_df["crisis_norm"] = normalize_series(segment_df["crisis_count"])

    segment_df["word_count_norm"] = normalize_series(segment_df["word_count"])
    segment_df["negation_norm"] = normalize_series(segment_df["negation_count"])
    segment_df["modal_norm"] = normalize_series(segment_df["modal_count"])
    segment_df["repetition_norm"] = normalize_series(segment_df["repetition_score"])

    # Lower lexical diversity often suggests more compressed / repetitive language
    segment_df["lexical_compression_component"] = 1 - normalize_series(segment_df["lexical_diversity"])

    segment_df["combined_score"] = (
        0.35 * segment_df["energy_norm"]
        + 0.35 * segment_df["antagonism_norm"]
        + 0.15 * segment_df["nationalist_norm"]
        + 0.15 * segment_df["crisis_norm"]
    ).round(4)

    segment_df["linguistic_compression_score"] = (
        0.40 * segment_df["repetition_norm"]
         + 0.25 * segment_df["negation_norm"]
        + 0.15 * segment_df["modal_norm"]
        + 0.20 * segment_df["lexical_compression_component"]
     ).round(4)

    st.dataframe(segment_df, use_container_width=True)

    # ----------------------------
    # CSV DOWNLOAD
    # ----------------------------
    st.subheader("Download Segment Data")
    csv_data = segment_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download segment analysis as CSV",
        data=csv_data,
        file_name="rhetorictrace_segment_analysis.csv",
        mime="text/csv"
    )

    # ----------------------------
    # AVERAGE ENERGY BY SEGMENT
    # ----------------------------
    st.subheader("Average Energy by 5-Second Segment")
    fig3, ax3 = plt.subplots()
    ax3.plot(segment_df["segment_start"], segment_df["mean_energy"], marker="o")
    ax3.set_title("Average Energy by Segment")
    ax3.set_xlabel("Segment Start Time (seconds)")
    ax3.set_ylabel("Mean Energy")
    st.pyplot(fig3)

    # ----------------------------
    # ANTAGONISM SCORE BY SEGMENT
    # ----------------------------
    st.subheader("Antagonism Score by 5-Second Segment")
    fig4, ax4 = plt.subplots()
    ax4.plot(segment_df["segment_start"], segment_df["antagonism_score"], marker="o")
    ax4.set_title("Antagonism Score by Segment")
    ax4.set_xlabel("Segment Start Time (seconds)")
    ax4.set_ylabel("Antagonism Score")
    st.pyplot(fig4)

    # ----------------------------
    # RHETORICAL MARKER TRENDS
    # ----------------------------
    st.subheader("Rhetorical Marker Trends by Segment")

    fig5, ax5 = plt.subplots()
    ax5.plot(segment_df["segment_start"], segment_df["we_count"], marker="o", label="We / Us")
    ax5.plot(segment_df["segment_start"], segment_df["nationalist_count"], marker="o", label="Nationalist")
    ax5.plot(segment_df["segment_start"], segment_df["crisis_count"], marker="o", label="Crisis")
    ax5.plot(segment_df["segment_start"], segment_df["antagonism_score"], marker="o", label="Antagonism")

    ax5.set_title("Rhetorical Marker Trends by Segment")
    ax5.set_xlabel("Segment Start Time (seconds)")
    ax5.set_ylabel("Marker Count")
    ax5.legend()
    st.pyplot(fig5)

    # ----------------------------
    # ENERGY + ANTAGONISM OVERLAY
    # ----------------------------
    st.subheader("Energy + Antagonism Overlay")

    fig6, ax6 = plt.subplots()

    ax6.plot(
        segment_df["segment_start"],
        segment_df["mean_energy"],
        marker="o",
        label="Mean Energy"
    )
    ax6.set_xlabel("Segment Start Time (seconds)")
    ax6.set_ylabel("Mean Energy")

    ax6b = ax6.twinx()
    ax6b.plot(
        segment_df["segment_start"],
        segment_df["antagonism_score"],
        marker="s",
        linestyle="--",
        label="Antagonism Score"
    )
    ax6b.set_ylabel("Antagonism Score")

    lines_1, labels_1 = ax6.get_legend_handles_labels()
    lines_2, labels_2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    ax6.set_title("Energy and Antagonism by Segment")
    st.pyplot(fig6)

    # ----------------------------
    # LINGUISTIC COMPRESSION BY SEGMENT
    # ----------------------------
    st.subheader("Linguistic Compression by Segment")
    fig7, ax7 = plt.subplots()
    ax7.plot(segment_df["segment_start"], segment_df["linguistic_compression_score"], marker="o")
    ax7.set_title("Linguistic Compression by Segment")
    ax7.set_xlabel("Segment Start Time (seconds)")
    ax7.set_ylabel("Linguistic Compression Score")
    st.pyplot(fig7)

    # ----------------------------
    # ENERGY + LINGUISTIC COMPRESSION OVERLAY
    # ----------------------------
    st.subheader("Energy + Linguistic Compression Overlay")

    fig8, ax8 = plt.subplots()

    ax8.plot(
        segment_df["segment_start"],
        segment_df["mean_energy"],
        marker="o",
        label="Mean Energy"
    )
    ax8.set_xlabel("Segment Start Time (seconds)")
    ax8.set_ylabel("Mean Energy")

    ax8b = ax8.twinx()
    ax8b.plot(
        segment_df["segment_start"],
        segment_df["linguistic_compression_score"],
        marker="s",
        linestyle="--",
        label="Linguistic Compression"
    )
    ax8b.set_ylabel("Linguistic Compression Score")

    lines_1, labels_1 = ax8.get_legend_handles_labels()
    lines_2, labels_2 = ax8b.get_legend_handles_labels()
    ax8.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    ax8.set_title("Energy and Linguistic Compression by Segment")
    st.pyplot(fig8)

    # ----------------------------
    # TOP SCORING SEGMENTS
    # ----------------------------
    st.subheader("Top Overall Scoring Segments")
    top_combined = segment_df.sort_values(by="combined_score", ascending=False).head(10)
    st.dataframe(top_combined, use_container_width=True)

    # ----------------------------
    # TOP ENERGY SEGMENTS
    # ----------------------------
    st.subheader("Top Energy Segments")
    top_energy = segment_df.sort_values(by="mean_energy", ascending=False).head(10)
    st.dataframe(top_energy, use_container_width=True)

    # ----------------------------
    # TOP ANTAGONISTIC SEGMENTS
    # ----------------------------
    st.subheader("Top Antagonistic Segments")
    top_antagonistic = segment_df.sort_values(by="antagonism_score", ascending=False).head(10)
    st.dataframe(top_antagonistic, use_container_width=True)

    # ----------------------------
    # TOP NATIONALIST SEGMENTS
    # ----------------------------
    st.subheader("Top Nationalist Segments")
    top_nationalist = segment_df.sort_values(by="nationalist_count", ascending=False).head(10)
    st.dataframe(top_nationalist, use_container_width=True)

    # ----------------------------
    # TOP WE / US SEGMENTS
    # ----------------------------
    st.subheader("Top We / Us Segments")
    top_we = segment_df.sort_values(by="we_count", ascending=False).head(10)
    st.dataframe(top_we, use_container_width=True)

    # ----------------------------
    # TOP CRISIS SEGMENTS
    # ----------------------------
    st.subheader("Top Crisis Segments")
    top_crisis = segment_df.sort_values(by="crisis_count", ascending=False).head(10)
    st.dataframe(top_crisis, use_container_width=True)

    # ----------------------------
    # TOP LINGUISTICALLY COMPRESSED SEGMENTS
    # ----------------------------
    st.subheader("Top Linguistically Compressed Segments")
    top_linguistic = segment_df.sort_values(by="linguistic_compression_score", ascending=False).head(10)
    st.dataframe(top_linguistic, use_container_width=True)
