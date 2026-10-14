# ============================================================

# IPL FORENSICS

# Data-Driven IPL Match Investigation System

# ============================================================



import streamlit as st

from pathlib import Path

import pandas as pd

import numpy as np

import plotly.express as px

import json

import joblib





# ============================================================

# PAGE CONFIGURATION

# ============================================================



st.set_page_config(

    page_title="IPL Forensics",

    page_icon="🏏",

    layout="wide",

    initial_sidebar_state="expanded"

)





# ============================================================

# PROJECT PATHS

# ============================================================



BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "models"





# ============================================================

# CUSTOM CSS

# ============================================================



st.markdown(

    """

    <style>



    .main-title {

        font-size: 48px;

        font-weight: 800;

        margin-bottom: 0px;

    }



    .subtitle {

        font-size: 19px;

        opacity: 0.75;

        margin-top: 0px;

    }



    .section-title {

        font-size: 28px;

        font-weight: 700;

    }



    .forensic-box {

        padding: 20px;

        border-radius: 12px;

        border: 1px solid rgba(128,128,128,0.25);

        margin-bottom: 15px;

    }



    .small-text {

        font-size: 14px;

        opacity: 0.7;

    }



    </style>

    """,

    unsafe_allow_html=True

)





# ============================================================

# LOAD DATA

# ============================================================



@st.cache_data

def load_data():



    deliveries = pd.read_csv(

        DATA_DIR / "ipl_deliveries.csv"

    )



    turning_points = pd.read_csv(

        DATA_DIR / "top_turning_points.csv"

    )



    what_if = pd.read_csv(

        DATA_DIR / "what_if_scenarios.csv"

    )



    match_autopsy = pd.read_csv(

        DATA_DIR / "match_autopsy_summary.csv"

    )



    metadata_file = MODEL_DIR / "model_metadata.json"



    metadata = {}



    if metadata_file.exists():



        with open(

            metadata_file,

            "r",

            encoding="utf-8"

        ) as f:

            metadata = json.load(f)



    return (

        deliveries,

        turning_points,

        what_if,

        match_autopsy,

        metadata

    )





# ============================================================

# LOAD DATA SAFELY

# ============================================================



try:



    (

        deliveries,

        turning_points,

        what_if,

        match_autopsy,

        metadata

    ) = load_data()



except Exception as e:



    st.error(

        "IPL Forensics could not load the processed data."

    )



    st.code(str(e))



    st.stop()





# ============================================================

# DATA PREPARATION

# ============================================================



deliveries["match_id"] = pd.to_numeric(

    deliveries["match_id"],

    errors="coerce"

)



turning_points["match_id"] = pd.to_numeric(

    turning_points["match_id"],

    errors="coerce"

)



what_if["match_id"] = pd.to_numeric(

    what_if["match_id"],

    errors="coerce"

)



match_autopsy["match_id"] = pd.to_numeric(

    match_autopsy["match_id"],

    errors="coerce"

)





# ============================================================

# SIDEBAR

# ============================================================



st.sidebar.title("🏏 IPL FORENSICS")



st.sidebar.markdown(

    """

    ### Match Investigation



    Use this dashboard to investigate:



    🎯 Turning Points  

    📊 Win Probability  

    🕵️ Match Autopsy  

    🔮 What-If Scenarios  

    ⏱️ Event Timeline  

    📈 Match Statistics

    """

)



st.sidebar.divider()





# ============================================================

# MATCH SELECTION

# ============================================================



match_options = sorted(

    deliveries["match_id"]

    .dropna()

    .unique()

    .tolist()

)



if not match_options:



    st.error("No matches are available.")

    st.stop()





selected_match = st.sidebar.selectbox(

    "Select IPL Match",

    match_options

)





# ============================================================

# SELECT MATCH DATA

# ============================================================



match_data = deliveries[

    deliveries["match_id"] == selected_match

].copy()





# ============================================================

# BASIC MATCH INFORMATION

# ============================================================



teams = (

    match_data["batting_team"]

    .dropna()

    .unique()

    .tolist()

)



winner_values = (

    match_data["winner"]

    .dropna()

    .unique()

    .tolist()

)



winner = (

    winner_values[0]

    if winner_values

    else "Unknown"

)



loser = "Unknown"



if winner != "Unknown":



    loser_candidates = [

        team

        for team in teams

        if team != winner

    ]



    if loser_candidates:

        loser = loser_candidates[0]





# ============================================================

# HEADER

# ============================================================



st.markdown(

    '<div class="main-title">🏏 IPL FORENSICS</div>',

    unsafe_allow_html=True

)



st.markdown(

    '<div class="subtitle">'

    'Data-Driven IPL Match Investigation System'

    '</div>',

    unsafe_allow_html=True

)



st.write("")



st.markdown(

    """

    **IPL Forensics** investigates a cricket match like a

    forensic case — identifying **when the match changed,

    what caused the major momentum shifts, and how different

    scenarios could have changed the estimated outcome.**

    """

)





# ============================================================

# MATCH HEADER

# ============================================================



st.divider()



st.header("🔎 Match Investigation")



col1, col2, col3, col4 = st.columns(4)



with col1:



    st.metric(

        "Match ID",

        int(selected_match)

    )



with col2:



    st.metric(

        "Teams",

        f"{teams[0] if len(teams) > 0 else 'Unknown'}"

        + " vs "

        + f"{teams[1] if len(teams) > 1 else 'Unknown'}"

    )



with col3:



    st.metric(

        "Innings",

        int(match_data["innings"].nunique())

    )



with col4:



    st.metric(

        "Winner",

        winner

    )





# ============================================================

# MATCH DETAILS

# ============================================================



if "date" in match_data.columns:



    date_value = match_data["date"].dropna()



    if len(date_value) > 0:



        st.caption(

            f"📅 Date: {date_value.iloc[0]}"

        )



if "venue" in match_data.columns:



    venue_value = match_data["venue"].dropna()



    if len(venue_value) > 0:



        st.caption(

            f"📍 Venue: {venue_value.iloc[0]}"

        )





# ============================================================

# TABS

# ============================================================



tab1, tab2, tab3, tab4 = st.tabs(

    [

        "📊 Match Overview",

        "🔥 Forensics",

        "🔮 What-If Simulator",

        "⏱️ Timeline"

    ]

)





# ============================================================

# TAB 1 — MATCH OVERVIEW

# ============================================================



with tab1:



    st.header("📊 Match Overview")





    # --------------------------------------------------------

    # SCORE PROGRESSION

    # --------------------------------------------------------



    st.subheader("📈 Score Progression")



    if all(

        col in match_data.columns

        for col in [

            "innings",

            "over",

            "delivery",

            "score"

        ]

    ):



        score_chart = match_data.copy()



        score_chart["Ball"] = range(

            1,

            len(score_chart) + 1

        )



        score_chart["Innings Label"] = (

            "Innings "

            + score_chart["innings"].astype(str)

            + " - "

            + score_chart["batting_team"].astype(str)

        )



        fig_score = px.line(

            score_chart,

            x="Ball",

            y="score",

            color="Innings Label",

            markers=False,

            title="Score Progression"

        )



        fig_score.update_layout(

            height=450,

            xaxis_title="Delivery",

            yaxis_title="Cumulative Score",

            legend_title="Innings"

        )



        st.plotly_chart(

            fig_score,

            use_container_width=True

        )



    else:



        st.info(

            "Score progression data is unavailable."

        )





    # --------------------------------------------------------

    # FINAL SCORE SUMMARY

    # --------------------------------------------------------



    st.subheader("🏏 Innings Summary")



    if all(

        col in match_data.columns

        for col in [

            "innings",

            "batting_team",

            "runs_total",

            "is_wicket"

        ]

    ):



        innings_summary = (

            match_data

            .groupby(

                ["innings", "batting_team"],

                as_index=False

            )

            .agg(

                Runs=("runs_total", "sum"),

                Wickets=("is_wicket", "sum"),

                Deliveries=("match_id", "count")

            )

        )



        st.dataframe(

            innings_summary,

            use_container_width=True,

            hide_index=True

        )





    # --------------------------------------------------------

    # MATCH STATISTICS

    # --------------------------------------------------------



    st.subheader("📋 Match Statistics")



    stat1, stat2, stat3, stat4 = st.columns(4)



    with stat1:



        total_runs = int(

            match_data["runs_total"].sum()

        )



        st.metric(

            "Total Runs",

            total_runs

        )



    with stat2:



        total_wickets = int(

            match_data["is_wicket"].sum()

        )



        st.metric(

            "Wickets",

            total_wickets

        )



    with stat3:



        if "is_four" in match_data.columns:



            fours = int(

                match_data["is_four"].sum()

            )



        else:



            fours = int(

                (

                    match_data["runs_batter"] == 4

                ).sum()

            )



        st.metric(

            "Fours",

            fours

        )



    with stat4:



        if "is_six" in match_data.columns:



            sixes = int(

                match_data["is_six"].sum()

            )



        else:



            sixes = int(

                (

                    match_data["runs_batter"] == 6

                ).sum()

            )



        st.metric(

            "Sixes",

            sixes

        )





# ============================================================

# TAB 2 — FORENSICS

# ============================================================



with tab2:



    st.header("🕵️ Match Forensics")





    # --------------------------------------------------------

    # WIN PROBABILITY

    # --------------------------------------------------------



    st.subheader("🎯 Win Probability")



    if "win_probability" in match_data.columns:



        probability_chart = match_data.copy()



        probability_chart["Ball"] = range(

            1,

            len(probability_chart) + 1

        )



        # The model predicts the probability that the

        # CURRENT BATTING TEAM eventually wins.

        #

        # For a consistent match-level chart, convert it

        # into the probability that the ACTUAL WINNER wins.



        probability_chart[

            "Winner Win Probability"

        ] = np.where(

            probability_chart["batting_team"] == winner,

            probability_chart["win_probability"],

            1 - probability_chart["win_probability"]

        )



        probability_chart[

            "Winner Win Probability (%)"

        ] = (

            probability_chart[

                "Winner Win Probability"

            ] * 100

        )



        fig_probability = px.line(

            probability_chart,

            x="Ball",

            y="Winner Win Probability (%)",

            title="Estimated Probability of the Actual Winner Winning",

            hover_data=[

                "innings",

                "over",

                "delivery",

                "batting_team",

                "batter",

                "bowler"

            ]

        )



        fig_probability.add_hline(

            y=50,

            line_dash="dash",

            annotation_text="50%"

        )



        fig_probability.update_layout(

            height=500,

            yaxis_range=[0, 100],

            xaxis_title="Delivery",

            yaxis_title="Winner Win Probability (%)"

        )



        st.plotly_chart(

            fig_probability,

            use_container_width=True

        )



        st.caption(

            "The model estimates the probability that the "

            "batting team eventually wins. The chart converts "

            "this into a consistent probability for the actual "

            "match winner."

        )



    else:



        st.warning(

            "Win probability data is unavailable."

        )





    # --------------------------------------------------------

    # TOP TURNING POINTS

    # --------------------------------------------------------



    st.divider()



    st.subheader("🔥 Major Turning Points")



    match_turning_points = turning_points[

        turning_points["match_id"] == selected_match

    ].copy()



    if len(match_turning_points) > 0:



        match_turning_points = (

            match_turning_points

            .sort_values(

                "turning_point_score",

                ascending=False

            )

        )



        display_columns = [

            "innings",

            "over",

            "delivery",

            "batter",

            "bowler",

            "event_type",

            "momentum_direction",

            "turning_point_score",

            "probability_change",

            "explanation"

        ]



        display_columns = [

            col

            for col in display_columns

            if col in match_turning_points.columns

        ]



        st.dataframe(

            match_turning_points[

                display_columns

            ].head(10),

            use_container_width=True,

            hide_index=True

        )



        # ----------------------------------------------------

        # MOST IMPORTANT TURNING POINT

        # ----------------------------------------------------



        biggest = (

            match_turning_points

            .sort_values(

                "turning_point_score",

                ascending=False

            )

            .iloc[0]

        )



        st.success(

            f"🔥 Biggest detected turning point: "

            f"{biggest.get('event_type', 'EVENT')} "

            f"at over "

            f"{biggest.get('over', '?')}."

            f"{biggest.get('delivery', '?')}"

        )



        if "explanation" in biggest.index:



            st.info(

                str(biggest["explanation"])

            )



    else:



        st.info(

            "No major turning points were found."

        )





    # --------------------------------------------------------

    # MATCH AUTOPSY

    # --------------------------------------------------------



    st.divider()



    st.subheader("🧠 Match Autopsy")



    autopsy_result = match_autopsy[

        match_autopsy["match_id"] == selected_match

    ].copy()



    if len(autopsy_result) > 0:



        row = autopsy_result.iloc[0]



        # Main explanation

        if "final_explanation" in row.index:



            st.info(

                str(row["final_explanation"])

            )



        # Summary metrics

        a1, a2, a3 = st.columns(3)



        with a1:



            if "winner_final_score" in row.index:



                st.metric(

                    "Winner Score",

                    int(row["winner_final_score"])

                )



        with a2:



            if "loser_final_score" in row.index:



                st.metric(

                    "Loser Score",

                    int(row["loser_final_score"])

                )



        with a3:



            if "largest_probability_swing" in row.index:



                st.metric(

                    "Largest Probability Swing",

                    f"{float(row['largest_probability_swing']):.1f} pts"

                )





        # Investigation details



        st.markdown("### 🔍 Investigation Findings")



        findings = []



        if "decisive_event" in row.index:



            findings.append(

                (

                    "🎯 Decisive Event",

                    row["decisive_event"]

                )

            )



        if "largest_turning_point" in row.index:



            findings.append(

                (

                    "🔥 Largest Turning Point",

                    row["largest_turning_point"]

                )

            )

        if "major_wicket" in row.index:



            major_wicket = row["major_wicket"]



            if pd.isna(major_wicket) or str(major_wicket).strip().lower() in ["nan", "none", ""]:

                major_wicket = "No major wicket detected"



            findings.append(

                (

                    "🪓 Major Wicket",

                    major_wicket

                )

            )



        if "strongest_phase" in row.index:



            findings.append(

                (

                    "📈 Strongest Phase",

                    row["strongest_phase"]

                )

            )



        if "weakest_phase" in row.index:



            findings.append(

                (

                    "📉 Weakest Phase",

                    row["weakest_phase"]

                )

            )



        if "pressure_phase" in row.index:



            findings.append(

                (

                    "⚠️ Pressure Phase",

                    row["pressure_phase"]

                )

            )



        for label, value in findings:



            st.write(

                f"**{label}:** {value}"

            )



    else:



        st.info(

            "Match autopsy information is unavailable."

        )





# ============================================================

# TAB 3 — WHAT-IF SIMULATOR

# ============================================================



with tab3:



    st.header("🔮 What-If Simulator")



    st.markdown(

        """

        Explore a hypothetical alternative to an important

        match event.



        **Important:** this is a model-based scenario simulation,

        not a claim of causal certainty.

        """

    )



    match_scenarios = what_if[

        what_if["match_id"] == selected_match

    ].copy()



    if len(match_scenarios) > 0:



        match_scenarios = (

            match_scenarios

            .sort_values(

                "probability_difference",

                key=lambda x: x.abs(),

                ascending=False

            )

        )



        scenario_labels = (

            match_scenarios

            .apply(

                lambda r:

                f"Over {r['over']}.{r['delivery']} — "

                f"{r['scenario']}",

                axis=1

            )

            .tolist()

        )



        selected_label = st.selectbox(

            "Choose an event to investigate:",

            scenario_labels

        )



        selected_index = (

            scenario_labels.index(

                selected_label

            )

        )



        scenario_row = (

            match_scenarios

            .iloc[selected_index]

        )





        # ----------------------------------------------------

        # Scenario metrics

        # ----------------------------------------------------



        s1, s2, s3 = st.columns(3)



        with s1:



            st.metric(

                "Actual Probability",

                f"{float(scenario_row['actual_win_probability']) * 100:.1f}%"

            )



        with s2:



            st.metric(

                "Hypothetical Probability",

                f"{float(scenario_row['hypothetical_win_probability']) * 100:.1f}%"

            )



        with s3:



            difference = (

                float(

                    scenario_row[

                        "probability_difference"

                    ]

                ) * 100

            )



            st.metric(

                "Scenario Difference",

                f"{difference:+.1f} pts"

            )





        st.divider()



        st.subheader(

            f"🧪 {scenario_row['scenario']}"

        )



        st.write(

            f"**Team batting:** "

            f"{scenario_row['batting_team']}"

        )



        st.write(

            f"**Batter:** "

            f"{scenario_row['batter']}"

        )



        st.write(

            f"**Bowler:** "

            f"{scenario_row['bowler']}"

        )



        st.write(

            f"**Original event:** "

            f"{scenario_row['event_type']}"

        )



        st.info(

            str(

                scenario_row["interpretation"]

            )

        )





        # ----------------------------------------------------

        # Scenario comparison chart

        # ----------------------------------------------------



        comparison_df = pd.DataFrame(

            {

                "Scenario": [

                    "Actual",

                    "Hypothetical"

                ],

                "Win Probability": [

                    float(

                        scenario_row[

                            "actual_win_probability"

                        ]

                    ) * 100,

                    float(

                        scenario_row[

                            "hypothetical_win_probability"

                        ]

                    ) * 100

                ]

            }

        )



        fig_scenario = px.bar(

            comparison_df,

            x="Scenario",

            y="Win Probability",

            text="Win Probability",

            title="Actual vs Hypothetical Win Probability"

        )



        fig_scenario.update_traces(

            texttemplate="%{text:.1f}%",

            textposition="outside"

        )



        fig_scenario.update_layout(

            yaxis_range=[0, 100],

            yaxis_title="Win Probability (%)"

        )



        st.plotly_chart(

            fig_scenario,

            use_container_width=True

        )



    else:



        st.info(

            "No what-if scenarios are available for this match."

        )





# ============================================================

# TAB 4 — EVENT TIMELINE

# ============================================================



with tab4:



    st.header("⏱️ Match Event Timeline")



    timeline = match_data.copy()



    if len(timeline) > 0:



        # ----------------------------------------------------

        # Event classification

        # ----------------------------------------------------



        timeline["Event"] = "NORMAL"



        if "runs_batter" in timeline.columns:



            timeline.loc[

                timeline["runs_batter"] == 4,

                "Event"

            ] = "FOUR"



            timeline.loc[

                timeline["runs_batter"] == 6,

                "Event"

            ] = "SIX"



        if "runs_total" in timeline.columns:



            timeline.loc[

                timeline["runs_total"] == 0,

                "Event"

            ] = "DOT BALL"



        if "is_wicket" in timeline.columns:



            timeline.loc[

                timeline["is_wicket"] == 1,

                "Event"

            ] = "WICKET"





        # ----------------------------------------------------

        # Filters

        # ----------------------------------------------------



        event_filter = st.multiselect(

            "Filter events",

            [

                "WICKET",

                "SIX",

                "FOUR",

                "DOT BALL",

                "NORMAL"

            ],

            default=[

                "WICKET",

                "SIX",

                "FOUR"

            ]

        )



        filtered_timeline = timeline[

            timeline["Event"].isin(

                event_filter

            )

        ].copy()





        # ----------------------------------------------------

        # Timeline columns

        # ----------------------------------------------------



        timeline_columns = [

            "innings",

            "over",

            "delivery",

            "batting_team",

            "batter",

            "bowler",

            "runs_batter",

            "runs_total",

            "is_wicket",

            "Event"

        ]



        timeline_columns = [

            col

            for col in timeline_columns

            if col in filtered_timeline.columns

        ]



        st.dataframe(

            filtered_timeline[

                timeline_columns

            ].tail(100),

            use_container_width=True,

            hide_index=True

        )



    else:



        st.info(

            "Timeline data is unavailable."

        )





# ============================================================

# SIDEBAR — MODEL INFORMATION

# ============================================================



st.sidebar.divider()



st.sidebar.subheader("🤖 ML Model")



st.sidebar.write(

    "Random Forest Classifier"

)



if metadata:



    metrics = metadata.get(

        "metrics",

        {}

    )



    if "accuracy" in metrics:



        st.sidebar.metric(

            "Test Accuracy",

            f"{metrics['accuracy'] * 100:.2f}%"

        )



    if "roc_auc" in metrics:



        st.sidebar.metric(

            "ROC-AUC",

            f"{metrics['roc_auc']:.3f}"

        )



    st.sidebar.caption(

        f"Features used: "

        f"{len(metadata.get('features', []))}"

    )





# ============================================================

# SIDEBAR — DATASET INFORMATION

# ============================================================



st.sidebar.divider()



st.sidebar.subheader("📦 Dataset")



st.sidebar.write(

    f"Matches: {deliveries['match_id'].nunique():,}"

)



st.sidebar.write(

    f"Deliveries: {len(deliveries):,}"

)



st.sidebar.write(

    f"Turning Points: {len(turning_points):,}"

)



st.sidebar.write(

    f"What-If Scenarios: {len(what_if):,}"

)





# ============================================================

# PROJECT SUMMARY

# ============================================================



st.divider()



st.header("🏆 About IPL Forensics")



st.markdown(

    """

    **IPL Forensics** is a Data Science and Machine Learning

    project designed to investigate IPL matches rather than

    simply predict player performance.



    The system combines:



    - 🗃️ Ball-by-ball IPL data

    - 🧹 Data preprocessing

    - ⚙️ Feature engineering

    - 🤖 Random Forest Machine Learning

    - 🎯 Win probability estimation

    - 🔥 Turning-point detection

    - 🕵️ Match autopsy

    - 🔮 What-if scenario simulation

    - 📊 Interactive Streamlit visualization



    ### Investigation Question



    > **What happened, when did the match change, why did it

    > change, and what might have happened under a different

    > scenario?**

    """

)



st.success(

    "Investigation complete. Select another match from the "

    "sidebar to investigate."

)



st.caption(

    "IPL Forensics • Data Science Project • "

    "Ball-by-Ball Match Investigation"

)
# ============================================================
# NEW / CURRENT MATCH PREDICTION
# ============================================================

import joblib

st.markdown("---")
st.header("🔮 New / Current Match Prediction")

st.write(
    "Enter the current match situation to estimate the "
    "batting team's win probability."
)

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    batting_team = st.text_input(
        "Batting Team",
        "RCB"
    )

    opponent_team = st.text_input(
        "Opponent Team",
        "CSK"
    )

    current_score = st.number_input(
        "Current Score",
        min_value=0,
        value=85
    )

    wickets_lost = st.number_input(
        "Wickets Lost",
        min_value=0,
        max_value=10,
        value=2
    )

with col2:

    completed_overs = st.number_input(
        "Completed Overs",
        min_value=0,
        max_value=20,
        value=10
    )

    balls_in_current_over = st.number_input(
        "Balls in Current Over",
        min_value=0,
        max_value=5,
        value=3
    )

    target_score = st.number_input(
        "Target Score",
        min_value=1,
        value=180
    )

# ------------------------------------------------------------
# RECENT PERFORMANCE
# ------------------------------------------------------------

runs_last_6 = st.number_input(
    "Runs in Last 6 Balls",
    min_value=0,
    value=8
)

wickets_last_6 = st.number_input(
    "Wickets in Last 6 Balls",
    min_value=0,
    max_value=10,
    value=0
)

fours_last_6 = st.number_input(
    "Fours in Last 6 Balls",
    min_value=0,
    value=1
)

sixes_last_6 = st.number_input(
    "Sixes in Last 6 Balls",
    min_value=0,
    value=0
)

# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

if st.button(
    "🔮 Predict Win Probability",
    type="primary"
):

    # --------------------------------------------------------
    # BALL CALCULATION
    # --------------------------------------------------------

    balls_before = (
        completed_overs * 6
        + balls_in_current_over
    )

    # --------------------------------------------------------
    # CURRENT RUN RATE
    # --------------------------------------------------------

    if balls_before > 0:

        current_run_rate_before = (
            current_score
            / (balls_before / 6)
        )

    else:

        current_run_rate_before = 0

    # --------------------------------------------------------
    # CHASE FEATURES
    # --------------------------------------------------------

    runs_required_before = max(
        target_score - current_score,
        0
    )

    balls_remaining = max(
        120 - balls_before,
        0
    )

    if balls_remaining > 0:

        required_run_rate = (
            runs_required_before
            / (balls_remaining / 6)
        )

    else:

        required_run_rate = 0

    # --------------------------------------------------------
    # LOAD TRAINED MODEL
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_DIR / "win_probability_model.pkl"
    )

    # --------------------------------------------------------
    # CREATE ALL POSSIBLE MODEL FEATURES
    #
    # This supports both the older and newer versions
    # of the saved model.
    # --------------------------------------------------------

    feature_values = {

        "score_before":
            current_score,

        "wickets_before":
            wickets_lost,

        "balls_before":
            balls_before,

        "current_run_rate_before":
            current_run_rate_before,

        "runs_last_6_before":
            runs_last_6,

        "wickets_last_6_before":
            wickets_last_6,

        "fours_last_6_before":
            fours_last_6,

        "sixes_last_6_before":
            sixes_last_6,

        "runs_required_before":
            runs_required_before,

        "balls_remaining":
            balls_remaining,

        "required_run_rate":
            required_run_rate,

        # Support for the older feature naming
        "score":
            current_score,

        "wickets_lost":
            wickets_lost,

        "balls_bowled":
            balls_before,

        "current_run_rate": current_run_rate_before
            

    }

    # --------------------------------------------------------
    # USE THE ACTUAL FEATURES STORED IN THE MODEL
    # --------------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        model_features = list(
            model.feature_names_in_
        )

    else:

        st.error(
            "The saved model does not contain its "
            "feature names."
        )

        st.stop()

    # --------------------------------------------------------
    # CHECK THAT ALL REQUIRED FEATURES ARE AVAILABLE
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in model_features
        if feature not in feature_values
    ]

    if missing_features:

        st.error(
            "Missing model features: "
            + ", ".join(missing_features)
        )

        st.stop()

    # --------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------

    prediction_input = pd.DataFrame([

        {
            feature:
                feature_values[feature]
            for feature in model_features
        }

    ])

    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    probability = model.predict_proba(
        prediction_input
    )[0][1]

    # Keep probability between 0 and 1

    probability = max(
        0,
        min(1, probability)
    )

    batting_probability = (
        probability * 100
    )

    opponent_probability = (
        100 - batting_probability
    )

  # --------------------------------------------------------
# PREDICTION RESULT
# --------------------------------------------------------

st.markdown("---")
st.markdown("## 📊 Prediction Result")

# Main probabilities
result1, result2 = st.columns(2)

with result1:
    st.metric(
        f"🏏 {batting_team} Win Probability",
        f"{batting_probability:.1f}%"
    )

with result2:
    st.metric(
        f"🏆 {opponent_team} Win Probability",
        f"{opponent_probability:.1f}%"
    )

# Probability bar
st.progress(float(probability))

# --------------------------------------------------------
# MATCH SITUATION
# --------------------------------------------------------

st.markdown("### 🏏 Current Match Situation")

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric(
        "Current Score",
        f"{current_score}/{wickets_lost}"
    )

with s2:
    st.metric(
        "Target",
        target_score
    )

with s3:
    st.metric(
        "Runs Required",
        runs_required_before
    )

with s4:
    st.metric(
        "Balls Remaining",
        balls_remaining
    )

# --------------------------------------------------------
# REQUIRED RUN RATE
# --------------------------------------------------------

st.markdown("### 🎯 Chase Analysis")

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "Current Run Rate",
        f"{current_run_rate_before:.2f}"
    )

with c2:
    st.metric(
        "Required Run Rate",
        f"{required_run_rate:.2f}"
    )

# --------------------------------------------------------
# INTERPRETATION
# --------------------------------------------------------

st.markdown("### 🧠 ML Interpretation")

if batting_probability >= 70:

    st.success(
        f"🔥 {batting_team} is strongly favored according "
        f"to the current match situation."
    )

elif batting_probability >= 55:

    st.info(
        f"📈 {batting_team} currently has a slight advantage."
    )

elif batting_probability >= 45:

    st.warning(
        "⚖️ The match is currently very closely balanced."
    )

else:

    st.warning(
        f"📉 {opponent_team} currently has the advantage."
    )

# --------------------------------------------------------
# RECENT FORM
# --------------------------------------------------------

st.markdown("### ⚡ Recent 6-Ball Performance")

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.metric(
        "Runs",
        runs_last_6
    )

with r2:
    st.metric(
        "Wickets",
        wickets_last_6
    )

with r3:
    st.metric(
        "Fours",
        fours_last_6
    )

with r4:
    st.metric(
        "Sixes",
        sixes_last_6
    )

# --------------------------------------------------------
# DISCLAIMER
# --------------------------------------------------------

st.markdown("---")

st.caption(
    "🤖 Prediction generated using the IPL FORENSICS "
    "Machine Learning model."
)

st.caption(
    "⚠️ This is an estimated in-match probability, "
    "not a guarantee of the final result."
)