import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("Credit Risk — Loan Default Prediction")
st.markdown("Built with LendingClub data · Random Forest model · AUC 0.70")

@st.cache_data
def load_data():
    return pd.read_csv("loans_clean.csv")


@st.cache_resource
def load_model():
    return joblib.load("model_credit_risk.pkl")

df = load_data()
model = load_model()

st.sidebar.header("Filters")
grades = sorted(df["grade"].dropna().unique())
selected_grades = st.sidebar.multiselect("Loan grade", grades, default=grades)
purposes = sorted(df["purpose"].dropna().unique())
selected_purpose = st.sidebar.multiselect("Loan purpose", purposes, default=purposes)

filtered = df[
    df["grade"].isin(selected_grades) &
    df["purpose"].isin(selected_purpose)
]

st.subheader("Portfolio overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total loans",       f"{len(filtered):,}")
col2.metric("Total defaults",    f"{filtered['is_default'].sum():,}")
col3.metric("Default rate",      f"{filtered['is_default'].mean():.1%}")
col4.metric("Avg interest rate", f"{filtered['int_rate'].mean():.1f}%")

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Default rate by grade")
    grade_df = (
        filtered.groupby("grade")["is_default"]
        .mean().reset_index()
        .rename(columns={"is_default": "default_rate"})
    )
    grade_df["default_rate_pct"] = grade_df["default_rate"] * 100
    fig1 = px.bar(grade_df, x="grade", y="default_rate_pct",
                  color="default_rate_pct", color_continuous_scale="Reds",
                  labels={"default_rate_pct": "Default rate (%)"})
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("Loan amount distribution")
    fig2 = px.histogram(filtered, x="loan_amnt", color="is_default",
                        nbins=40, barmode="overlay",
                        color_discrete_map={0: "steelblue", 1: "tomato"},
                        labels={"loan_amnt": "Loan amount", "is_default": "Defaulted"})
    st.plotly_chart(fig2, use_container_width=True)

col_c, col_d = st.columns(2)
with col_c:
    st.subheader("Default rate by purpose")
    purpose_df = (
        filtered.groupby("purpose")["is_default"]
        .mean().reset_index()
        .rename(columns={"is_default": "default_rate"})
        .sort_values("default_rate", ascending=False)
    )
    purpose_df["default_rate_pct"] = purpose_df["default_rate"] * 100
    fig3 = px.bar(purpose_df, x="default_rate_pct", y="purpose",
                  orientation="h", color="default_rate_pct",
                  color_continuous_scale="Oranges",
                  labels={"default_rate_pct": "Default rate (%)", "purpose": ""})
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.subheader("Interest rate vs DTI")
    fig4 = px.scatter(
        filtered.sample(min(2000, len(filtered))),
        x="dti", y="int_rate", color="is_default",
        color_discrete_map={0: "steelblue", 1: "tomato"},
        opacity=0.5,
        labels={"dti": "Debt-to-income ratio",
                "int_rate": "Interest rate (%)",
                "is_default": "Defaulted"}
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("Predict default risk for a new applicant")

c1, c2, c3 = st.columns(3)
with c1:
    loan_amnt      = st.number_input("Loan amount (USD)", 1000, 40000, 10000, step=500)
    int_rate       = st.slider("Interest rate (%)", 5.0, 30.0, 12.0, step=0.1)
    installment    = st.number_input("Monthly installment (USD)", 50, 1500, 300, step=10)
    annual_inc     = st.number_input("Annual income (USD)", 10000, 300000, 60000, step=1000)
with c2:
    dti            = st.slider("Debt-to-income ratio", 0.0, 40.0, 15.0, step=0.1)
    delinq_2yrs    = st.number_input("Delinquencies (last 2 yrs)", 0, 10, 0)
    open_acc       = st.number_input("Open credit accounts", 1, 40, 8)
    revol_bal      = st.number_input("Revolving balance (USD)", 0, 100000, 10000, step=500)
with c3:
    revol_util     = st.slider("Revolving utilization (%)", 0.0, 100.0, 50.0, step=1.0)
    total_acc      = st.number_input("Total accounts", 1, 80, 20)
    pub_rec        = st.number_input("Public records", 0, 10, 0)
    grade          = st.selectbox("Loan grade", ["A","B","C","D","E","F","G"])
    home_ownership = st.selectbox("Home ownership", ["RENT","OWN","MORTGAGE","OTHER"])
    purpose        = st.selectbox("Purpose", sorted(df["purpose"].dropna().unique()))
    term           = st.selectbox("Term", ["36 months","60 months"])

if st.button("Predict default risk", type="primary"):
    input_df = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "installment": installment,
        "annual_inc": annual_inc,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "open_acc": open_acc,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "pub_rec": pub_rec,
        "grade": grade,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "term": term
    }])

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.markdown("---")
    if pred == 1:
        st.error(f"High default risk — probability: {prob:.1%}")
    else:
        st.success(f"Low default risk — probability: {prob:.1%}")
    st.progress(float(prob), text=f"Default probability: {prob:.1%}")