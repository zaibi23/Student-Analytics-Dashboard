import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipe
from imblearn.pipeline import Pipeline as ImbPipe
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("OULAD_DIR", BASE)
ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]
PAL = {"Pass":"#2E7D32","Distinction":"#1565C0","Fail":"#C62828","Withdrawn":"#EF6C00"}
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.color":"#E5E7EB","axes.axisbelow":True,
    "font.size":10,"axes.titleweight":"bold","legend.frameon":False})

# ----------------------------------------------------------------------------
#  ENGINE  (plain functions so they can be tested without the Streamlit runtime)
# ----------------------------------------------------------------------------
def _load_tables(base):
    need = ["studentInfo","studentRegistration","studentAssessment",
            "assessments","courses","vle"]
    return {n: pd.read_csv(os.path.join(base, n + ".csv")) for n in need}

def _build_frames(t):
    info = t["studentInfo"].copy()
    reg, sasmt, asmt, crs = (t["studentRegistration"], t["studentAssessment"],
                             t["assessments"], t["courses"])
    info["final_result"] = pd.Categorical(info["final_result"], ORDER)
    info["success"] = info.final_result.isin(["Pass","Distinction"])
    key = ["id_student","code_module","code_presentation"]
    sa = sasmt.merge(asmt, on="id_assessment", how="left")
    sa["date"] = pd.to_numeric(sa["date"], errors="coerce")
    sa["submit_gap"] = sa["date"] - sa["date_submitted"]
    ar = (asmt[asmt.assessment_type.isin(["TMA","CMA"])]
          .assign(date=lambda d: pd.to_numeric(d.date, errors="coerce"))
          .sort_values(["code_module","code_presentation","date"]))
    ar["a_order"] = ar.groupby(["code_module","code_presentation"]).cumcount()+1
    first_ids = ar[ar.a_order==1][["id_assessment"]]
    info = info.merge(crs, on=["code_module","code_presentation"], how="left")
    first = (sa.merge(first_ids, on="id_assessment").groupby(key)
             .agg(first_score=("score","mean"),
                  first_submit_gap=("submit_gap","mean")).reset_index())
    sal = sa.merge(crs, on=["code_module","code_presentation"], how="left")
    early = sal[sal["date"] <= sal["module_presentation_length"]/3]
    ef = (early.groupby(key).agg(early_mean_score=("score","mean"),
          early_n_submitted=("score","count"),
          early_mean_gap=("submit_gap","mean")).reset_index())
    feat = first.merge(ef, on=key, how="outer")
    m = (info.merge(feat, on=key, how="left")
             .merge(reg[key+["date_registration"]], on=key, how="left"))
    m["submitted_first"] = m.first_score.notna().astype(int)
    m["submitted_any_early"] = m.early_n_submitted.fillna(0).gt(0).astype(int)
    m["at_risk"] = m.final_result.isin(["Fail","Withdrawn"]).astype(int)
    num = ["num_of_prev_attempts","studied_credits","date_registration","first_score",
           "first_submit_gap","early_mean_score","early_n_submitted","early_mean_gap",
           "submitted_first","submitted_any_early"]
    cat = ["gender","region","highest_education","imd_band","age_band","disability","code_module"]
    return info, m, num, cat

def _train(m, num, cat):
    """Same student-grouped holdout and model settings as validate_grouped_model.py."""
    from validate_grouped_model import make_pipeline
    X, y = m[num + cat], m.at_risk.astype(int)
    groups = m["id_student"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(splitter.split(X, y, groups))
    assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[te]))
    pipe = make_pipeline(num, cat)
    pipe.fit(X.iloc[tr], y.iloc[tr])
    pred = pipe.predict(X.iloc[te])
    probs = pipe.predict_proba(X.iloc[te])[:, 1]
    metrics = {"Accuracy":accuracy_score(y.iloc[te],pred),
               "Precision":precision_score(y.iloc[te],pred,zero_division=0),
               "Recall":recall_score(y.iloc[te],pred,zero_division=0),
               "F1":f1_score(y.iloc[te],pred,zero_division=0),
               "ROC-AUC":roc_auc_score(y.iloc[te],probs)}
    return pipe, metrics, confusion_matrix(y.iloc[te], pred, labels=[0, 1])


def _defaults(m, num, cat):
    d = {c: float(m[c].median()) for c in num}
    d.update({c: m[c].mode(dropna=True).iloc[0] for c in cat})
    return d

# ----------------------------------------------------------------------------
#  CHARTS
# ----------------------------------------------------------------------------
def chart_outcomes(df):
    vc = df.final_result.value_counts().reindex(ORDER)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(vc.index, vc.values, color=[PAL[c] for c in vc.index])
    ax.set_ylabel("Registrations"); ax.set_title("Final outcomes")
    for i,v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=15); return fig

def chart_by_module(df):
    comp = (df.groupby("code_module").final_result.value_counts(normalize=True)
            .unstack().reindex(columns=ORDER)*100).sort_values("Withdrawn")
    fig, ax = plt.subplots(figsize=(5,3)); bottom=np.zeros(len(comp))
    for c in ORDER:
        ax.bar(comp.index, comp[c], bottom=bottom, color=PAL[c]); bottom+=comp[c].values
    ax.set_ylabel("Share (%)"); ax.set_title("Outcome mix by module"); ax.set_ylim(0,100)
    return fig

def chart_imd(df):
    order=['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%']
    imd=(df.dropna(subset=["imd_band"]).groupby("imd_band").success.mean().reindex(order).dropna()*100)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(range(len(imd)), imd.values, "-o", color="#1565C0")
    ax.set_xticks(range(len(imd))); ax.set_xticklabels(imd.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Success (%)"); ax.set_title("Success by deprivation band"); return fig

def chart_prev(df):
    pa=df.groupby("num_of_prev_attempts").success.agg(["mean","count"]); pa=pa[pa["count"]>=30]
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(pa.index.astype(str), pa["mean"]*100, color="#2E7D32")
    ax.set_ylabel("Success (%)"); ax.set_xlabel("Previous attempts"); ax.set_title("Success by prior attempts")
    return fig

# ----------------------------------------------------------------------------
#  UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="OULAD BI Dashboard", layout="wide")
    load_tables = st.cache_data(_load_tables)
    build_frames = st.cache_data(_build_frames)
    train = st.cache_resource(_train)

    st.title("📊 Higher-Education Analytics & At-Risk Dashboard")
    st.caption("Open University Learning Analytics Dataset (OULAD) · descriptive insight + predictive early-warning · prototype for academic demonstration only.")

    try:
        tables = load_tables(DATA_DIR)
    except FileNotFoundError:
        st.error("Could not find the OULAD CSV files. Put the 6 CSVs in the SAME folder as this dashboard, then refresh."); st.stop()
    info, m, num, cat = build_frames(tables)
    pipe, metrics, cm = train(m, num, cat)
    defaults = _defaults(m, num, cat)

    tab1, tab2, tab3, tab4 = st.tabs(["🏫 Overview","🎯 At-risk predictor","📈 Model performance","ℹ️ About"])

    # ---------------- OVERVIEW ----------------
    with tab1:
        st.sidebar.header("Filters")
        mods = st.sidebar.multiselect("Module", sorted(info.code_module.unique()), default=list(sorted(info.code_module.unique())))
        ages = st.sidebar.multiselect("Age band", sorted(info.age_band.dropna().unique()), default=list(sorted(info.age_band.dropna().unique())))
        gens = st.sidebar.multiselect("Gender", sorted(info.gender.dropna().unique()), default=list(sorted(info.gender.dropna().unique())))
        f = info[info.code_module.isin(mods) & info.age_band.isin(ages) & info.gender.isin(gens)]
        if len(f)==0:
            st.warning("No records match the current filters."); st.stop()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Registrations", f"{len(f):,}")
        c2.metric("Unique students", f"{f.id_student.nunique():,}")
        c3.metric("At-risk (Fail/Withdrawn)", f"{f.final_result.isin(['Fail','Withdrawn']).mean()*100:.1f}%")
        c4.metric("Withdrawn", f"{(f.final_result=='Withdrawn').mean()*100:.1f}%")
        st.divider()
        a,b = st.columns(2)
        a.pyplot(chart_outcomes(f)); b.pyplot(chart_by_module(f))
        c,d = st.columns(2)
        c.pyplot(chart_imd(f)); d.pyplot(chart_prev(f))
        plt.close("all")

    # ---------------- PREDICTOR ----------------
    with tab2:
        st.subheader("Predict a student's risk from early-course information")
        st.caption("Enter the details known early in a module. The model returns an at-risk probability. This is a demonstration indicator, **not** a real decision about a real student.")
        col1, col2, col3 = st.columns(3)
        with col1:
            module = st.selectbox("Module", sorted(info.code_module.unique()))
            prev = st.number_input("Previous attempts", 0, 6, 0)
            credits = st.number_input("Studied credits", 0, 655, 60, step=30)
        with col2:
            edu = st.selectbox("Highest education", sorted(info.highest_education.dropna().unique()))
            imd = st.selectbox("Deprivation band (IMD)", sorted(info.imd_band.dropna().unique()))
            age = st.selectbox("Age band", sorted(info.age_band.dropna().unique()))
        with col3:
            gender = st.selectbox("Gender", sorted(info.gender.dropna().unique()))
            disability = st.selectbox("Disability", sorted(info.disability.dropna().unique()))
            region = st.selectbox("Region", sorted(info.region.dropna().unique()))
        st.markdown("**Early assessment**")
        submitted = st.checkbox("Submitted the first assessment?", value=True)
        first_score = st.slider("First-assessment score (%)", 0, 100, 65, disabled=not submitted)

        if st.button("🔮 Predict risk", type="primary"):
            row = dict(defaults)
            row.update({"code_module":module,"num_of_prev_attempts":prev,"studied_credits":credits,
                        "highest_education":edu,"imd_band":imd,"age_band":age,
                        "gender":gender,"disability":disability,"region":region})
            if submitted:
                row.update({"first_score":first_score,"early_mean_score":first_score,
                            "early_n_submitted":1,"submitted_first":1,"submitted_any_early":1})
            else:
                row.update({"first_score":np.nan,"early_mean_score":np.nan,
                            "early_n_submitted":0,"submitted_first":0,"submitted_any_early":0})
            Xrow = pd.DataFrame([row])[num+cat]
            prob = float(pipe.predict_proba(Xrow)[0,1])
            st.divider()
            k1,k2 = st.columns([1,2])
            if prob >= 0.5:
                k1.error(f"### ⚠️ AT RISK\n**{prob*100:.0f}%** probability")
            else:
                k1.success(f"### ✅ Not at risk\n**{prob*100:.0f}%** risk probability")
            k2.progress(prob, text=f"At-risk probability: {prob*100:.0f}%")
            k2.caption("Interpretation: higher = more likely to Fail or Withdraw. This is an exploratory risk indicator, not an individual decision. Use it for demonstration, not to label students.")

    # ---------------- MODEL PERFORMANCE ----------------
    with tab3:
        st.subheader("Random Forest — student-grouped held-out test performance")
        st.caption("Train/test partition uses student IDs: the same student cannot appear in both sets. These results are distinct from the original random 75/25 split.")
        st.dataframe(pd.DataFrame({k:[f"{v*100:.1f}%"] for k,v in metrics.items()}), hide_index=True, use_container_width=True)
        st.markdown("**Confusion matrix (held-out student-grouped test)**")
        cmdf = pd.DataFrame(cm, index=["Actual: Not at-risk","Actual: At-risk"],
                            columns=["Predicted: Not","Predicted: At-risk"])
        st.dataframe(cmdf, use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        st.info(f"Of {tp+fn:,} genuinely at-risk registrations, the model correctly flags **{tp:,} ({tp/(tp+fn)*100:.1f}%)**; "
                f"of everyone it flags, **{tp/(tp+fp)*100:.1f}%** truly are at risk.")
        st.subheader("Five-fold student-grouped cross-validation")
        cv_path = os.path.join(BASE, "grouped_cv_folds.csv")
        if os.path.isfile(cv_path):
            cv_df = pd.read_csv(cv_path)
            expected = ["Fold", "accuracy", "precision", "recall", "f1", "roc_auc"]
            if all(col in cv_df.columns for col in expected):
                means = cv_df[expected[1:]].mean()
                sds = cv_df[expected[1:]].std()
                summary = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                    "Mean": [f"{means[c]*100:.2f}%" for c in expected[1:]],
                    "Standard deviation": [f"{sds[c]*100:.2f} pp" for c in expected[1:]],
                })
                st.dataframe(summary, hide_index=True, use_container_width=True)
                folds_display = cv_df.copy()
                for c in expected[1:]:
                    folds_display[c] = folds_display[c].map(lambda v: f"{v*100:.2f}%")
                st.markdown("**Individual validation folds**")
                st.dataframe(folds_display, hide_index=True, use_container_width=True)
                st.caption("Cross-validation was calculated offline on training students only, using StratifiedGroupKFold and SMOTE within each fold.")
            else:
                st.warning("grouped_cv_folds.csv exists but has unexpected columns. Rerun validate_grouped_model.py.")
        else:
            st.warning("Upload grouped_cv_folds.csv to the GitHub repository to display the five-fold results.")

    # ---------------- ABOUT ----------------
    with tab4:
        st.markdown("""
**Dataset.** Open University Learning Analytics Dataset (OULAD) — Kuzilek, Hlosta & Zdrahal (2017), CC-BY 4.0. 32,593 registrations across 7 modules.

**What this tool does.** Descriptive institutional reporting (outcomes, modules, deprivation, prior attempts) plus an interpretable at-risk predictor trained on demographic, registration and early-assessment features. Feature availability at the prediction date still requires evaluation.

**Ethics.** Secondary, anonymised, publicly-licensed data only. No primary data, no human participants. Predictive output is a prototype indicator for academic demonstration — not a real decision-making tool.

**Note.** VLE clickstream (studentVle.csv) engagement features are a documented extension; drop that file in the folder to include them.
        """)

if __name__ == "__main__":
    main()
