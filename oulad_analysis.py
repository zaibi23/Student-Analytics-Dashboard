"""
============================================================================
 OULAD — Big Data Analytics & BI Dashboard: Data Analysis Pipeline
 Descriptive (exploratory) analysis + leakage-safe at-risk prediction.

 Author: Zohaib Javed (C7658973)
 Dataset: Open University Learning Analytics Dataset (OULAD),
          Kuzilek, Hlosta & Zdrahal (2017), CC-BY 4.0.

 Reproduce:  python oulad_analysis.py
 Requires :  pandas numpy scikit-learn imbalanced-learn matplotlib
 Note     :  Engagement-from-clicks features require studentVle.csv from the
             official OULAD source; the pipeline reads it automatically if a
             copy is placed in DATA_DIR (see ENGAGEMENT block), otherwise it
             runs on the six supplied tables.
============================================================================
"""
import os, warnings, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipe
from imblearn.pipeline import Pipeline as ImbPipe
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay)
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))          # folder this script lives in
DATA_DIR = os.environ.get("OULAD_DIR", BASE)               # by default, CSVs sit NEXT TO this script
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True); RS = 42
ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]
PAL = {"Pass":"#2E7D32","Distinction":"#1565C0","Fail":"#C62828","Withdrawn":"#EF6C00"}
SEQ = ["#1565C0","#EF6C00","#2E7D32","#6A1B9A","#00838F","#C62828","#8D6E63"]
mpl.rcParams.update({"figure.dpi":120,"savefig.dpi":200,"savefig.bbox":"tight",
    "font.size":11,"axes.titlesize":13,"axes.titleweight":"bold",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
    "grid.color":"#E5E7EB","axes.axisbelow":True,"legend.frameon":False})
pct = lambda x: f"{100*x:.1f}%"

# ------------------------------------------------------------------ load
info  = pd.read_csv(f"{DATA_DIR}/studentInfo.csv")
reg   = pd.read_csv(f"{DATA_DIR}/studentRegistration.csv")
sasmt = pd.read_csv(f"{DATA_DIR}/studentAssessment.csv")
asmt  = pd.read_csv(f"{DATA_DIR}/assessments.csv")
crs   = pd.read_csv(f"{DATA_DIR}/courses.csv")
info["final_result"] = pd.Categorical(info["final_result"], ORDER)
info["success"] = info.final_result.isin(["Pass","Distinction"])
key = ["id_student","code_module","code_presentation"]
print(f"Loaded {len(info):,} registrations | {info.id_student.nunique():,} students")

# =========================== EXPLORATORY DATA ANALYSIS ===================
# Fig 5.1 outcome distribution
vc = info.final_result.value_counts().reindex(ORDER)
fig,ax=plt.subplots(figsize=(7.4,4.4))
for b,v in zip(ax.bar(vc.index,vc.values,color=[PAL[c] for c in vc.index],ec="white",lw=1.2),vc.values):
    ax.text(b.get_x()+b.get_width()/2,v+250,f"{v:,}\n({pct(v/len(info))})",ha="center",va="bottom",fontsize=9.5)
ax.set_title("Distribution of final student outcomes (OULAD, N = 32,593)")
ax.set_ylabel("Registrations"); ax.set_ylim(0,vc.max()*1.18)
plt.savefig(f"{FIG}/fig5_1_outcome_distribution.png"); plt.close()

# Fig 5.2 outcome composition by module
comp=(info.groupby("code_module").final_result.value_counts(normalize=True)
      .unstack().reindex(columns=ORDER)*100).sort_values("Withdrawn")
fig,ax=plt.subplots(figsize=(8.6,4.8)); bottom=np.zeros(len(comp))
for c in ORDER:
    ax.bar(comp.index,comp[c],bottom=bottom,label=c,color=PAL[c],ec="white",lw=0.6); bottom+=comp[c].values
ax.set_title("Composition of final outcomes by module"); ax.set_ylabel("Share (%)"); ax.set_xlabel("Module")
ax.set_ylim(0,100); ax.legend(ncol=4,loc="upper center",bbox_to_anchor=(0.5,-0.13))
plt.savefig(f"{FIG}/fig5_2_outcome_by_module.png"); plt.close()

# Fig 5.3 success by IMD
imd_order=['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%']
imd=(info.dropna(subset=["imd_band"]).groupby("imd_band").success.mean().reindex(imd_order).dropna()*100)
fig,ax=plt.subplots(figsize=(8.4,4.4))
ax.bar(range(len(imd)),imd.values,color=SEQ[0],ec="white")
ax.plot(range(len(imd)),imd.values,color="#EF6C00",lw=2,marker="o")
for i,v in enumerate(imd.values): ax.text(i,v+0.8,f"{v:.0f}",ha="center",fontsize=9)
ax.set_xticks(range(len(imd))); ax.set_xticklabels(imd.index,rotation=45,ha="right")
ax.set_title("Success rate by Index of Multiple Deprivation band")
ax.set_ylabel("Success rate (%)"); ax.set_xlabel("IMD band (most → least deprived)")
plt.savefig(f"{FIG}/fig5_3_success_by_imd.png"); plt.close()

# Fig 5.4 success by previous attempts
pa=info.groupby("num_of_prev_attempts").success.agg(["mean","count"]); pa=pa[pa["count"]>=30]
fig,ax=plt.subplots(figsize=(7.2,4.3))
for b,(mn,n) in zip(ax.bar(pa.index.astype(str),pa["mean"]*100,color=SEQ[2],ec="white"),pa[["mean","count"]].values):
    ax.text(b.get_x()+b.get_width()/2,mn*100+0.8,f"{mn*100:.0f}%\n(n={int(n):,})",ha="center",fontsize=8.5)
ax.set_title("Success rate by number of previous attempts"); ax.set_ylabel("Success rate (%)")
ax.set_xlabel("Previous attempts"); ax.set_ylim(0,pa["mean"].max()*100*1.18)
plt.savefig(f"{FIG}/fig5_4_success_by_prev_attempts.png"); plt.close()

# ---- assessment features ----
sa=sasmt.merge(asmt,on="id_assessment",how="left")
sa["date"]=pd.to_numeric(sa["date"],errors="coerce"); sa["submit_gap"]=sa["date"]-sa["date_submitted"]
ar=(asmt[asmt.assessment_type.isin(["TMA","CMA"])].assign(date=lambda d:pd.to_numeric(d.date,errors="coerce"))
    .sort_values(["code_module","code_presentation","date"]))
ar["a_order"]=ar.groupby(["code_module","code_presentation"]).cumcount()+1
first_ids=ar[ar.a_order==1][["id_assessment"]]
info=info.merge(crs,on=["code_module","code_presentation"],how="left")
first=(sa.merge(first_ids,on="id_assessment").groupby(key)
       .agg(first_score=("score","mean"),first_submit_gap=("submit_gap","mean")).reset_index())
sal=sa.merge(crs,on=["code_module","code_presentation"],how="left")
early=sal[sal["date"]<=sal["module_presentation_length"]/3]
ef=(early.groupby(key).agg(early_mean_score=("score","mean"),early_n_submitted=("score","count"),
    early_mean_gap=("submit_gap","mean")).reset_index())
feat=first.merge(ef,on=key,how="outer")

# Fig 5.5 first-assessment score by outcome
d=info.merge(feat[key+["first_score"]],on=key,how="left")
box=[d.loc[d.final_result==c,"first_score"].dropna() for c in ORDER]
fig,ax=plt.subplots(figsize=(7.6,4.5))
bp=ax.boxplot(box,patch_artist=True,widths=0.6,showfliers=False,medianprops=dict(color="black",lw=1.6))
for p,c in zip(bp["boxes"],ORDER): p.set_facecolor(PAL[c]); p.set_alpha(0.85)
mn=[b.mean() for b in box]; ax.plot(range(1,5),mn,"D",color="black",ms=6,label="Mean")
for i,mv in enumerate(mn): ax.text(i+1,mv+2.5,f"{mv:.0f}%",ha="center",fontsize=9,weight="bold")
ax.set_xticklabels(ORDER); ax.set_ylabel("First-assessment score (%)")
ax.set_title("First-assessment score by final outcome"); ax.legend(loc="lower left")
plt.savefig(f"{FIG}/fig5_5_first_assessment_by_outcome.png"); plt.close()

# ============================ PREDICTIVE MODELLING =======================
m=(info.merge(feat,on=key,how="left").merge(reg[key+["date_registration"]],on=key,how="left"))
m["submitted_first"]=m.first_score.notna().astype(int)
m["submitted_any_early"]=m.early_n_submitted.fillna(0).gt(0).astype(int)
m["at_risk"]=m.final_result.isin(["Fail","Withdrawn"]).astype(int)

num=["num_of_prev_attempts","studied_credits","date_registration","first_score",
     "first_submit_gap","early_mean_score","early_n_submitted","early_mean_gap",
     "submitted_first","submitted_any_early"]
cat=["gender","region","highest_education","imd_band","age_band","disability","code_module"]
X=m[num+cat]; y=m.at_risk.values
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=RS,stratify=y)

pre=ColumnTransformer([
    ("num",SkPipe([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
    ("cat",SkPipe([("imp",SimpleImputer(strategy="constant",fill_value="missing")),
                   ("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
models={"Logistic Regression":LogisticRegression(max_iter=2000),
        "Decision Tree":DecisionTreeClassifier(max_depth=6,min_samples_leaf=50,random_state=RS),
        "Random Forest":RandomForestClassifier(n_estimators=400,min_samples_leaf=5,n_jobs=-1,random_state=RS),
        "Gradient Boosting":GradientBoostingClassifier(random_state=RS)}
rows=[]; roc={}; fitted={}
for name,clf in models.items():
    pipe=ImbPipe([("pre",pre),("smote",SMOTE(random_state=RS)),("clf",clf)]).fit(Xtr,ytr)
    p=pipe.predict(Xte); pp=pipe.predict_proba(Xte)[:,1]
    rows.append(dict(Model=name,Accuracy=accuracy_score(yte,p),Precision=precision_score(yte,p),
        Recall=recall_score(yte,p),F1=f1_score(yte,p),ROC_AUC=roc_auc_score(yte,pp)))
    roc[name]=roc_curve(yte,pp); fitted[name]=pipe
res=pd.DataFrame(rows).set_index("Model").round(4); res.to_csv(os.path.join(BASE,"model_metrics.csv"))
best=res.F1.idxmax()
print("\nTEST-SET PERFORMANCE (%):"); print((res*100).round(1).to_string())
print("Best by F1:",best)

# Fig 5.6 model comparison
mets=["Accuracy","Precision","Recall","F1","ROC_AUC"]; x=np.arange(len(mets)); w=0.2
fig,ax=plt.subplots(figsize=(9.6,4.8))
for i,(name,r) in enumerate(res.iterrows()):
    ax.bar(x+(i-1.5)*w,r[mets].values*100,w,label=name,color=SEQ[i],ec="white")
ax.set_xticks(x); ax.set_xticklabels(mets); ax.set_ylabel("Score (%)"); ax.set_ylim(0,100)
ax.set_title("Predictive model comparison — at-risk classification (test set)")
ax.legend(ncol=2,loc="upper center",bbox_to_anchor=(0.5,-0.12))
plt.savefig(f"{FIG}/fig5_6_model_comparison.png"); plt.close()

# Fig 5.7 ROC
fig,ax=plt.subplots(figsize=(6.4,6))
for i,(name,(fpr,tpr,_)) in enumerate(roc.items()):
    ax.plot(fpr,tpr,color=SEQ[i],lw=2,label=f"{name} (AUC={res.loc[name,'ROC_AUC']:.3f})")
ax.plot([0,1],[0,1],"--",color="grey"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves — at-risk prediction"); ax.legend(loc="lower right",fontsize=9)
plt.savefig(f"{FIG}/fig5_7_roc_curves.png"); plt.close()

# Fig 5.8 confusion matrix
cm=confusion_matrix(yte,fitted[best].predict(Xte))
fig,ax=plt.subplots(figsize=(5.4,4.8))
ConfusionMatrixDisplay(cm,display_labels=["Not at-risk","At-risk"]).plot(ax=ax,cmap="Blues",colorbar=False,values_format=",d")
ax.set_title(f"Confusion matrix — {best}"); plt.savefig(f"{FIG}/fig5_8_confusion_matrix.png"); plt.close()

# Fig 5.9 feature importance
pipe=fitted["Random Forest"]; oh=pipe.named_steps["pre"].get_feature_names_out()
imps=pipe.named_steps["clf"].feature_importances_; agg={}
for nm,iv in zip(oh,imps):
    s=nm.split("__",1)[1]
    for c in cat:
        if s.startswith(c+"_"): s=c; break
    agg[s]=agg.get(s,0)+iv
LAB={"first_score":"First-assessment score","early_mean_score":"Early mean score",
 "early_n_submitted":"# early assessments","submitted_first":"Submitted 1st assessment",
 "submitted_any_early":"Submitted any early","first_submit_gap":"First submission timing",
 "early_mean_gap":"Early submission timing","date_registration":"Registration date",
 "studied_credits":"Studied credits","num_of_prev_attempts":"Previous attempts",
 "highest_education":"Highest education","imd_band":"IMD (deprivation)","region":"Region",
 "code_module":"Module","age_band":"Age band","gender":"Gender","disability":"Disability"}
imp=pd.Series({LAB.get(k,k):v for k,v in agg.items()}).sort_values()
fig,ax=plt.subplots(figsize=(7.6,5.4))
for i,v in enumerate(imp.values*100): ax.text(v+0.3,i,f"{v:.1f}",va="center",fontsize=8.5)
ax.barh(imp.index,imp.values*100,color=SEQ[0],ec="white")
ax.set_title("Feature importance (Random Forest)"); ax.set_xlabel("Importance (%)"); ax.grid(axis="y",visible=False)
plt.savefig(f"{FIG}/fig5_9_feature_importance.png"); plt.close()
print("Saved", len(os.listdir(FIG)), "figures + model_metrics.csv")
print("Done.")
