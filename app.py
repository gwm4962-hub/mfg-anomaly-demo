import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="製造ライン 異常検知デモ", page_icon="🔍", layout="wide")
st.title("🔍 製造ライン 異常検知デモ")
st.caption("Isolation Forest + SHAP による異常検知・説明可視化")


# ─────────────────────────── サンプルデータ生成 ────────────────────────────
@st.cache_data
def generate_sample() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    t = pd.date_range("2024-01-01", periods=n, freq="1min")
    temp = 70 + 5 * np.sin(np.linspace(0, 20, n)) + rng.normal(0, 1.5, n)
    vib  = 0.5 + rng.normal(0, 0.1, n)
    pres = 100 + 3 * np.cos(np.linspace(0, 10, n)) + rng.normal(0, 0.5, n)
    curr = 5.0 + rng.normal(0, 0.3, n)

    for i in [50, 120, 200, 350, 430]:
        temp[i] += rng.uniform(12, 20)
        vib[i]  += rng.uniform(1.0, 2.0)
        pres[i] -= rng.uniform(15, 25)
        curr[i] += rng.uniform(2.0, 4.0)

    return pd.DataFrame({
        "timestamp":   t,
        "temperature": np.round(temp, 2),
        "vibration":   np.round(vib, 3),
        "pressure":    np.round(pres, 2),
        "current":     np.round(curr, 2),
    })


# ─────────────────────────── サイドバー ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")

    uploaded = st.file_uploader("CSVをアップロード", type="csv",
                                help="1行目をヘッダーとして読み込みます")

    if st.button("📦 サンプルデータを使用", type="primary", use_container_width=True):
        st.session_state["df"] = generate_sample()
        st.session_state.pop("feat_cols", None)

    if uploaded:
        st.session_state["df"] = pd.read_csv(uploaded)
        st.session_state.pop("feat_cols", None)

    if "df" not in st.session_state:
        st.session_state["df"] = generate_sample()

    df_raw: pd.DataFrame = st.session_state["df"]
    numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()

    default_feats = st.session_state.get("feat_cols", numeric_cols[:4])
    feat_cols: list[str] = st.multiselect(
        "特徴量列", numeric_cols,
        default=[c for c in default_feats if c in numeric_cols],
    )
    st.session_state["feat_cols"] = feat_cols

    contamination: float = st.slider(
        "異常率の想定 (contamination)", 0.01, 0.20, 0.05, 0.01,
        help="全データ中の異常の割合の見積もり",
    )

    if not feat_cols:
        st.warning("特徴量列を1つ以上選択してください")
        st.stop()


# ─────────────────────────── 検知 ─────────────────────────────────────────
@st.cache_data
def run_detection(values: np.ndarray, cols: list[str], cont: float):
    X = pd.DataFrame(values, columns=cols)
    model = IsolationForest(contamination=cont, n_estimators=200, random_state=42)
    model.fit(X)
    labels = model.predict(X)       # 1=正常, -1=異常
    scores = model.decision_function(X)  # 高いほど正常
    return model, labels, scores


X_df = df_raw[feat_cols].dropna().reset_index(drop=True)
model, labels, scores = run_detection(X_df.values, feat_cols, contamination)

anomaly_mask = labels == -1
df_result = X_df.copy()
df_result["anomaly"] = anomaly_mask
df_result["score"]   = np.round(scores, 4)

# タイムスタンプ列があれば結合
time_col = next((c for c in df_raw.columns if "time" in c.lower() or "date" in c.lower()), None)
if time_col:
    df_result[time_col] = df_raw[time_col].iloc[:len(df_result)].values

n_anom = anomaly_mask.sum()

col_a, col_b, col_c = st.columns(3)
col_a.metric("総データ数",   f"{len(df_result):,} 行")
col_b.metric("検知された異常", f"{n_anom:,} 点",
             delta=f"{n_anom/len(df_result)*100:.1f}%", delta_color="inverse")
col_c.metric("使用特徴量数",  f"{len(feat_cols)} 列")


# ─────────────────────────── SHAP ─────────────────────────────────────────
@st.cache_data
def compute_shap(values: np.ndarray, cols: list[str], cont: float):
    X = pd.DataFrame(values, columns=cols)
    m = IsolationForest(contamination=cont, n_estimators=200, random_state=42)
    m.fit(X)
    expl = shap.TreeExplainer(m)
    sv   = expl.shap_values(X)
    ev = expl.expected_value
    return sv, float(np.squeeze(ev) if hasattr(ev, "__len__") else ev)


shap_values, expected_val = compute_shap(X_df.values, feat_cols, contamination)


# ─────────────────────────── タブ ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 時系列・異常点", "🌍 SHAP Global", "🔎 SHAP Local"])


# ── Tab 1: 時系列グラフ ───────────────────────────────────────────────────
with tab1:
    x_vals = df_result[time_col] if time_col else df_result.index

    for col in feat_cols:
        normal_idx = df_result.index[~df_result["anomaly"]]
        anom_idx   = df_result.index[df_result["anomaly"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals.iloc[normal_idx] if time_col else normal_idx,
            y=df_result.loc[normal_idx, col],
            mode="lines", name="正常",
            line=dict(color="#3B82F6", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=x_vals.iloc[anom_idx] if time_col else anom_idx,
            y=df_result.loc[anom_idx, col],
            mode="markers", name="異常",
            marker=dict(color="#EF4444", size=9, symbol="x", line=dict(width=2)),
        ))
        fig.update_layout(
            title=dict(text=col, font=dict(size=15)),
            height=260, margin=dict(t=45, b=20, l=40, r=10),
            legend=dict(orientation="h", y=1.15),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("異常点データ一覧"):
        st.dataframe(
            df_result[df_result["anomaly"]].sort_values("score").reset_index(),
            use_container_width=True,
        )


# ── Tab 2: SHAP Global ────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("特徴量の重要度 (Bar)")
        fig_bar, _ = plt.subplots(figsize=(6, 3.5))
        shap.summary_plot(shap_values, X_df, plot_type="bar", show=False,
                          color="#3B82F6")
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with c2:
        st.subheader("SHAP Beeswarm")
        fig_bee, _ = plt.subplots(figsize=(6, 3.5))
        shap.summary_plot(shap_values, X_df, show=False)
        plt.tight_layout()
        st.pyplot(fig_bee)
        plt.close(fig_bee)

    st.caption(
        "赤＝値が大きいほど異常スコアを下げる（正常方向）方向に寄与 / "
        "青＝値が小さいほど正常方向に寄与。横軸が大きいほど影響大。"
    )


# ── Tab 3: SHAP Local ─────────────────────────────────────────────────────
with tab3:
    anom_indices = df_result.index[df_result["anomaly"]].tolist()

    if not anom_indices:
        st.info("異常点が検知されませんでした。contamination を上げてみてください。")
    else:
        sel = st.selectbox(
            "調べる異常点を選択",
            anom_indices,
            format_func=lambda i: (
                f"Index {i}  |  score={df_result.loc[i, 'score']:.4f}"
                + (f"  |  {df_result.loc[i, time_col]}" if time_col else "")
            ),
        )

        explanation = shap.Explanation(
            values=shap_values[sel],
            base_values=expected_val,
            data=X_df.iloc[sel].values,
            feature_names=feat_cols,
        )

        col_w, col_d = st.columns([2, 1])

        with col_w:
            st.subheader("Waterfall Plot（この1点の要因分解）")
            fig_wf, _ = plt.subplots(figsize=(7, 4))
            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            st.pyplot(fig_wf)
            plt.close(fig_wf)

        with col_d:
            st.subheader("センサー値")
            row = df_result.loc[sel, feat_cols]
            for f in feat_cols:
                st.metric(f, f"{row[f]:.3f}")
            st.caption(f"anomaly score: **{df_result.loc[sel, 'score']:.4f}**")

        st.caption(
            "E[f(x)] = ベースライン（平均異常スコア）。各バーが各特徴量の寄与を示す。"
            "最終的な f(x) が異常スコア。"
        )
