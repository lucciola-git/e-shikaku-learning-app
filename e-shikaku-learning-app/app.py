import streamlit as st
import pandas as pd
import random

# ページ設定（スマホ閲覧を意識したワイドモードとタイトル）
st.set_page_config(page_title="E資格 学習アプリ", layout="centered")

# データの読み込み
@st.cache_data
def load_data():
    # data/current_vocabulary.csv から単語帳データを読み込む
    try:
        df = pd.read_csv("data/current_vocabulary.csv")
        # 欠損値対策
        df.fillna("", inplace=True)
        return df
    except FileNotFoundError:
        # データが存在しない場合のダミーデータ
        return pd.DataFrame(columns=["category", "word", "description"])

df = load_data()

st.title("📚 E資格 学習支援サイト")

# タブ機能による「単語帳一覧」と「ランダム出題」の分離
tab1, tab2 = st.tabs(["🗂 単語帳一覧", "🎲 ランダム出題"])

# --- タブ1: 単語帳一覧 ---
with tab1:
    if df.empty:
        st.warning("単語データ（data/current_vocabulary.csv）が見つかりません。")
    else:
        # カテゴリの一覧を取得
        categories = df["category"].unique()
        selected_category = st.selectbox("章（カテゴリ）を選択", categories)
        
        # 選択されたカテゴリの単語をフィルタリング
        filtered_df = df[df["category"] == selected_category]
        
        st.write(f"### 【{selected_category}】の単語一覧")
        
        # 各単語をアコーディオン形式（expander）で表示し、詳細を参照可能にする
        for _, row in filtered_df.iterrows():
            with st.expander(row["word"]):
                st.markdown(row["description"])

# --- タブ2: ランダム出題（フラッシュカード機能） ---
with tab2:
    if df.empty:
        st.warning("出題する単語データがありません。")
    else:
        # セッション状態の初期化
        if "current_question" not in st.session_state:
            st.session_state.current_question = df.iloc[random.randint(0, len(df) - 1)]
            st.session_state.show_answer = False

        q = st.session_state.current_question

        st.write("### 問題")
        st.info(f"**分類: {q['category']}**")
        st.subheader(f"「 {q['word']} 」 の定義・意味を説明せよ。")

        # 「答えを表示」ボタン
        if st.button("答えを表示"):
            st.session_state.show_answer = True

        # 答えの描画
        if st.session_state.show_answer:
            st.write("---")
            st.write("### 解答・解説")
            st.markdown(q["description"])

        # 「次の問題へ」ボタン
        if st.button("次の問題へ"):
            st.session_state.current_question = df.iloc[random.randint(0, len(df) - 1)]
            st.session_state.show_answer = False
            st.rerun()
