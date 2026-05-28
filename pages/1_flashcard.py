import streamlit as st
import pandas as pd
import os
import csv

st.title("E資格学習用 フラッシュカード")

DATA_PATH = "vocab_list.tsv"

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        return pd.read_csv(file_path, sep='\t', quoting=csv.QUOTE_MINIMAL)
    except Exception as e:
        st.error(f"データの読み込み中にエラーが発生しました: {e}")
        return None

df = load_data(DATA_PATH)

if df is None:
    st.error(f"データファイル '{DATA_PATH}' が見つかりません。")
else:
    st.header("３－２．フラッシュカード")

    # セッション状態の初期化
    if 'current_word' not in st.session_state:
        # 初回実行時にランダムで1つ選択
        random_row = df.sample(n=1).iloc[0]
        st.session_state.current_word = random_row['word']
        st.session_state.current_desc = random_row['description']
        st.session_state.show_answer = False

    # 「次の単語へ」ボタンが押された場合の処理
    if st.button("次の単語を変更（ランダム抽出）"):
        random_row = df.sample(n=1).iloc[0]
        st.session_state.current_word = random_row['word']
        st.session_state.current_desc = random_row['description']
        st.session_state.show_answer = False
        st.rerun()

    # カードの表示エリア
    st.markdown("---")
    st.subheader("問題（単語）")
    st.code(st.session_state.current_word, language="text")

    # 「答えを表示」ボタン
    if st.button("答えを表示する"):
        st.session_state.show_answer = True

    # 答えの表示エリア
    if st.session_state.show_answer:
        st.markdown("### 定義・説明")
        st.success(st.session_state.current_desc)
    st.markdown("---")
