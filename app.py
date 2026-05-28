import streamlit as st
import pandas as pd
import os

st.title("E資格学習用 単語帳アプリケーション")

# データファイルのパス定義
DATA_PATH = "vocab_list.tsv"

@st.cache_data
def load_data(file_path):
    """
    TSVファイルを読み込みDataFrameとして返す。
    """
    if not os.path.exists(file_path):
        return None
    # TSVのため区切り文字をタブに指定
    return pd.read_csv(file_path, sep='\t')

df = load_data(DATA_PATH)

if df is None:
    st.error(f"データファイル '{DATA_PATH}' が見つかりません。ファイルを配置してください。")
else:
    st.header("３－１．単語帳一覧")
    
    # 単語リストを抽出
    words = df['word'].tolist()
    
    # セレクトボックスによる単語選択（インデックス管理）
    selected_word = st.selectbox(
        "定義を確認する単語を選択してください：",
        options=words,
        index=0
    )
    
    # 選択された単語に対応する説明文を抽出
    description = df[df['word'] == selected_word]['description'].values[0]
    
    # 説明文の表示（数式表現を有効化するため markdown を使用）
    st.markdown("### 定義・説明")
    st.info(description)

