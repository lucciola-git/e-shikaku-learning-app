import streamlit as st
import pandas as pd
import os
import csv

st.title("E資格学習用 単語帳アプリケーション")

# データファイルのパス定義
DATA_PATH = "vocab_list.tsv"

@st.cache_data
def load_data(file_path):
    """
    TSVファイルを読み込みDataFrameとして返す。
    改行を含むフィールドを正しく処理するため、quotingパラメータを指定。
    """
    if not os.path.exists(file_path):
        return None
    try:
        # csv.QUOTE_MINIMAL を指定してダブルクォーテーション内の改行を保護
        return pd.read_csv(file_path, sep='\t', quoting=csv.QUOTE_MINIMAL)
    except Exception as e:
        st.error(f"データの読み込み中にエラーが発生しました: {e}")
        return None

df = load_data(DATA_PATH)

if df is None:
    st.error(f"データファイル '{DATA_PATH}' が見つかりません。ファイルを配置してください。")
else:
    st.header("３－１．単語帳一覧")
    
    # 単語リストを抽出
    words = df['word'].tolist()
    
    # セレクトボックスによる単語選択
    selected_word = st.selectbox(
        "定義を確認する単語を選択してください：",
        options=words,
        index=0
    )
    
    # 選択された単語に対応する説明文を抽出
    selected_row = df[df['word'] == selected_word]
    
    if not selected_row.empty:
        description = selected_row['description'].values[0]
        
        # 説明文の表示（数式表現を有効化するため markdown を使用）
        st.markdown("### 定義・説明")
        st.info(description)
    else:
        st.warning("選択された単語の説明文が見つかりませんでした。")