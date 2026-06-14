import os
import glob

def merge_markdown_files(input_dir, output_filepath):
    """
    指定されたディレクトリ内のすべての.mdファイルを取得し、
    1つのファイルに結合して出力する。
    """
    # 指定フォルダ内の.mdファイルを検索
    search_path = os.path.join(input_dir, "*.md")
    md_files = sorted(glob.glob(search_path))
    
    if not md_files:
        print(f"指定されたディレクトリにMarkdownファイルが見つかりません: {input_dir}")
        return

    print(f"{len(md_files)} 個のファイルを結合します...")

    with open(output_filepath, "w", encoding="utf-8") as outfile:
        for filepath in md_files:
            filename = os.path.basename(filepath)
            
            # 各ファイルの境界が明確になるようヘッダーを挿入
            outfile.write(f"\n\n\n")
            
            with open(filepath, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                
            outfile.write(f"\n\n")

    print(f"結合完了: {output_filepath}")

if __name__ == "__main__":
    # 運用の実態に合わせてパスは適宜変更可能
    INPUT_DIRECTORY = "./raw_logs"       # 未結合の会話ログを配置するローカルフォルダ
    OUTPUT_FILE = "./notebooks_source/log_2026_05_merged.md" # 出力先
    
    # 出力先フォルダがない場合は作成
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    merge_markdown_files(INPUT_DIRECTORY, OUTPUT_FILE)
