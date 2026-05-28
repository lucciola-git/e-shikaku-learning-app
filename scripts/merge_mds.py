import os
import glob

def combine_markdown_files(input_dir, output_filepath):
    """
    指定されたフォルダ内のすべての.mdファイルを結合し、1つのファイルに出力する。
    """
    # 指定フォルダ内の.mdファイルをすべて取得
    search_path = os.path.join(input_dir, "*.md")
    md_files = glob.glob(search_path)
    
    # 結合順序を安定させるためファイル名でソート
    md_files.sort()
    
    if not md_files:
        print(f"対象フォルダ内にマークダウンファイルが見つかりません: {input_dir}")
        return

    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for filepath in md_files:
            # 処理中のファイル名を出力（ログ用）
            print(f"結合中: {os.path.basename(filepath)}")
            
            with open(filepath, 'r', encoding='utf-8') as infile:
                # ファイル内容を書き込み
                outfile.write(infile.read())
                # ファイル間に適切な改行を挿入
                outfile.write("\n\n")
                
    print(f"結合が完了しました。出力先: {output_filepath}")

if __name__ == "__main__":
    # 運用に合わせてパスを変更してください
    INPUT_DIRECTORY = "./markdown_inputs" 
    OUTPUT_FILE = "./combined_notes_month.md"
    
    # 入力フォルダが存在しない場合は作成
    if not os.path.exists(INPUT_DIRECTORY):
        os.makedirs(INPUT_DIRECTORY)
        print(f"フォルダ '{INPUT_DIRECTORY}' を作成しました。ここに結合したいファイルを配置してください。")
    else:
        combine_markdown_files(INPUT_DIRECTORY, OUTPUT_FILE)
