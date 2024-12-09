import json

def read_file(path):
    with open(path, "r") as f:
        return f.readlines()

def filter_by_word(lines, word):
    return [line for line in lines if word in line.split(' ')]

def filter_by_full_word(lines, word):
    return [line for line in lines if word in line.split(' ')]

if __name__ == "__main__":
    # file_path = "/data/hkkang/NL2QGM/samsung_analysis/model_1_samsung_data.txt"
    file_path = "/data/hkkang/NL2QGM/data/samsung-addop-hkkang/all.tsv"
    target_words = ["line_id"]
    target_full_words = [""]
    
    lines = read_file(file_path)
    for word in target_words:
        if word:
            lines = filter_by_word(lines, word)

    for word in target_full_words:
        if word:
            lines = filter_by_full_word(lines, word)
    
    # show
    for idx, line in enumerate(lines):
        print(f"{idx}: {line}")