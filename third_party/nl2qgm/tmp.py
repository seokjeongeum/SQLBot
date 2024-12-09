from typing import *

import en_core_web_trf
import spacy


def extract_nouns(
    model: spacy.Language, sentence: str, enable_PROPN: bool = False
) -> List:
    # Define target POS tags
    target_pos = ["PROPN", "NOUN"] if enable_PROPN else ["NOUN"]

    # Perform parsing
    parsed_doc = model(sentence)

    # Extract nouns
    flag = False
    nouns = []
    tmp_word = []
    for word in parsed_doc:
        w_text = word.text
        w_pos = word.pos_
        # If the word is a noun
        if w_pos in target_pos:
            if flag == False:
                # If it is the beginning of a noun phrase
                tmp_word.append(w_text)
                flag = True
            else:
                # Add the word to the noun phrase (if it is not the beginning)
                tmp_word.append(w_text)
        else:
            if flag == True:
                # End of noun phrase
                nouns.append(" ".join(tmp_word))
                tmp_word = []
                flag = False

    if flag:
        nouns.append(" ".join(tmp_word))

    # # Debugging
    # print([(word.text, word.pos_) for word in parsed_doc])

    return nouns


def extract_dependencies(model: spacy.Language, sentence: str) -> List:
    pass


def main():
    # Load model
    model = spacy.load("en_core_web_trf")
    model = en_core_web_trf.load()

    while True:
        sentence = input("Enter sentence: ").strip()
        if sentence == "q":
            print("Bye!")
            break
        noun_words = extract_nouns(model=model, sentence=sentence, enable_PROPN=True)
        print(noun_words)


if __name__ == "__main__":
    main()
