import pandas as pd
import sys

data_filepath = sys.argv[1]
data = pd.read_csv(data_filepath, delimiter=';', index_col='id')

def tag_word(tagged_words, word, polarity, word_start_idx, aspect_start_pos, aspect_end_pos):
    tag = 'O'
    if word_start_idx >= aspect_start_pos and word_start_idx <= aspect_end_pos:
        # Check if previous word is part of the aspect
        if len(tagged_words) > 0 and tagged_words[-1][1] in ['B-ASP', 'I-ASP']:
            # If previous is part of aspect, tag current as I
            tag = 'I-ASP'
        else:
            # Else, this word starts the aspect, so tag it as B
            tag = 'B-ASP'
    converted_polarity = -999
    if tag != 'O':
        if polarity == -1:
            converted_polarity = 'Negative'
        elif polarity == 1:
            converted_polarity = 'Positive'
        else:
            converted_polarity = 'Neutral'
    tagged_words.append((word, tag, converted_polarity))

tagged_reviews = []
for _, row in data.iterrows():
    review, polarity, _, start_pos, end_pos = row
    word = ''
    start_idx = 0
    tagged_words = []
    review += ' ' # Add extra space at end to guarantee that all words are tagged inside the loop
    for idx, char in enumerate(list(review)):
        if char.isalnum():
            word += char
        else:
            if len(word) > 0:
                tag_word(tagged_words, word, polarity, start_idx, start_pos, end_pos)
            if not char.isspace():
                # Tag special charactes, with exception of spaces
                tag_word(tagged_words, char, polarity, idx, start_pos, end_pos)
            start_idx = idx + 1
            word = ''
    tagged_reviews.append(tagged_words)

with open(sys.argv[2], "w") as out:
    for review_words in tagged_reviews:
        for word, tag, polarity in review_words:
            out.write(f"{word} {tag} {polarity}\n")
        out.write('\n')