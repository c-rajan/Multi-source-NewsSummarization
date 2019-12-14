import numpy as np
import spacy
from spacy.lang.en import English



def split_into_sentences(nlp, text):
    doc = nlp(text)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences


if __name__ == '__main__':
    raw_text = 'Hello, world. Here are two sentences.'
    nlp = English()
    
