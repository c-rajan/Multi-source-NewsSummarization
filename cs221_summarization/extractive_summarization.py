from summarizer import Summarizer


def extractive_summary(body, ratio, min_length=25, max_length=500):
    model = Summarizer(model='bert-base-uncased')
    result = model(body, ratio, min_length, max_length)
    full = ''.join(result)
    return full
