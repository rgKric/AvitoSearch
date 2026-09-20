import pandas as pd
import re
import pymorphy3
from nltk.corpus import stopwords


targets = ["Вид услуги", "Тип услуги",]
queries_renames = {"вид услуги": "search_vid_services", "тип услуги": "search_type_services"}
items_renames = {"вид услуги": "item_vid_services", "тип услуги": "item_type_services"}

def parse_params(text, targets):
    parts = re.split(r'\s+(?=[А-Я])', text)
    idxs = {k: parts.index(k) 
            if k in parts else None 
            for k in targets }

    for k, v in idxs.items():
        if v == len(parts) - 1:
            idxs[k] = None

    result = {k.strip().lower(): parts[idx + 1].strip().lower() 
              if idx != None else None 
              for k, idx in idxs.items() }    

    return result

def extract_params(df, column, targets, renames):
    df[column] = df[column].apply(parse_params, targets=targets)
    df_expanded = pd.json_normalize(df[column]).rename(columns=renames)
    df = pd.concat([df.drop(column, axis=1), df_expanded], axis=1)
    return df


morph = pymorphy3.MorphAnalyzer()
stop_words = set(stopwords.words("russian"))

def preprocess(df, column):
    def preprocess_(text):
        text = str(text).lower()
        words = re.findall(r"[a-zа-яё0-9]+", text)
        result = []
        for word in words:
            if word in stop_words:
                continue
            lemma = morph.parse(word)[0].normal_form
            if lemma not in stop_words:
                result.append(lemma)
        if len(result) == 0:
            return words
        return result

    norm_df = df[column].apply(preprocess_)
    df[f'{column}_norm'] = norm_df
    return df


def preprocess_hf(text, prefix):
    text = str(text).lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return prefix + text.strip()