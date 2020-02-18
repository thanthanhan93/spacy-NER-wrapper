from nltk.tokenize import word_tokenize
import re 
import json


def _spans(txt):
    tokens = word_tokenize(txt)
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield token, offset, offset + len(token)
        offset += len(token)
        

def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.
    Args:
        data (list): The data to be cleaned in spaCy JSON format.
    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')
    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    return cleaned_data


def convert_to_sentence_entities(train_path):
    f = open(train_path, 'r', encoding='utf-8')
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                yield ' '.join(sentence), label
                sentence = []
                label = []
            continue
        splits = line.split('	')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])
    if len(sentence) > 0:
        yield ' '.join(sentence), label
        
    
def convert_to_spacy_format(**kwargs):
    data = convert_to_sentence_entities(**kwargs)
    parsed_data = []
    for sample in data:
        parsed_annos = []
        text = ' '.join(sample[0])
        bla = list(_spans(text))
        
        for token, anno in zip(bla, sample[1]):
            if anno == 'O':
                continue
            else:
                parsed_annos.append((token[1], token[2], anno))
        parsed_data.append((text, {'entities': parsed_annos}))
    return trim_entity_spans(parsed_data)

def doccano_to_spacy(train_path):
    f = open(train_path, 'r', encoding='utf-8')
    lines = []
    for line in f:
        line = json.loads(line)
        lines.append((line['text'], {"entities": line['labels']}))
    return trim_entity_spans(lines)