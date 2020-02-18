from spacy.gold import iob_to_biluo, GoldParse, docs_to_json
from spacy.scorer import Scorer

def evaluate(model, examples, verbose=1):
    scorer = Scorer()
    for input_, annot in examples:
        
        doc_gold_text = model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = model(input_)
        #return gold
        
        current_score = scorer.ents_f
        scorer.score(pred_value, gold)
        if ((verbose==1) & (current_score > scorer.ents_f)):
            print_beauty_NER(prediction_to_IOB(pred_value, gold))
        current_score = scorer.ents_f
        
    return scorer.scores

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def prediction_to_IOB(prediction, gt):
    """
    CONVERT PREDICTION AND GROUTH TRUTH TO BILOU SCHEMA

    Input:
        - prediction: spacy.Doc
        - gt : spacy.GoldParse
    
    Output:
        - list of list contains every token info in prediction.text
    """
    tag_iob = []
    for token in prediction:
        if (token.ent_type_ == ''):
            tag_iob.append(token.ent_iob_)
        else:
            tag_iob.append('-'.join([token.ent_iob_,token.ent_type_]))
    tokens = [token.text for token in prediction]
    
    tag_new_iob = iob_to_biluo(tag_iob)
    gt_NER = gt.ner
    return [[token, 
            true_label, 
            pred_iob] for token, true_label, pred_iob in zip(tokens, gt_NER, tag_new_iob)]

def print_beauty_NER(prediction, **kwargs):
    """
    Display beautiful prediction 
    """
    print('{:<16s}{:>10s}{:>14s}'.format("TOKEN", "TRUE_LABEL", "PRED_LABEL"))
    print('-' * 40)
    for item in prediction:
        if item[1] == item[2]:
            print('{:<16s}{:>10s}{:>14s}'.format(*item))
        else:
            rules = ["<16s", ">10s", ">14s"]
            print('{}{}{}'.format(*[color_your_str(word, rule, **kwargs) for word, rule in zip(item, rules)]))
    print('\n')
    
def filter_unmatch_prediction(prediction):
    return list(filter(lambda x: x[1]!=x[2], prediction))

def color_your_str(text, rule, color=bcolors.WARNING):
    """
    Colorize the string by rule
    """
    return color + f"{text:{rule}}{bcolors.ENDC}".format(text= text, rule= rule)

