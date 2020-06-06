import codecs
import csv
import re
from builtins import range

from fonduer.learning.utils import confusion_matrix
from fonduer.supervision.models import GoldLabel, GoldLabelKey
from fonduer.utils.utils_table import (
    is_row_aligned, 
    is_col_aligned 
)
from fonduer.utils.data_model_utils.visual import (
    is_horz_aligned,
    is_vert_aligned
)
from fonduer.candidates.models import Candidate 
from IPython.display import display, clear_output
from pprint import pprint

try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm


# Define labels
ABSTAIN = -1
FALSE = 0
TRUE = 1


def get_gold_dict(
    filename, docs=None, row_on = True, col_on = False
):
    with codecs.open(filename, encoding="utf-8") as csvfile:
        gold_reader = csv.reader(csvfile)
        gold_dict = set()
        headers = next(gold_reader, None)  # skip the headers
        for row in gold_reader:
            if (row == headers):
                continue
            (data, rowh, colh, doc, r2, r3, r4, r5) = row

            if (docs is None or re.sub("\.xlsx", "", doc).upper() in docs):
                key = []
                key.append(re.sub("\.xlsx", "", doc).upper())
                key.append(data.upper())

                # Test complete candidate
                if (row_on and col_on):
                    key.append(f"{rowh}| {r2} | {r3} | {r4} | {r5}".upper())
                    key.append(colh.upper())
                    gold_dict.add(tuple(key))
                # Otherwise test pair-wise (similar to original TROY200 - TabbyXL)
                elif (row_on):
                    for r in [rowh, r2, r3, r4, r5]:
                        if (r is not None and r is not ""):
                            key_copy = key.copy()
                            key_copy.append(r.upper())
                            gold_dict.add(tuple(key_copy))
                elif (col_on):
                    for c in colh.split("|"):
                        key_copy = key.copy()
                        key_copy.append(c.strip().upper())
                        gold_dict.add(tuple(key_copy))

                # if (docs is None or re.sub("\.xlsx", "", doc).upper() in docs):
                #     key = []
                #     key.append(re.sub("\.xlsx", "", doc).upper())
                #     key.append(data.upper())
                #     if (row_on):
                #         key.append(rowh.upper())
                #     if (col_on):
                #         key.append(colh.upper())
                #     gold_dict.add(tuple(key))

    return gold_dict



def get_gold_func(
    gold_file, row_on=True, col_on=False
): 
    gold_dict = get_gold_dict(
        gold_file, 
        row_on = row_on,
        col_on = col_on,
    )
    def gold(c: Candidate) -> int:
        doc = (c[0].context.sentence.document.name).upper()
        data = c[0].context.get_span()
        align = (c[1].context.get_span()).upper()

        # Account for pre-processing in gold-data 
        if (len([x for x in gold_dict if x[0] == doc and data_matches_gold(data, x[1]) and align_matches_gold(align, x[2])]) > 0):
            return TRUE
        return FALSE
    return gold


# Recover from Gold data formatting errors. 
# E.g. sometimes the commas are replaces by spaces, other times the punctuation is used.
# Not necessarily the exact string from the field.
def data_matches_gold(data, gold):
    # Gold data replaces E or rounds. e.g. "33.30" is "33.3" and "970E" is "970"
    def with_appended(n, gold):
        if (len(n) <= 1):
            return False
        last_letter = n[len(n)-1]
        return (last_letter == "E" or last_letter == "0") and n[0:len(n)-1] == gold
    
    # Gold data rounds # to 0 based on footnote
    if (data == "#" and gold == '0'):
        return True
    n = re.sub("\\xa0", " ", data)
    if (n == gold or with_appended(n, gold)):
        return True
    n = re.sub("\.", ",", n)
    if (n == gold or with_appended(n, gold)):
        return True
    n = re.sub("\\xa0", " ", data)
    n = re.sub(",", " ", n)
    n = re.sub("\.", ",", n)
    if (n == gold or with_appended(n, gold)):
        return True
    n = re.sub("-", "–", n)
    return n == gold or with_appended(n, gold)

# Recover from 2 Gold data formatting errors:
# 1.) Account for the multiple labels given with | separators (one match suffies)
# 2.) Account for the removal of foot-notes number, e.g. the true string was "MATERIAL RECOVERY1", but gold data "MATERIAL RECOVERY"
def align_matches_gold(align, gold):
    def with_footnote(align, gold):
        # Cases such as: "Wood and Derived Fuels3" is "  Wood and Derived Fuels"
        if (re.sub("\(?[.\d]\)?", "", align) in gold):
            return True
        # "2002 1)" is "2002"
        return re.sub("\(?[.\d]\)+", "", align) in gold
    
    align = re.sub("\\xa0", " ", align)
    with_multiple_labels = align in gold
    with_different_minus = re.sub("-", "–", align) in gold or re.sub("–", "-", align) in gold
    # Gold data formats years e.g. 1992 is '1992
    with_gold_different_date_format = f"'{align}" in re.sub(" ", "", gold)
    return with_multiple_labels or with_footnote(align, gold) or with_different_minus or with_gold_different_date_format

def entity_level_f1(
    candidates, gold_file, attribute=None, corpus=None, row_on=True, col_on=False
):
    """Checks entity-level recall of candidates compared to gold.

    Turns a CandidateSet into a normal set of entity-level tuples
    (doc, data, [attribute_value])
    then compares this to the entity-level tuples found in the gold.
    """
    docs = [(re.sub("Document ", "", doc.name)).upper() for doc in corpus] if corpus else None
    price_on = attribute is not None
    gold_set = get_gold_dict(
        gold_file,
        docs=docs,
        row_on=row_on,
        col_on=col_on,
    )
    if len(gold_set) == 0:
        print(f"Gold File: {gold_file}\n Attribute: {attribute}")
        print("Gold set is empty.")
        return
    # Turn CandidateSet into set of tuples
    print("Preparing candidates...")
    entities = set()
    for i, c in enumerate(tqdm(candidates)):
        doc = (c[0].context.sentence.document.name).upper()
        data = (c[0].context.get_span()).upper()
        align = (c[1].context.get_span()).upper()
        if (row_on and col_on):
            align2 = (c[2].context.get_span()).upper()

        # Account for the multiple labels given with | separators (one match suffies)
        matches = [x for x in gold_set if (
                x[0] == doc and 
                data_matches_gold(data, x[1]) and 
                align_matches_gold(align, x[2]) and 
                (not (row_on and col_on) or align_matches_gold(align2, x[3]))
        )]
        if (len(matches) > 0):
            for match in matches:
                align_complete = match[2]
                data = match[1]
                if (row_on and col_on):
                    entities.add((doc, data, align_complete, match[3]))
                else:
                    entities.add((doc, data, align_complete))
        else:
            if (row_on and col_on):
                entities.add((doc, data, align, align2))
            else:
                entities.add((doc, data, align))
    
    (TP_set, FP_set, FN_set) = confusion_matrix(entities, gold_set)
    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)

    prec = TP / (TP + FP) if TP + FP > 0 else float("nan")
    rec = TP / (TP + FN) if TP + FN > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")
    print("========================================")
    print("Scoring on Entity-Level Gold Data")
    print("========================================")
    print(f"Corpus Precision {prec:.3}")
    print(f"Corpus Recall    {rec:.3}")
    print(f"Corpus F1        {f1:.3}")
    print("----------------------------------------")
    print(f"TP: {TP} | FP: {FP} | FN: {FN}")
    print("========================================\n")
    return [sorted(list(x)) for x in [TP_set, FP_set, FN_set]]

def entity_to_candidates(entity, candidate_subset):
    matches = []
    for c in candidate_subset:
        c_entity = tuple(
            [c[0].context.sentence.document.name.upper()]
            + [c[i].context.get_span().upper() for i in range(len(c))]
        )
        c_entity = tuple([str(x) for x in c_entity])
        if c_entity == entity:
            matches.append(c)
    return matches

def merge_candidates(row_candidates, col_candidates):
    candidates_merged = {}
    counter_unmerged = 0
    for c in row_candidates:
        candidates_merged[c.data.context.stable_id] = [c.data, c.row, ""]

    for c in col_candidates:
        if c.data.context.stable_id in candidates_merged:
            candidates_merged[c.data.context.stable_id][2] = c.col
        else:
            counter_unmerged += 1
    cands_merged = [c for i,c in candidates_merged.items() if c[2] != ""]
    counter_unmerged += len(candidates_merged) - len(cands_merged)
    print(f"Merged {len(cands_merged)} Row/Col candidates by stable IDs of the data items. {counter_unmerged} unmerged items discarded.")
    return cands_merged

def run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, align_type='row', doc_type=2):
    docs = all_docs[doc_type]
    test_cands = cands[0][doc_type][0] if align_type == 'row' else cands[1][doc_type][0]
    test_cands_formatted = [[x.data, getattr(x, align_type)] for x in test_cands]
    row_on = True if align_type == 'row' else False
    col_on = True if align_type == 'col' else False
    (TP, FP, FN) = entity_level_f1(test_cands_formatted, gold_file, ATTRIBUTE, docs, row_on=row_on, col_on=col_on)
    return (TP, FP, FN)


# Helper functions for analysis of FN, FP candidates and the corresponding mentions

def split_mentions(mentions, mention_subclasses):
    data_mentions = [x for x in mentions if isinstance(x, mention_subclasses[0])]
    row_mentions = [x for x in mentions if isinstance(x, mention_subclasses[1])]
    col_mentions = [x for x in mentions if isinstance(x, mention_subclasses[2])]

    print(f"Data mentions={len(data_mentions)}, Row mentions={len(row_mentions)}, Col mentions={len(col_mentions)}")
    return (data_mentions, row_mentions, col_mentions)

def mention_analysis(align_val, data_val, document_name, align_type, mentions, mention_subclasses):
    (data_mentions, row_mentions, col_mentions) = split_mentions(mentions, mention_subclasses)
    align_mentions = row_mentions if align_type == 'row' else col_mentions
    s1s = [x for x in data_mentions if data_val == x.context.get_span().upper() and x.document.name == document_name]
    s2s = [x for x in align_mentions if align_val in x.context.get_span().upper() and x.document.name == document_name]
    print("######################################")
    print(f"Mention Analysis {align_type}")
    print(f"{len(s1s)} mentions for the data value {data_val} in document {document_name}.")
    print(f"{len(s2s)} mentions for the align value {align_val} in document {document_name}.")
    if (len(s1s) > 0 and len(s2s) > 0):
        for s1 in s1s:
            for s2 in s2s:
                print("")
                print(f"Sentence of the data value: {s1.context.sentence}")
                print(f"Sentence of the align value: {s2.context.sentence}")
                aligned = is_row_aligned(s1.context.sentence, s2.context.sentence) if align_type == 'row' else is_col_aligned(s1.context.sentence, s2.context.sentence)
                print(f"The values are {align_type} aligned: {aligned}")
                hv_aligned = is_horz_aligned((s1, s2)) if align_type == 'row' else is_vert_aligned((s1, s2))
                hv = "horizontally" if align_type == 'row' else "vertically"
                print(f"The values are {hv} aligned: {hv_aligned}")
    elif (len(s1s) > 0):
        for s1 in s1s:
            print(f"Sentence of the data value: {s1.context.sentence}")
        print("")
        print(f"No align value found in all align mentions for the document {document_name}:")
        pprint([x for x in align_mentions if x.document.name == document_name])
    elif (len(s2s) > 0):
        for s2 in s2s:
            print(f"Sentence of the align value: {s2.context.sentence}")
        print("")
        print(f"No data value found in all data mentions for the document {document_name}:")
        pprint([x for x in data_mentions if x.document.name == document_name])
    print("######################################")


def fn_analysis(cand_nr, FN, mentions, mention_subclasses):
    # Only unique key values (skip same error in docs)
    fn_diff_documents = [x for k,x in { x[0]:x for x in FN }.items()]
    
    (doc, data, row, col) = fn_diff_documents[cand_nr]
    
    print("######################################")
    print(f"FN Candidate: {cand_nr+1}/{len(fn_diff_documents)}")
    print(doc)
    print(data)
    print(row)
    print(col)
    print("######################################")
    mention_analysis(row, data, doc, "row", mentions, mention_subclasses)
    mention_analysis(col, data, doc, "col", mentions, mention_subclasses)
    
    
def gold_analysis(gold_file, align_val, data_val, document_name, align_type):
    row_on = True if align_type == 'row' else False
    col_on = True if align_type == 'col' else False
    docs = [document_name]
    gold_dict = get_gold_dict(gold_file, docs=docs, row_on=row_on, col_on=col_on)
    
    align_gold = [x for x in gold_dict if align_matches_gold(align_val,x[2])]
    data_gold = [x for x in gold_dict if data_matches_gold(data_val, x[1])]
    print("######################################")
    print(f"Gold analysis {align_type}")
    if (len(align_gold) > 0):
        print(f"Could match the align value")
        pprint(align_gold)
    if (len(data_gold) > 0):
        print(f"Could match the data value")
        pprint(data_gold)
    if (len(data_gold) + len(align_gold) == 0):
        print("Could not match align nor data value")
        pprint(gold_dict)
    print("######################################")

    
class Counter:
    def __init__(self, params, plus_btn, minus_btn, d_type="fp", initial=0, maximum=0, minimum=0):
        self.d_type = d_type
        self.value = initial
        self.maximum = maximum
        self.minimum = 0
        self.params = params
        self.plus_btn = plus_btn
        self.minus_btn = minus_btn
        self.cand = self._display()

    def increment(self, amount=1):
        if (self.value+amount > self.maximum):
            return self.value
        self.value += amount
        return self.value
    
    def decrement(self, amount=1):
        if (self.value-amount < 0):
            return self.value
        self.value -= amount
        return self.value

    def display(self):
        # Clear previous
        clear_output(wait=True)
        # Redraw
        display(self.minus_btn)
        display(self.plus_btn)
        self.cand = self._display()

    def _display(self):
        cand_nr = self.value
        (gold_file, FP, FN, mentions, mention_subclasses, cands_merged) = self.params
        
        if (self.d_type == "fp"):
            # Only unique key values (skip same error in docs)
            f_diff_documents = [x for k,x in { x[0]:x for x in FP }.items()]
            
            # Get a list of candidates that match the FN[10] entity
            f_cands = entity_to_candidates(f_diff_documents[cand_nr], cands_merged)

            # Display a candidate
            f_cand = f_cands[0]
            (data, row, col) = f_cand
            row_val = row.context.get_span().upper()
            col_val = col.context.get_span().upper()
            data_val = data.context.get_span().upper()
            print("######################################")
            print(f"{self.d_type.upper()} Candidate: {cand_nr+1}/{len(f_diff_documents)}")
            print(data.context.sentence)
            print(row.context.sentence)
            print(col.context.sentence)
            print("######################################")

            gold_analysis(gold_file, row_val, data_val, data.document.name, "row")
            gold_analysis(gold_file, col_val, data_val, data.document.name, "col")
            return f_cand
        else:
            fn_analysis(cand_nr, FN, mentions, mention_subclasses)
        # vis.display_candidates([fp_cand])

    def __iter__(self, sentinal=False):
        return iter(self.increment, sentinal)   
