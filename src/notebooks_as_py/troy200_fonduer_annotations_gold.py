#!/usr/bin/env python
# coding: utf-8

# # Extract Column/Row Heading with Data information from Tabby200

# # Introduction
# 
# In this notebook we use `Fonduer` to extract relations from the `Tabby200` dataset.  
# This code is a modified version of their original hardware [tutorial](https://github.com/HazyResearch/fonduer-tutorials/tree/master/hardware).  
# The `Fonduer` pipeline (as outlined in the [paper](https://arxiv.org/abs/1703.05028)), and the iterative KBC process:
# 
# 1. KBC Initialization
# 2. Candidate Generation and Multimodal Featurization
# 3. Probabilistic Relation Classification
# 4. Error Analysis and Iterative KBC
# 
# Additionally we assume that the dataset of spreadsheets has been preprocessed with cell annotations and table recognition as described in [paper1](https://ieeexplore.ieee.org/document/8970946/), [paper2](https://wwwdb.inf.tu-dresden.de/wp-content/uploads/demo_paper.pdf). The spreadsheets are converted to HTML/PDF format by libreoffice: `libreoffice --headless --calc --convert-to html --outdir html/ spreadsheet/*`. This results in an HTML format which is one large table. However, the cells are styled to indicate their classification:
# 
# * "Table": 'bgcolor=#FFFFFE' background
# * "Data": 'color=#000001' font
# * "Header": 'color=#000002' font
# * "MetaTitle": 'color=#000003' font
# * "Notes": 'color=#000004' font 

# ## Setup
# 
# First we import the relevant libraries and connect to the local database.  
# Follow the README instructions to setup the connection to the postgres DB correctly.
# 
# If the database has existing candidates with generated features, the will not be overriden.  
# To re-run the entire pipeline including initialization drop the database first.

# In[ ]:


get_ipython().system(' dropdb -h postgres -h postgres -h postgres -h postgres --if-exists troy200_col_row_data_gold')
get_ipython().system(' createdb -h postgres -h postgres -h postgres -h postgres troy200_col_row_data_gold')


# In[ ]:


# source .venv/bin/activate


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sys
import logging


# In[ ]:


PARALLEL = 8 # 4  # assuming a quad-core machine
ATTRIBUTE = "troy200_col_row_data_gold"

DB_USERNAME = 'user'
DB_PASSWORD = 'venron'
conn_string = f'postgresql://{DB_USERNAME}:{DB_PASSWORD}@postgres:5432/{ATTRIBUTE}'
    
docs_path = 'data/gold/html/'
pdf_path = 'data/gold/pdf/'
gold_file = 'data/troy200_gold.csv'
max_docs = 200 # 50 # 200


# ## 1.1 Parsing and Transforming the Input Documents into Unified Data Models
# 
# We first initialize a `Meta` object, which manages the connection to the database automatically, and enables us to save intermediate results.

# In[ ]:


from fonduer import Meta, init_logging

# Configure logging for Fonduer
init_logging(log_dir="logs")

session = Meta.init(conn_string).Session()


# In[ ]:


from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.parser.models import Document, Sentence
from fonduer.parser import Parser

has_documents = session.query(Document).count() > 0

corpus_parser = Parser(session, structural=True, lingual=True, visual=True, pdf_path=pdf_path)

if (not has_documents): 
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    get_ipython().run_line_magic('time', 'corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)')
    
print(f"Documents: {session.query(Document).count()}")
print(f"Sentences: {session.query(Sentence).count()}")


# ## 1.2 Dividing the Corpus into Test and Train
# 
# We'll split the documents 80/10/10 into train/dev/test splits. Note that here we do this in a non-random order to preserve the consistency and we reference the splits by 0/1/2 respectively.

# In[ ]:


docs = session.query(Document).order_by(Document.name).all()
ld   = len(docs)

train_docs = set()
dev_docs   = set()
test_docs  = set()
splits = (0.8, 0.9)
data = [(doc.name, doc) for doc in docs]
data.sort(key=lambda x: x[0])
for i, (doc_name, doc) in enumerate(data):
    if i < splits[0] * ld:
        train_docs.add(doc)
    elif i < splits[1] * ld:
        dev_docs.add(doc)
    else:
        test_docs.add(doc)
all_docs = [train_docs, dev_docs, test_docs]
from pprint import pprint
pprint([x.name for x in train_docs][0:5])
print(f"Number of documents split: {len(docs)}")


# # Phase 2: Mention Extraction, Candidate Extraction Multimodal Featurization
# 
# Given the unified data model from Phase 1, `Fonduer` extracts relation
# candidates based on user-provided **matchers** and **throttlers**. Then,
# `Fonduer` leverages the multimodality information captured in the unified data
# model to provide multimodal features for each candidate.
# 
# ## 2.1 Mention Extraction & Candidate Generation
# 
# 1. Define mention classes
# 2. Use matcher functions to define the format of potential mentions
# 3. Define Mentionspaces (Ngrams)
# 4. Run Mention extraction (all possible ngrams in the document, API [ReadTheDocs](https://fonduer.readthedocs.io/en/stable/user/candidates.html#fonduer.candidates.MentionExtractor))

# In[ ]:


from fonduer.candidates.models import mention_subclass
from fonduer.candidates.matchers import RegexMatchSpan, LambdaFunctionMatcher, Intersect, Union
from fonduer.utils.data_model_utils.structural import _get_node
from fonduer.candidates import MentionNgrams, MentionSentences
from fonduer.candidates import MentionExtractor 
from fonduer.candidates.models import Mention

hasMentions = session.query(Mention).count() > 0

# 1.) Mention subclasses
Data = mention_subclass("Data")
Row = mention_subclass("Row")
Col = mention_subclass("Col")

def get_label_matcher(label):
    def label_matcher(mention):
        html_attrs = mention.sentence.html_attrs
        parent_attrs = [f"{k}={v}" for k,v in _get_node(mention.sentence).getparent().attrib.items()]
        
        return {
            "Table": 'bgcolor=#FFFFFE' in (html_attrs + parent_attrs) ,
            "Data": 'color=#000001' in html_attrs,
            "Header": 'color=#000002' in html_attrs,
            "MetaTitle": 'color=#000003' in html_attrs,
            "Notes": 'color=#000004' in html_attrs,
        }[label]
    return label_matcher

if (not hasMentions):

    # 2.) Matcher functions
    # Regex: Match any numbers, including points, commas, percentage, minus or the format "7 to 8" or simply "x"
    data_regex_matcher = RegexMatchSpan(rgx=r"[0-9-,.%$#]+( to | )?[0-9-,.%$#]*|^x$", longest_match_only=True)
    data_label_matcher = LambdaFunctionMatcher(func=get_label_matcher("Data"))
    data_matcher = Intersect(data_regex_matcher, data_label_matcher)
    # Regex-Matcher for only matching the longest string in all Headers
    row_regex_matcher = RegexMatchSpan(rgx=r"^.*$", longest_match_only=True)
    row_label_matcher = LambdaFunctionMatcher(func=get_label_matcher("Header"))
    row_matcher = Intersect(row_regex_matcher, row_label_matcher)
    col_regex_matcher = RegexMatchSpan(rgx=r"^.*$", longest_match_only=True)
    col_label_matcher = LambdaFunctionMatcher(func=get_label_matcher("Header"))
    col_matcher = Intersect(col_regex_matcher, col_label_matcher)

    # 3.) Mention spaces (Ngrams)
    data_ngrams = MentionSentences() # MentionNgrams(n_max=3)
    row_ngrams = MentionSentences() # MentionNgrams(n_min=1, n_max=8)
    col_ngrams = MentionSentences() # MentionNgrams(n_min=1, n_max=8)

    # 4.) Mention extraction
    mention_extractor = MentionExtractor(
        session, [Data, Row, Col],  [data_ngrams, row_ngrams, col_ngrams], [data_matcher, row_matcher, col_matcher]
    )
    docs = session.query(Document).order_by(Document.name).all()
    mention_extractor.apply(docs, parallelism=PARALLEL)

    
print(f"Total Mentions: {session.query(Mention).count()}")


# In[ ]:


mentions = session.query(Mention).all()


# ## 2.2 Candidate Extraction
# 
# 1. Define Candidate Class
# 2. Define trottlers to reduce the number of possible candidates
# 3. Extract candidates (View the API for the CandidateExtractor on [ReadTheDocs](https://fonduer.readthedocs.io/en/stable/user/candidates.html#fonduer.candidates.MentionExtractor).)
# 
# In the last part we specified that these `Candidates` belong to the training set by specifying `split=0`; recall that we're referring to train/dev/test as splits 0/1/2.

# In[ ]:


from fonduer.utils.data_model_utils import *
from fonduer.utils.utils_table import is_row_aligned, is_col_aligned
import re
from fonduer.candidates import CandidateExtractor
from fonduer.candidates.models import candidate_subclass
from fonduer.utils.visualizer import Visualizer


# 1.) Define Candidate class
RowCandidate = candidate_subclass("RowCandidate", [Data, Row])
ColCandidate = candidate_subclass("ColCandidate", [Data, Col])

has_candidates = (
    session.query(RowCandidate).filter(RowCandidate.split == 0).count() > 0 or
    session.query(ColCandidate).filter(ColCandidate.split == 0).count() > 0
)

# 2.) DefineThrottlers
def row_filter(c):
    (data, row) = c
     # Ignore only empty candidate values
    if (re.match("^[\., -]*$", data.context.get_span())):
        return False
    if same_table((data, row)):
        d = data.context.sentence
        r = row.context.sentence
        return (is_row_aligned(d, r)) # and is_horz_aligned((data, row)))
    return True

def col_filter(c):
    (data, col) = c
    # Ignore only empty candidate values
    if (re.match("^[\., -]*$", data.context.get_span())):
        return False
    if same_table((data, col)):
        d = data.context.sentence
        c = col.context.sentence
        return (is_col_aligned(d, c)) # and is_vert_aligned((data, col)))
    return True


# 3.) Candidate extraction
candidate_extractor = CandidateExtractor(session, [RowCandidate], throttlers=[row_filter])

for i, docs in enumerate([train_docs, dev_docs, test_docs]):
    if (not has_candidates):
        candidate_extractor.apply(docs, split=i, parallelism=PARALLEL)
    print(f"Number of Candidates in split={i}: {session.query(RowCandidate).filter(RowCandidate.split == i).count()}")

cands_row = [
    candidate_extractor.get_candidates(split = 0),
    candidate_extractor.get_candidates(split = 1),
    candidate_extractor.get_candidates(split = 2),
]

candidate_extractor = CandidateExtractor(session, [ColCandidate], throttlers=[col_filter])

for i, docs in enumerate([train_docs, dev_docs, test_docs]):
    if (not has_candidates):
        candidate_extractor.apply(docs, split=i, parallelism=PARALLEL)
    print(f"Number of Candidates in split={i}: {session.query(ColCandidate).filter(ColCandidate.split == i).count()}")

cands_col = [
    candidate_extractor.get_candidates(split = 0),
    candidate_extractor.get_candidates(split = 1),
    candidate_extractor.get_candidates(split = 2),
]
                
cands = [cands_row, cands_col]

# 4.) Visualize some candidate for error analysis
train_cand = cands[0][0][0][2]
pprint(train_cand)
vis = Visualizer(pdf_path)

# Display a candidate
vis.display_candidates([train_cand])


# ## 2.3 Rule-based Pair-wise Evaluation Test
# 
# We test the performance only based on the cell annotation rules for mentions (similar to TabbyXL rule-based algorithm), without any trained model.

# In[ ]:


from troy200_utils import run_pairwise_eval

get_ipython().run_line_magic('time', "row_results_train = run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, 'row', 0)")
get_ipython().run_line_magic('time', "col_results_train = run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, 'col', 0)")

get_ipython().run_line_magic('time', "row_results_dev = run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, 'row', 1)")
get_ipython().run_line_magic('time', "col_results_dev = run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, 'col', 1)")

get_ipython().run_line_magic('time', "row_results_test = run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, 'row', 2)")
get_ipython().run_line_magic('time', "col_results_test = run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, 'col', 2)")


# In[ ]:


def summarize_results(results):
    row_results_train = results[0][0]
    row_results_dev = results[0][1]
    row_results_test = results[0][2]
    col_results_train = results[1][0]
    col_results_dev = results[1][1]
    col_results_test = results[1][2]
    
    prec_test = (
        len(row_results_test[0]) + 
        len(col_results_test[0])
    ) / (
        len(row_results_test[0]) + 
        len(col_results_test[0]) +
        len(row_results_test[1]) + 
        len(col_results_test[1])
    )

    rec_test = (
        len(row_results_test[0]) + 
        len(col_results_test[0])
    ) / (
        len(row_results_test[0]) + 
        len(col_results_test[0]) +
        len(row_results_test[2]) + 
        len(col_results_test[2])
    )
    f1_test = 2 * (prec_test * rec_test) / (prec_test + rec_test)

    pos_total = (
        len(row_results_train[0]) + 
        len(col_results_train[0]) + 
        len(row_results_dev[0]) + 
        len(col_results_dev[0]) + 
        len(row_results_test[0]) + 
        len(col_results_test[0])
    )
    prec_total = pos_total / (
        pos_total + 
        len(row_results_train[1]) + 
        len(col_results_train[1]) + 
        len(row_results_dev[1]) + 
        len(col_results_dev[1]) + 
        len(row_results_test[1]) + 
        len(col_results_test[1])
    )
    rec_total = pos_total / (
        pos_total + 
        len(row_results_train[2]) + 
        len(col_results_train[2]) + 
        len(row_results_dev[2]) + 
        len(col_results_dev[2]) + 
        len(row_results_test[2]) + 
        len(col_results_test[2])
    )
    f1_total = 2 * (prec_total * rec_total) / (prec_total + rec_total)
    
    return (prec_test, rec_test, f1_test, prec_total, rec_total, f1_total)


(prec_test, rec_test, f1_test, prec_total, rec_total, f1_total) = summarize_results(
[
    [row_results_train, row_results_dev, row_results_test],
    [col_results_train, col_results_dev, col_results_test]
])
print(f"TOTAL DOCS PAIRWISE: Precision={prec_total}, Recall={rec_total}, F1={f1_total}")
print(f"TEST PAIRWISE: Precision={prec_test}, Recall={rec_test}, F1={f1_test}")


# ## 2.4 Rule-based Candidate Evaluation Test
# 
# We also test the performance on merged candidates, so correct row and column labels for a specific data point.

# In[ ]:


from troy200_utils import merge_candidates, entity_level_f1

cands_merged = merge_candidates(cands[0][2][0], cands[1][2][0])
get_ipython().run_line_magic('time', '(TP, FP, FN) = entity_level_f1(cands_merged, gold_file, ATTRIBUTE, test_docs, row_on=True, col_on=True)')


# We can further analyze the false-positive and false-negative results via a simple counter-interface.

# In[ ]:


from troy200_utils import Counter
from ipywidgets import widgets
from functools import partial
from IPython.display import display

# Buttons
minus = widgets.Button(description='<')
plus = widgets.Button(description='>')

display(minus)
display(plus)

counter = Counter(
    params=(
        gold_file, 
        FP, 
        FN, 
        mentions,
        [Data, Row, Col],
        cands_merged,
    ), 
    plus_btn=plus,
    minus_btn=minus, 
    d_type="fn", # "fp"
    initial=0, 
    maximum=len(FN)-1, # len(FP)-1, 
)

def btn_inc(counter, w):
    counter.increment()  
    counter.display()

def btn_dec(counter, w):
    counter.decrement()
    counter.display()

minus.on_click(partial(btn_dec, counter))
plus.on_click(partial(btn_inc, counter))


# # Phase 3: Supervised Classification
# 
# 1. Featurize the candidates
# 2. Load Gold Data
# 3. Build and train a descriminative model and test on the test set
# 
# 
# For this data set we do not use the labeling functions provided by Fonduer, as it is extremely small and we have gold labels for all instances.
# 
# ### 3.1) Featurize the candidates
# 
# """
# Unlike dealing with plain unstructured text, `Fonduer` deals with richly formatted data, and consequently featurizes each candidate with a baseline library of multimodal features. 
# 
# ### Featurize with `Fonduer`'s optimized Postgres Featurizer
# We now annotate the candidates in our training, dev, and test sets with features. The `Featurizer` provided by `Fonduer` allows this to be done in parallel to improve performance.
# 
# View the API provided by the `Featurizer` on [ReadTheDocs](https://fonduer.readthedocs.io/en/stable/user/features.html#fonduer.features.Featurizer).
# 
# At the end of this phase, `Fonduer` has generated the set of candidates and the feature matrix. Note that Phase 1 and 2 are relatively static and typically are only executed once during the KBC process.
# """

# In[ ]:


from fonduer.features import Featurizer
from fonduer.features.models import Feature

has_features = session.query(Feature).count() > 0

# Features for row/column candidates (train, dev, test)
F = []

for i, cands_align in enumerate(cands):
    featurizer = Featurizer(session, [RowCandidate]) if i == 0 else Featurizer(session, [ColCandidate])
    train_cands = cands_align[0]
    dev_cands = cands_align[1]
    test_cands = cands_align[2]

    if (not has_features):
        # Training set
        get_ipython().run_line_magic('time', 'featurizer.apply(split=0, train=True, parallelism=PARALLEL)')
        get_ipython().run_line_magic('time', 'F_train = featurizer.get_feature_matrices(train_cands)')
        print(F_train[0].shape)

        # Dev set
        get_ipython().run_line_magic('time', 'featurizer.apply(split=1, parallelism=PARALLEL)')
        get_ipython().run_line_magic('time', 'F_dev = featurizer.get_feature_matrices(dev_cands)')
        print(F_dev[0].shape)

        # Test set
        get_ipython().run_line_magic('time', 'featurizer.apply(split=2, parallelism=PARALLEL)')
        get_ipython().run_line_magic('time', 'F_test = featurizer.get_feature_matrices(test_cands)')
        print(F_test[0].shape)
    else:
        get_ipython().run_line_magic('time', 'F_train = featurizer.get_feature_matrices(train_cands)')
        get_ipython().run_line_magic('time', 'F_dev = featurizer.get_feature_matrices(dev_cands)')
        get_ipython().run_line_magic('time', 'F_test = featurizer.get_feature_matrices(test_cands)')
    # Summarize for row/col
    F.append([F_train, F_dev, F_test])


# ### 3.2) Loading Gold LF

# In[ ]:


from fonduer.supervision.models import GoldLabel
from fonduer.supervision import Labeler
from troy200_utils import get_gold_func

# 1.1) Load the gold data rows
gold_row = get_gold_func(gold_file, row_on=True, col_on=False)
docs = corpus_parser.get_documents()
labeler = Labeler(session, [RowCandidate])
get_ipython().run_line_magic('time', 'labeler.apply(docs=docs, lfs=[[gold_row]], table=GoldLabel, train=True, parallelism=PARALLEL)')

# 1.2) Load the gold data cols
gold_col = get_gold_func(gold_file, row_on=False, col_on=True)
docs = corpus_parser.get_documents()
labeler = Labeler(session, [ColCandidate])
get_ipython().run_line_magic('time', 'labeler.apply(docs=docs, lfs=[[gold_col]], table=GoldLabel, train=True, parallelism=PARALLEL)')

gold = [gold_row, gold_col]


# ### 3.3) Training the Discriminative Model 
# 
# Fonduer uses the machine learning framework [Emmental](https://github.com/SenWu/emmental) to support all model training.

# In[ ]:


import emmental
import numpy as np

from emmental.modules.embedding_module import EmbeddingModule
from emmental.data import EmmentalDataLoader
from emmental.model import EmmentalModel
from emmental.learner import EmmentalLearner
from fonduer.learning.utils import collect_word_counter
from fonduer.learning.dataset import FonduerDataset
from fonduer.learning.task import create_task

ABSTAIN = -1
FALSE = 0
TRUE = 1

def train_model(cands, F, align_type, model_type="LogisticRegression"):
    # Extract candidates and features based on the align type (row/column)
    align_val = 0 if align_type == "row" else 1
    train_cands = cands[align_val][0]
    F_train = F[align_val][0]
    train_marginals = np.array([[0,1] if gold[align_val](x) else [1,0] for x in train_cands[0]])
    
    # 1.) Setup training config
    config = {
        "meta_config": {"verbose": True},
        "model_config": {"model_path": None, "device": 0, "dataparallel": False},
        "learner_config": {
            "n_epochs": 50,
            "optimizer_config": {"lr": 0.001, "l2": 0.0},
            "task_scheduler": "round_robin",
        },
        "logging_config": {
            "evaluation_freq": 1,
            "counter_unit": "epoch",
            "checkpointing": False,
            "checkpointer_config": {
                "checkpoint_metric": {f"{ATTRIBUTE}/{ATTRIBUTE}/train/loss": "min"},
                "checkpoint_freq": 1,
                "checkpoint_runway": 2,
                "clear_intermediate_checkpoints": True,
                "clear_all_checkpoints": True,
            },
        },
    }

    emmental.init(Meta.log_path)
    emmental.Meta.update_config(config=config)
    
    # 2.) Collect word counter from training data
    word_counter = collect_word_counter(train_cands)
    
    # 3.) Generate word embedding module for LSTM model
    # (in Logistic Regression, we generate it since Fonduer dataset requires word2id dict)
    # Geneate special tokens
    arity = 2
    specials = []
    for i in range(arity):
        specials += [f"~~[[{i}", f"{i}]]~~"]

    emb_layer = EmbeddingModule(
        word_counter=word_counter, word_dim=300, specials=specials
    )
    
    # 4.) Generate dataloader for training set
    # No noise in Gold labels
    train_dataloader = EmmentalDataLoader(
        task_to_label_dict={ATTRIBUTE: "labels"},
        dataset=FonduerDataset(
            ATTRIBUTE,
            train_cands[0],
            F_train[0],
            emb_layer.word2id,
            train_marginals,
        ),
        split="train",
        batch_size=100,
        shuffle=True,
    )
    
    # 5.) Training 
    tasks = create_task(
        ATTRIBUTE, 2, F_train[0].shape[1], 2, emb_layer, model=model_type # "LSTM" 
    )

    model = EmmentalModel(name=f"{ATTRIBUTE}_task")

    for task in tasks:
        model.add_task(task)

    emmental_learner = EmmentalLearner()
    emmental_learner.learn(model, [train_dataloader])
    
    return (model, emb_layer)


# In[ ]:


def eval_model(model, emb_layer, cands, F, align_type = "row"):
    # Extract candidates and features based on the align type (row/column)
    align_val = 0 if align_type == "row" else 1
    train_cands = cands[align_val][0]
    dev_cands = cands[align_val][1]
    test_cands = cands[align_val][2] 
    F_train = F[align_val][0]
    F_dev = F[align_val][1]
    F_test = F[align_val][2]
    row_on = True if align_type == "row" else False
    col_on = True if align_type == "col" else False
    
    # Generate dataloader for test data
    test_dataloader = EmmentalDataLoader(
        task_to_label_dict={ATTRIBUTE: "labels"},
        dataset=FonduerDataset(
            ATTRIBUTE, test_cands[0], F_test[0], emb_layer.word2id, 2
        ),
        split="test",
        batch_size=100,
        shuffle=False,
    )

    test_preds = model.predict(test_dataloader, return_preds=True)
    positive = np.where(np.array(test_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.6)
    true_pred = [test_cands[0][_] for _ in positive[0]]
    test_results = entity_level_f1(true_pred, gold_file, ATTRIBUTE, test_docs, row_on=row_on, col_on=col_on)
    
    # Run on dev and train set for validation
    # We run the predictions also on our training and dev set, to validate that everything seems to work smoothly
    
    # Generate dataloader for dev data
    dev_dataloader = EmmentalDataLoader(
        task_to_label_dict={ATTRIBUTE: "labels"},
        dataset=FonduerDataset(
            ATTRIBUTE, dev_cands[0], F_dev[0], emb_layer.word2id, 2
        ),
        split="test",
        batch_size=100,
        shuffle=False,
    )


    dev_preds = model.predict(dev_dataloader, return_preds=True)
    positive_dev = np.where(np.array(dev_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.6)
    true_dev_pred = [dev_cands[0][_] for _ in positive_dev[0]]
    dev_results = entity_level_f1(true_dev_pred, gold_file, ATTRIBUTE, dev_docs, row_on=row_on, col_on=col_on)
    
    # Generate dataloader for train data
    train_dataloader = EmmentalDataLoader(
        task_to_label_dict={ATTRIBUTE: "labels"},
        dataset=FonduerDataset(
            ATTRIBUTE, train_cands[0], F_train[0], emb_layer.word2id, 2
        ),
        split="test",
        batch_size=100,
        shuffle=False,
    )


    train_preds = model.predict(train_dataloader, return_preds=True)
    positive_train = np.where(np.array(train_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.6)
    true_train_pred = [train_cands[0][_] for _ in positive_train[0]]
    train_results = entity_level_f1(true_train_pred, gold_file, ATTRIBUTE, train_docs, row_on=row_on, col_on=col_on)
        
    return [train_results, dev_results, test_results]


# ## Evaluating on the Test Set 
# 
# We keep the results from the rule-based approaches in mind.
# 
# ```
# ========================================
# Scoring on Entity-Level Gold Data for only ROW and TEST
# ========================================
# Corpus Precision 1.0
# Corpus Recall    0.993
# Corpus F1        0.996
# ----------------------------------------
# TP: 2301 | FP: 0 | FN: 17
# ========================================
# 
# 
# ========================================
# Scoring on Entity-Level Gold Data only COL and TEST
# ========================================
# Corpus Precision 0.98
# Corpus Recall    0.973
# Corpus F1        0.977
# ----------------------------------------
# TP: 3149 | FP: 64 | FN: 87
# ========================================
# ```
# 
# 
# * TOTAL DOCS PAIRWISE: Precision=0.9906678865507776, Recall=0.9821951535570818, F1=0.9864133263925039
# * TEST PAIRWISE: Precision=0.9883931809938339, Recall=0.981274756931941, F1=0.9848211058908565

# In[ ]:


# Build model and evaluate for rows
(row_model, row_emb_layer) = train_model(cands, F, "row")
row_results = eval_model(row_model, row_emb_layer, cands, F, "row")


# In[ ]:


# Build model and evaluate for columns
(col_model, col_emb_layer) = train_model(cands, F, "col")
col_results = eval_model(col_model, col_emb_layer, cands, F, "col")


# In[ ]:


(prec_test, rec_test, f1_test, prec_total, rec_total, f1_total) = summarize_results([row_results, col_results])
print(f"TOTAL DOCS PAIRWISE (LR): Precision={prec_total}, Recall={rec_total}, F1={f1_total}")
print(f"TEST PAIRWISE (LR): Precision={prec_test}, Recall={rec_test}, F1={f1_test}")


# We can see a much lower recall for the training set.
# This could be due to the 1-5 documents that have formatting issues in the gold standard and thus yield all candidates incorrect. E.g.
# 
# * C10067 (x not included and footnotes)
# * C10086 (Date formatting)
# * C10106 (Wrong goldset column headers in gold)
# * ...

# ## Compare to LSTM training
# 
# This will take much longer. Lets see how the Bi-LSTM performs, even though the number of training samples is very small (~15k-80k candidates)

# In[ ]:


# Build model and evaluate for rows
(row_model, row_emb_layer) = train_model(cands, F, "row", "LSTM" )
row_results = eval_model(row_model, row_emb_layer, cands, F, "row")


# In[ ]:


# Build model and evaluate for columns
(col_model, col_emb_layer) = train_model(cands, F, "col", "LSTM" )
col_results = eval_model(col_model, col_emb_layer, cands, F, "col")


# In[ ]:


(prec_test, rec_test, f1_test, prec_total, rec_total, f1_total) = summarize_results([row_results, col_results])
print(f"TOTAL DOCS PAIRWISE (LSTM): Precision={prec_total}, Recall={rec_total}, F1={f1_total}")
print(f"TEST PAIRWISE (LSTM): Precision={prec_test}, Recall={rec_test}, F1={f1_test}")


# # Phase 4: Error Analysis & Iterative KBC 
# 
# - Analyise the false positive (FP) and false negative (FN) candidates
# - Use the visualization tool to better understand the errors
# 
# We could theoretically improve on this data set by iterating over more pre-processing assumptions (e.g. formatting dates, x/X discard/keep, etc.).
# 

# In[ ]:




