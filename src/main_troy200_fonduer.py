# 1st Part
import argparse

# 2nd Part
import os
from fonduer import Meta, init_logging

# 3rd Part
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.parser.models import Document, Sentence
from fonduer.parser import Parser
from pprint import pprint

# 4th Path
from my_subclasses import get_subclasses
from fonduer.candidates import MentionExtractor 
from fonduer.candidates.models import Mention

# 5th Part
from fonduer.candidates import CandidateExtractor

# 6th Part
from troy200_utils import run_pairwise_eval, summarize_results, merge_candidates, entity_level_f1

# 7th Part
from fonduer.features import Featurizer
from fonduer.features.models import Feature

# 8th Part
from fonduer.supervision.models import GoldLabel
from fonduer.supervision import Labeler
from troy200_utils import get_gold_func

# 9th Part
from fonduer_utils import get_methods


# 1.) PARAMETERS
if __name__=='__main__':
    arg_parser = argparse.ArgumentParser(description='processing inputs.')
    arg_parser.add_argument('--docs', type=int, default=200,
                      help='the number of docs to load.')
    arg_parser.add_argument('--exp', type=str, default="gold",
                    help='the experiment to run. Any option of "pred", "gold", "norm" which includes ' +
                    'with predicted cell annoations, manually labelled cell annoations or no cell annotations respectively.')
    arg_parser.add_argument('--cls_methods', type=str, default="rule-based, logistic-regression, lstm",
                    help='delimited list of the classification methods. Defaults to "rule-based, logistic-regression, lstm"')
    arg_parser.add_argument('--clear_db', type=int, default=1,
                    help='if the database should be cleared (recalculate mentions, candidates and features). 1 = True, 0 = False, Default = 1')
    args = arg_parser.parse_args()

    print("\n#1 Start configure parameters")
    max_docs = args.docs # 50 # 200
    experiment = args.exp # ["pred", "gold", "norm"]
    cls_methods = [m for m in args.cls_methods.split(', ')]
    clear_db = True if args.clear_db == 1 else False
                            
    PARALLEL = 8 # 4  # assuming a quad-core machine
    ATTRIBUTE = f"troy200_col_row_data_{experiment}"

    DB_USERNAME = 'user'
    DB_PASSWORD = 'venron'
    conn_string = f'postgresql://{DB_USERNAME}:{DB_PASSWORD}@postgres:5432/{ATTRIBUTE}'

    folder = "pred" if experiment == "pred" else "gold"    
    docs_path = f"data/{folder}/html/"
    pdf_path = f"data/{folder}/pdf/"
    gold_file = 'data/troy200_gold.csv'


    # 2.) Cleanup and initialize
    print("\n#2 Cleanup and initialize")

    # Clear database
    if (clear_db):
        os.system(f"dropdb -h postgres --if-exists {ATTRIBUTE}")
    os.system(f"createdb -h postgres {ATTRIBUTE}")

    # Configure logging for Fonduer
    init_logging(log_dir=f"logs_{ATTRIBUTE}")
    session = Meta.init(conn_string).Session()


    # 3.) Process documents into train,dev,test
    print("\n#3 Process Document into train, dev, test sets")

    # parse documents
    has_documents = session.query(Document).count() > 0
    corpus_parser = Parser(session, structural=True, lingual=True, visual=True, pdf_path=pdf_path)
    if (not has_documents): 
        doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
        corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
        
    print(f"Documents: {session.query(Document).count()}")
    print(f"Sentences: {session.query(Sentence).count()}")

    # split documents
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

    # 4.) Mention Extraction
    print("\n#4 Mention extraction")

    (mention_classes, mention_spaces, matchers, candidate_classes, throttlers) = get_subclasses(experiment)

    hasMentions = session.query(Mention).count() > 0
    if (not hasMentions):
        mention_extractor = MentionExtractor(
            session, mention_classes,  mention_spaces, matchers
        )
        docs = session.query(Document).order_by(Document.name).all()
        mention_extractor.apply(docs, parallelism=PARALLEL)
        mentions = session.query(Mention).all()
        print(f"Total Mentions: {len(mentions)}")

    # 5.) Candidate Extraction
    print("\n#5 Candidate extraction")
    RowCandidate = candidate_classes[0]
    ColCandidate = candidate_classes[1]
    row_filter = throttlers[0]
    col_filter = throttlers[1]
    has_candidates = (
        session.query(RowCandidate).filter(RowCandidate.split == 0).count() > 0 or
        session.query(ColCandidate).filter(ColCandidate.split == 0).count() > 0
    )

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

    # 6.) Rule-based evaluation
    if ("rule-based" in cls_methods):
        print("\n#6 Rule-based classification")
        row_results_train = run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, 'row', 0)
        col_results_train = run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, 'col', 0)

        row_results_dev = run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, 'row', 1)
        col_results_dev = run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, 'col', 1)

        row_results_test = run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, 'row', 2)
        col_results_test = run_pairwise_eval(gold_file, ATTRIBUTE, cands, all_docs, 'col', 2)

        (prec_test, rec_test, f1_test, prec_total, rec_total, f1_total) = summarize_results(
        [
            [row_results_train, row_results_dev, row_results_test],
            [col_results_train, col_results_dev, col_results_test]
        ])
        print(f"TOTAL DOCS PAIRWISE: Precision={prec_total}, Recall={rec_total}, F1={f1_total}")
        print(f"TEST PAIRWISE: Precision={prec_test}, Recall={rec_test}, F1={f1_test}")

        cands_merged = merge_candidates(cands[0][2][0], cands[1][2][0])
        (TP, FP, FN) = entity_level_f1(cands_merged, gold_file, ATTRIBUTE, test_docs, row_on=True, col_on=True)


    # 7.) Featurize candidates
    has_features = session.query(Feature).count() > 0
    print(f"\n#7 Candidate featurization ({not has_features})")
    # Features for row/column candidates (train, dev, test)
    F = []

    for i, cands_align in enumerate(cands):
        featurizer = Featurizer(session, [RowCandidate]) if i == 0 else Featurizer(session, [ColCandidate])
        train_cands = cands_align[0]
        dev_cands = cands_align[1]
        test_cands = cands_align[2]

        if (not has_features):
            # Training set
            featurizer.apply(split=0, train=True, parallelism=PARALLEL)
            F_train = featurizer.get_feature_matrices(train_cands)
            print(F_train[0].shape)

            # Dev set
            featurizer.apply(split=1, parallelism=PARALLEL)
            F_dev = featurizer.get_feature_matrices(dev_cands)
            print(F_dev[0].shape)

            # Test set
            featurizer.apply(split=2, parallelism=PARALLEL)
            F_test = featurizer.get_feature_matrices(test_cands)
            print(F_test[0].shape)
        else:
            F_train = featurizer.get_feature_matrices(train_cands)
            F_dev = featurizer.get_feature_matrices(dev_cands)
            F_test = featurizer.get_feature_matrices(test_cands)
        # Summarize for row/col
        F.append([F_train, F_dev, F_test])

    # 8.) Load gold data
    print("\n#8 Load Gold Data")
    # 1.1) Load the gold data rows
    gold_row = get_gold_func(gold_file, row_on=True, col_on=False)
    docs = corpus_parser.get_documents()
    labeler = Labeler(session, [RowCandidate])
    labeler.apply(docs=docs, lfs=[[gold_row]], table=GoldLabel, train=True, parallelism=PARALLEL)

    # 1.2) Load the gold data cols
    gold_col = get_gold_func(gold_file, row_on=False, col_on=True)
    docs = corpus_parser.get_documents()
    labeler = Labeler(session, [ColCandidate])
    labeler.apply(docs=docs, lfs=[[gold_col]], table=GoldLabel, train=True, parallelism=PARALLEL)
    gold = [gold_row, gold_col]

    # 9.) Supervised classification (Logistic Regression)
    (train_model, eval_model) = get_methods(ATTRIBUTE, gold, gold_file, all_docs)
    if ("logistic-regression" in cls_methods):
        print("\n#9 Train and classify with Logistic Regression")

        # Build model and evaluate for rows
        (row_model, row_emb_layer) = train_model(cands, F, "row")
        row_results = eval_model(row_model, row_emb_layer, cands, F, "row")

        # Build model and evaluate for columns
        (col_model, col_emb_layer) = train_model(cands, F, "col")
        col_results = eval_model(col_model, col_emb_layer, cands, F, "col")

        (prec_test, rec_test, f1_test, prec_total, rec_total, f1_total) = summarize_results([row_results, col_results])
        print(f"TOTAL DOCS PAIRWISE (LR): Precision={prec_total}, Recall={rec_total}, F1={f1_total}")
        print(f"TEST PAIRWISE (LR): Precision={prec_test}, Recall={rec_test}, F1={f1_test}")

    # 9.) Supervised classification (LSTM)

    if ("lstm" in cls_methods):
        print("\n#10 Train and classify with LSTM")

        # Build model and evaluate for rows
        (row_model, row_emb_layer) = train_model(cands, F, "row", "LSTM" )
        row_results = eval_model(row_model, row_emb_layer, cands, F, "row")

        # Build model and evaluate for columns
        (col_model, col_emb_layer) = train_model(cands, F, "col", "LSTM" )
        col_results = eval_model(col_model, col_emb_layer, cands, F, "col")

        (prec_test, rec_test, f1_test, prec_total, rec_total, f1_total) = summarize_results([row_results, col_results])
        print(f"TOTAL DOCS PAIRWISE (LSTM): Precision={prec_total}, Recall={rec_total}, F1={f1_total}")
        print(f"TEST PAIRWISE (LSTM): Precision={prec_test}, Recall={rec_test}, F1={f1_test}")
