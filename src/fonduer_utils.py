import emmental
import numpy as np

from fonduer import Meta
from emmental.modules.embedding_module import EmbeddingModule
from emmental.data import EmmentalDataLoader
from emmental.model import EmmentalModel
from emmental.learner import EmmentalLearner
from fonduer.learning.utils import collect_word_counter
from fonduer.learning.dataset import FonduerDataset
from fonduer.learning.task import create_task
from troy200_utils import entity_level_f1

ABSTAIN = -1
FALSE = 0
TRUE = 1

def get_methods(ATTRIBUTE, gold, gold_file, all_docs):
  train_docs = all_docs[0]
  dev_docs = all_docs[1]
  test_docs = all_docs[2]

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

  return (train_model, eval_model)