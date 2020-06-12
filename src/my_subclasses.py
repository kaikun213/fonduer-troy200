# Mentions
from fonduer.candidates.models import mention_subclass
from fonduer.candidates.matchers import RegexMatchSpan, LambdaFunctionMatcher, Intersect, Union
from fonduer.utils.data_model_utils.structural import _get_node
from fonduer.candidates import MentionNgrams, MentionSentences
# Candidates
from fonduer.candidates.models import candidate_subclass
from fonduer.utils.data_model_utils import *
from fonduer.utils.utils_table import is_row_aligned, is_col_aligned
import re


def get_label_matcher(label, experiment):
    def label_matcher(mention):
        
      # Assume no annotations in normal Fonduer (arbitrary cells in a single large HTML table)
      # Header cells etc. of each table are not identified in the large table
      if (experiment == "norm"):
          return True
      
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

# DefineThrottlers
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

def get_subclasses(experiment):
  # 1.) Mention subclasses
  Data = mention_subclass("Data")
  Row = mention_subclass("Row")
  Col = mention_subclass("Col")

  # 2.) Mention spaces
  data_ngrams = MentionSentences() # MentionNgrams(n_max=3)
  row_ngrams = MentionSentences() # MentionNgrams(n_min=1, n_max=8)
  col_ngrams = MentionSentences() # MentionNgrams(n_min=1, n_max=8)

  # 3.) Matchers
  data_regex_matcher = RegexMatchSpan(rgx=r"[0-9-,.%$#]+( to | )?[0-9-,.%$#]*|^x$", longest_match_only=True)
  data_label_matcher = LambdaFunctionMatcher(func=get_label_matcher("Data", experiment))
  data_matcher = Intersect(data_regex_matcher, data_label_matcher)
  row_regex_matcher = RegexMatchSpan(rgx=r"^.*$", longest_match_only=True)
  row_label_matcher = LambdaFunctionMatcher(func=get_label_matcher("Header", experiment))
  row_matcher = Intersect(row_regex_matcher, row_label_matcher)
  col_regex_matcher = RegexMatchSpan(rgx=r"^.*$", longest_match_only=True)
  col_label_matcher = LambdaFunctionMatcher(func=get_label_matcher("Header", experiment))
  col_matcher = Intersect(col_regex_matcher, col_label_matcher)

  # 4.) Candidate classes
  RowCandidate = candidate_subclass("RowCandidate", [Data, Row])
  ColCandidate = candidate_subclass("ColCandidate", [Data, Col])

  # 5.) Throttlers
  mention_classes = [Data, Row, Col]
  mention_spaces = [data_ngrams, row_ngrams, col_ngrams]
  matchers = [data_matcher, row_matcher, col_matcher]
  candidate_classes = [RowCandidate, ColCandidate]
  throttlers = [row_filter, col_filter]

  return (mention_classes, mention_spaces, matchers, candidate_classes, throttlers)