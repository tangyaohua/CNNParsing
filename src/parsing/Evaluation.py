from __future__ import division

class Evaluator:
    """Abstract parser evaluator class.  Subclasses perform various
    operations for some evaluation program (e.g. evalb, sparseval,
    etc.)."""
    def evaluate(self, test, gold):
        """Return the text from running this evaluator, comparing test
        trees against gold trees.  Implementors should allow test and
        gold to be in any format understood by open_tree_object."""
    def fscore_extractor(self, text):
        """Given output from the evaluator, return the f-scores as floats
        between 0.0 and 1.0"""
    def oracle_score_extractor(self, text):
        """Return a list of oracle scores (f-scores between 0.0 and 1.0)"""
    def per_line_stats_extractor(self, text):
        """Returns an iterable (list, iterator, etc.) of dictionaries.  Each
        dictionary will have information about a specific line, specified
        by the key 'id'.  Possible values include precision, recall,
        matchedbracket, goldbracket, testbracket, fscore, correcttags,
        tagaccuracy, len."""
    def summary_extractor(self, text):
        """Return a dictionary with summary information.  Different evaluators
        provide different information, so this method is almost completely
        type unsafe."""

def calc_fscore(precision, recall):
    """Calculate f-score from precision and recall numbers."""
    num = 2 * precision * recall
    denom = precision + recall
    if denom == 0:
        return 0.0
    else:
        return num / denom

def calc_precision_recall(matchedbracket, goldbracket, testbracket):
    """Calculates precision and recall from the number of matched brackets,
    gold brackets, and test brackets.  Returns numbers between 0.0 and 1.0."""
    if goldbracket > 0:
        recall = matchedbracket / goldbracket
    else:
        recall = 0.0

    if testbracket > 0:
        precision = matchedbracket / testbracket
    else:
        precision = 0.0

    return precision, recall

