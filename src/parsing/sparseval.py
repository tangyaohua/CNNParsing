from __future__ import division
import re
from Parsing.Evaluation import Evaluator, calc_precision_recall, calc_fscore

sparseval_fmeasure_re = \
    re.compile(r'Labeled Bracketing F-measure:\s+(\d+\.\d+)')
class sparseval(Evaluator):
    def fscore_extractor(self, text):
        (all_sents,) = sparseval_fmeasure_re.findall(self._process_output)
        return float(all_sents) / 100.0
