import re
from Strings import try_parse_int
from Files import keepable_tempfile
from AIMA import Struct

from InputTree import InputTree
verbs = re.compile(r'((AUX)|(VP)|(VB[GZDP]?))\s')
startswithtwoparens = re.compile(r'^\s*(\(\s*\()')

class UnbalancedParentheses(TypeError):
    pass
class NBestTreeFormatError(TypeError):
    pass

class NBestList(list):
    """Old and new attribute names: 
        sentence -> failed_sentence
        num -> sentence_id
        num_parses is now len()
        The actual object is now a list of parses, so the parses attr is
        no longer needed.
    Old names will still work for compatibility."""

    class Item(list):
        def __init__(self, reranker_score=None, parser_probability=None, 
                     parse=None):
            list.__init__(self)
            self.reranker_score = reranker_score
            self.parser_probability = parser_probability
            self.parse = parse

            # so we still look like a list
            for x in reranker_score, parser_probability, parse:
                if x is not None:
                    self.append(x)
        def __str__(self):
            return self.format(False)
        def format(self, force_parser_only=False):
            s = "%s\n%s" % (self.parser_probability, self.parse)
            if self.reranker_score and not force_parser_only:
                s = '%s\t%s' % (self.reranker_score, s)
            return s

    def __init__(self, failed_sentence=None, sentence_id=None, parses=None):
        list.__init__(self, parses)
        self.failed_sentence = failed_sentence
        self.failed = bool(self.failed_sentence)
        self.sentence_id = sentence_id
    def __str__(self):
        return self.format(False)
    def __getattr__(self, attr):
        # old attribute names
        if attr == 'sentence':
            return self.failed_sentence
        elif attr == 'num':
            return self.sentence_id
        elif attr == 'parses':
            return self
        elif attr == 'num_parses':
            return len(self)
        else:
            return list.__getattr__(self, attr)
    def contains_parse(self, parse, bleach_parses=True):
        """bleach_parses will remove whitespace before comparison."""
        try:
            index = self.index_of_parse(parse, bleach_parses)
            return True
        except IndexError:
            return False
    def index_of_parse(self, parse, bleach_parses=True):
        """bleach_parses will remove whitespace before comparison."""
        if bleach_parses:
            parse = strip_whitespace_from_tree(parse)
            parse = verbs.sub('V ', parse)
        for index, item in enumerate(self):
            itemparse = item.parse
            if bleach_parses:
                itemparse = strip_whitespace_from_tree(itemparse)
                itemparse = verbs.sub('V ', itemparse)
            if itemparse == parse:
                return index
        else:
            raise IndexError("Can't find a sentence with parse %r" % parse)
    def clone(self):
        return self.__class__(failed_sentence=self.failed_sentence,
                              sentence_id=self.sentence_id,
                              parses=self[:])
    def format(self, force_parser_only=False):
        """force_parser_only means only print parser probabilities
        (ignore reranker scores)"""
        s = '%d\t%s\n' % (len(self), self.sentence_id) 
        s += '\n'.join([parse.format(force_parser_only) for parse in self])
        s += '\n'
        return s

NBestTree = NBestList # old name


"""
reranked nbest-list looks like:

50 latwp950127.0021_14
3.80869 -67.1356
(S1 (S (PP (IN On) (NP (JJ great) (NNS fields))) (NP (NN something)) (VP (VBZ stays)) (. .)))
2.00742 -72.5027
(S1 (S (PP (IN On) (NP (NNP great) (NNPS fields))) (NP (NN something)) (VP (VBZ stays)) (. .)))
1.92863 -69.3684
(S1 (S (PP (IN On) (NP (JJ great) (NNS fields))) (NP (NN something)) (VP (NNS stays)) (. .)))
1.91697 -71.7111
...

non-reranked nbest-list looks like:
0       50
-50.4477
(S1 (FRAG (PRN (S (NP (PRP I)) (VP (VBP mean) (NP (DT that))))) (NP (DT that) (JJ particular) (NN case))))
-52.0303
(S1 (S (NP (PRP I)) (VP (VBP mean) (S (NP (DT that)) (NP (DT that) (JJ particular) (NN case))))))
-52.1324
(S1 (S (PRN (S (NP (PRP I)) (VP (VBP mean) (NP (DT that))))) (NP (DT that) (JJ particular) (NN case))))
-52.988
...
"""
def NBestListReader(stringiter, from_reranker=None):
    """Takes any thing that yields a list of strings (file object,
    FileInput object, itertools.chain of BZ2File objects, etc.) and
    yields NBestTree objects.  from_reranker indicates whether the n-best
    list came from the parser or reranker (True in the latter case).
    If from_reranker=None, it will attempt to autodetect the format,
    raising an NBestTreeFormatError if autodetection fails."""
    if from_reranker is None: # try to autodetect
        import itertools
        stringiter = iter(stringiter)
        try:
            line1, line2 = stringiter.next(), stringiter.next()
            # put the first two lines back in
            stringiter = itertools.chain((line1, line2), stringiter)

            # line 2 is either a log probability (non-reranked) or a reranker
            # score and a log probability (reranked)
            line2len = len(line2.split())
            assert line2len in (1, 2), \
                "2nd line of n-best list has %d items instead of 1 or 2" % \
                line2len
            from_reranker = (line2len == 2)
        except:
            raise NBestTreeFormatError("Couldn't autodetect format of the n-best list, please specify type with from_reranker=True or False")

    for line in stringiter:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Parse failed"):
            sentence = stringiter.next()
            yield NBestList(failed_sentence=sentence)
            continue

        # this was once different, but I'm assuming everyone has a modern
        # parser to simplify my design.  So sue me.
        num_parses, sentence_number = \
            [try_parse_int(x, x) for x in line.split()]

        parses = []
        for x in range(num_parses):
            # TODO: this could be simpler
            # extra score from reranker
            if from_reranker:
                scores = [float(x) for x in stringiter.next().split()]
                score_rerank, prob_ecparser = scores
                parse = stringiter.next().strip()
                parses.append(NBestList.Item(score_rerank, prob_ecparser, 
                                             parse))
            else:
                prob = float(stringiter.next())
                parse = stringiter.next().strip()
                parses.append(NBestList.Item(parser_probability=prob, 
                                             parse=parse))
        yield NBestList(sentence_id=sentence_number, 
                        parses=parses)
NBestTreeReader = NBestListReader # old name

leftparen = re.compile('\s*\(\s*')
rightparen = re.compile('\s*\)\s*')
def strip_whitespace_from_tree(tree):
    tree = rightparen.sub(')', leftparen.sub('(', tree))
    return tree

def TreeReader(stringiter, strip_whitespace=True, as_inputtrees=False):
    """Given a string iterator of trees (which may take up one or more
    lines), yields tree strings.  If strip_whitespace is True, the tree
    returned will only be on one line.  Otherwise, if it was originally
    pretty-printed, that will be maintained.  Thus, this iterator is
    useful for converting a text stream of trees to a distinct set of
    strings or de-pretty-printing trees (which is useful for evalb)."""
    buffer = ''
    for line in stringiter:
        if not line.strip():
            continue
        buffer += line
        matchedparens = buffer.count('(') == buffer.count(')')
        if buffer and matchedparens:
            if strip_whitespace:
                buffer = strip_whitespace_from_tree(buffer)
            if as_inputtrees:
                buffer = InputTree(buffer)
            yield buffer
            buffer = ''
    if buffer:
        raise UnbalancedParentheses("Remaining buffer: %r" % buffer)

# TODO this needs to go back into the Right Module (InputTree)
_replacements = {'-LRB-' : '(',
                 '-RRB-' : ')',
                 '-LSB-' : '[',
                 '-RSB-' : ']',
                 '-RCB-' : '}',
                 '-LCB-' : '{',
                 r'\/' : '/',
                 }
def getTreeYield(tree, replacements=_replacements):
    """This function is no longer necessary, due to the InputTree.getYield()
    method.  It is kept for backwards compatibility."""
    sentence = ' '.join(InputTree(tree).getYield())
    for old, new in replacements.items():
        sentence = sentence.replace(old, new)
    return sentence

def TreeConverter(stringiter, sgml=True, ident_prefix='stream', 
                  replacements=_replacements):
    """Convert a string iterator of trees to an iterator of the SGML
    sentences in them."""
    for count, tree in enumerate(TreeReader(stringiter, 
                                            strip_whitespace=False)):
        sentence = getTreeYield(tree, replacements=replacements)

        if sgml:
            sentence = "<s %s.%d> %s </s>\n" % (ident_prefix, count, sentence)
        yield sentence
del _replacements

def fix_top_node(tree):
    if startswithtwoparens.match(tree):
        return startswithtwoparens.sub('(S1 (', tree)
    else:
        return tree

def open_tree_object(obj):
    """obj can be a file, filename, TreeReader, or NBestList objects.
    Returns a temporary file object.  This is used for converting various
    formats into files for evaluation with evalb, etc."""
    if isinstance(obj, basestring):
        raise "broken"
        return file(obj, 'r')
    else:
        # reopen files readonly if necessary
        if isinstance(obj, file):
            if 'r' not in obj.mode:
                obj = file(obj.name, 'r')
        temp = keepable_tempfile(keep=True)
        for elt in obj:
            if isinstance(elt, basestring):
                s = elt
            elif isinstance(elt, NBestList):
                s = elt[0].parse
            else:
                raise TypeError("Unknown tree format: %r" % elt)
            s = fix_top_node(s)
            temp.write(s.strip() + "\n")
        temp.flush()
        return temp

if __name__ == "__main__":
    test_trees = """
    ( (S 
        (NP-SBJ 
          (NP (NNP Pierre) (NNP Vinken) )
          (, ,) 
          (ADJP 
            (NP (CD 61) (NNS years) )
            (JJ old) )
          (, ,) )
        (VP (MD will) 
          (VP (VB join) 
            (NP (DT the) (NN board) )
            (PP-CLR (IN as) 
              (NP (DT a) (JJ nonexecutive) (NN director) ))
            (NP-TMP (NNP Nov.) (CD 29) )))
        (. .) ))
    ( (S 
        (NP-SBJ (NNP Mr.) (NNP Vinken) )
        (VP (VBZ is) 
          (NP-PRD 
            (NP (NN chairman) )
            (PP (IN of) 
              (NP 
                (NP (NNP Elsevier) (NNP N.V.) )
                (, ,) 
                (NP (DT the) (NNP Dutch) (VBG publishing) (NN group) )))))
        (. .) ))
    """
    for tree in TreeConverter(test_trees.splitlines(True)):
        print tree
    for tree in TreeReader(test_trees.splitlines(True)):
        print repr(tree)
    for tree in TreeReader(test_trees.splitlines(True), strip_whitespace=False):
        print "+++"
        print tree
        print "---"

    from ECParser import parse, DEFAULT_DATA
    lines = parse(["this is a sentence",
                 "this is also a sentence",
                 "this is a fairly long sentence and this is the second half of that fairly long sentence and this would be the third half but there are only two halves."], mode='parser', datadir=DEFAULT_DATA, debug=False, nbest=2)
    print "real nbest list"
    print '\n'.join(lines)
    print "stringified NBestList objects"
    for forest in NBestTreeReader(lines):
        print forest
        print getTreeYield(forest[0].parse)
