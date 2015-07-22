from path import path

class Terms:
    def __init__(self, datadir):
        self.filename = path(datadir) / 'terms.txt'
        self.terms_list = []    # e.g. 0 -> AUX, 1 -> AUXG
        self.terms_to_type = {} # e.g. NN -> 2

        lineprocessor_skip_whitespace(file(self.filename), self.parse_term)
    def parse_term(self, line):
        name, term_type = line.split()
        term_type = int(term_type)
        self.terms_list.append(name)
        self.terms_to_type[name] = term_type
    def __getitem__(self, index):
        return self.terms_list[index]

def lineprocessor_skip_whitespace(stream, func):
    for line in stream:
        line = line.strip()
        if not line:
            continue
        func(line)

class Vocabulary:
    def __init__(self, datadir):
        self.filename = path(datadir) / 'pSgT.txt'
        self.terms = Terms(datadir)
        self.words = {} # word text : Word object

        self.parse_pSgT()
    def parse_pSgT(self):
        pSgT = file(self.filename)
        for line in pSgT:
            line = line.strip()
            if not line:
                continue
            if len(line.split()) == 1:
                vocabsize = int(line)
                break # after we see vocabsize, we look for words
        lineprocessor_skip_whitespace(pSgT, self.read_word)
        assert len(self) == vocabsize
    def read_word(self, line):
        word, rest = line.split('\t')
        freq = None
        conditional_probs = {}
        from iterextras import batch
        for (first, second) in batch(rest.split()):
            if first == '|':
                freq = int(second)
            else:
                termint = int(first)
                prob = float(second)
                termname = self.terms[termint]
                conditional_probs[termname] = prob
        w = Word(word, conditional_probs, freq)
        self.words[word] = w
    def __iter__(self):
        for word in self.words.values():
            yield word
    def __len__(self):
        return len(self.words)
    def __getitem__(self, word):
        return self.words[word]

class Word:
    def __init__(self, word, conditional_probs, total_count):
        self.word = word
        self.conditional_probs = conditional_probs
        self.total_count = total_count
    def __str__(self):
        return self.word

if __name__ == "__main__":
    p = Vocabulary('/u/dmcc/res/ntc/models/add-0')
    p2 = Vocabulary('/u/dmcc/res/ntc/models/wsj-x5-ntc-35')
    pos_per_word = 0.0
    pos_per_word2 = 0.0
    count = 0
    for word in p:
        if word.total_count > 10:
            count += 1
            pos_per_word += len(word.conditional_probs)
            pos_per_word2 += len(p2[str(word)].conditional_probs)
    print pos_per_word / count
    print pos_per_word2 / count
