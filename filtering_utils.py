import os

import nltk
import pandas as pd
import spacy

# load spacy model
NLP = spacy.load('en_core_web_sm')


def tokenize(s):
    """Tokenize"""
    doc = NLP(s)
    return [tok.text for tok in doc]


def get_key_idx(tokens, key):
    """Get key index from in the tokens"""
    assert key in tokens, f'{key} not found in {tokens}'
    return tokens.index(key)


def load_synonym_pairs():
    """Load synonyms from file"""
    df_syn = pd.read_csv(
        filepath_or_buffer=os.path.join(
            '../data',
            'SynonymsFromWordNet.csv',
            'SynonymsFromWordNet.csv'
        ),
        encoding='utf-8',
        header=None,
        names=['synsetID', 'pos', 'synonym1', 'synonym2']
    )
    synonym1_list = df_syn['synonym1'].to_list()
    synonym2_list = df_syn['synonym2'].to_list()

    # sanity check
    assert len(synonym1_list) == len(synonym2_list)

    syn_pairs = []
    for syn1, syn2 in zip(synonym1_list, synonym2_list):
        syn_pairs.append((syn1, syn2))

    # sanity check
    assert len(syn_pairs) == len(synonym1_list)
    syn_pairs = set(syn_pairs)
    return syn_pairs


def char_ngrams(word, n):
    """Create char ngrams from word"""
    for idx in range(len(word) - n + 1):
        yield tuple(word[idx:idx + n])


def filter_lemmasyn_ngrammatch(li, sol, syn_pairs, verbose=False):
    """Lemmatized synonyms and 4-gram overlap filtering"""
    fli = list(filter(lambda x: (NLP(x)[0].lemma_, NLP(sol)[0].lemma_) not in syn_pairs and (
        NLP(sol)[0].lemma_, NLP(x)[0].lemma_) not in syn_pairs, li))
    if verbose and len(fli) != len(li):
        print('[SYN]\t', sol, set(li) - set(fli), end='\t')
    temp_li = fli.copy()
    # ngram match
    fli = list(filter(lambda x: len(set(char_ngrams(x, 4)) & set(char_ngrams(sol, 4))) == 0, fli))
    if verbose and len(fli) != len(temp_li):
        print('[NGRAM]\t', sol, set(temp_li) - set(fli), end='\t')
    return fli


def filter_same_pos(li, tokens, solidx, verbose=False):
    """Same POS filtering"""
    sol_pos_tag = nltk.pos_tag(tokens)[solidx][1]
    if sol_pos_tag in ['VBD', 'VBN', 'JJ']:
        sol_pos_tag = 'VBD_VBN_JJ'
    fli = []
    for w in li:
        tokens_copy = tokens.copy()
        tokens_copy[solidx] = w
        cand_pos_tag = nltk.pos_tag(tokens_copy)[solidx][1]
        if cand_pos_tag in ['VBD', 'VBN', 'JJ']:
            cand_pos_tag = 'VBD_VBN_JJ'
        if sol_pos_tag == cand_pos_tag:
            fli.append(w)
    if verbose and len(fli) != len(li):
        print('[POS]\t', tokens[solidx], set(li) - set(fli), end='\t')

    return fli


def filter_out_lemmasyn_ngrammatch_pos(li, sol, syn_pairs, sent_tokens, sol_idx, verbose=False):
    # filter out (lemmatized) synonyms, n-gram overlap and different POS

    assert sent_tokens[sol_idx] == sol

    # (lemmatized) synonyms and 4-gram overlap
    fli = filter_lemmasyn_ngrammatch(li, sol, syn_pairs, verbose=verbose)

    # POS
    fli = filter_same_pos(fli, sent_tokens, sol_idx, verbose=verbose)
    return fli


if __name__ == "__main__":
    SYNONYMS_PAIRS = load_synonym_pairs()

    question = "If you read Anna's comments in ______, you'll see there wasn't anything wrong with them."
    key = 'context'
    tokens = tokenize(question.replace('______', key))
    key_idx = get_key_idx(tokens, key)

    candidate_distractors_before_filtering = [
        'contexts', 'perspective', 'contextualization', 'reference', 'relation', 'relevant', 'situation', 'text',
        'sort', 'account', 'background', 'case', 'detail', 'mind', 'standpoint', 'topic', 'something', 'issue',
        'contemplation', 'subject', 'relating', 'respect', 'link', 'concurrence', 'connection', 'time', 'way', 'look',
        'message', 'conjunction', 'book', 'interface', 'glance', 'relationship', 'project', 'combination', 'hand',
        'find', 'website', 'line', 'na', 'association', 'anna', 'scratch'
    ]

    candidate_distractors_after_filtering = filter_out_lemmasyn_ngrammatch_pos(
        li=candidate_distractors_before_filtering,
        sol=key,
        syn_pairs=SYNONYMS_PAIRS,
        sent_tokens=tokens,
        sol_idx=key_idx,
        verbose=True
    )

    print('\nQuestion:', question)
    print('Key:', key)
    print('Candidates:')
    print('\tBefore filtering:', candidate_distractors_before_filtering)
    print('\tAfter filtering:', candidate_distractors_after_filtering)
