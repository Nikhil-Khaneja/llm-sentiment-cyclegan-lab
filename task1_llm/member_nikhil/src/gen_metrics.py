"""Diversity / repetition metrics over generated text (word level)."""
import re

WORD = re.compile(r"[a-z']+")


def words(text):
    return WORD.findall(text.lower())


def ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(texts, n):
    """Unique n-grams / total n-grams, pooled over all samples."""
    all_ng = [g for t in texts for g in ngrams(words(t), n)]
    return len(set(all_ng)) / len(all_ng) if all_ng else 0.0


def repeated_4gram_rate(texts):
    """Share of 4-grams in a sample that already occurred earlier in the same sample,
    averaged over samples. 0 = no looping, close to 1 = stuck in a loop."""
    rates = []
    for t in texts:
        ng = ngrams(words(t), 4)
        if not ng:
            continue
        seen, rep = set(), 0
        for g in ng:
            rep += g in seen
            seen.add(g)
        rates.append(rep / len(ng))
    return sum(rates) / len(rates) if rates else 0.0
