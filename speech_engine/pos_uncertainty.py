"""Local uncertainty algorithms; model marginals are not correctness rates."""
import math
import json
from pathlib import Path
import re
from hashlib import sha256

from speech_engine.linguistic_resources import language_code

SCORING_VERSION = 'hmm_shared_sentence_boundaries_v4'


def pos_model_fingerprint(transitions, emissions):
    payload = [SCORING_VERSION, sorted((left, right, count) for (left, right), count in transitions.items()),
               sorted((form, sorted(counts.items())) for form, counts in emissions.items())]
    return sha256(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def pos_review_recommended(evidence):
    """Return a POS-specific review hint only when supported by calibration."""
    if not evidence or not evidence.get('applies_to_returned_pos'):
        return None
    calibration = evidence.get('calibration')
    if not calibration or calibration.get('outside_observed_bins'):
        return None
    return calibration['accuracy_estimate'] < 0.8


def logsumexp(values):
    values = list(values)
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def forward_backward(lattice, transitions):
    """Normalize all permitted paths in a log-space tag lattice, including EOS."""
    if not lattice:
        return []
    forward = []
    previous = {'BOS': 0.0}
    for emissions in lattice:
        current = {tag: emission + logsumexp(score + transitions[(left, tag)]
                                             for left, score in previous.items())
                   for tag, emission in emissions.items()}
        forward.append(current)
        previous = current
    backward = [None] * len(lattice)
    backward[-1] = {tag: transitions[(tag, 'EOS')] for tag in lattice[-1]}
    for index in range(len(lattice) - 2, -1, -1):
        backward[index] = {
            tag: logsumexp(transitions[(tag, right)] + emission + backward[index + 1][right]
                           for right, emission in lattice[index + 1].items())
            for tag in lattice[index]}
    result = []
    for left, right in zip(forward, backward):
        scores = {tag: score + right[tag] for tag, score in left.items()}
        normalizer = logsumexp(scores.values())
        result.append({tag: math.exp(score - normalizer) for tag, score in scores.items()})
    return result


class POSCalibration:
    """Binned isotonic top-label calibration fitted locally on development data.

    Seen/unseen forms have separate reliability curves. These empirical estimates
    apply to POS only and do not establish calibration on another domain.
    """
    def __init__(self, language, training_sha256, development_sha256, model_sha256, bins):
        self.language = language_code(language)
        if (not all(isinstance(value, str) and re.fullmatch(r'[a-f0-9]{64}', value)
                    for value in (training_sha256, development_sha256, model_sha256))
                or training_sha256 == development_sha256):
            raise ValueError('Calibration needs distinct training/development checksums')
        if not isinstance(bins, dict) or set(bins) != {'known', 'unknown'}:
            raise ValueError('Calibration needs known and unknown score bins')
        self.training_sha256 = training_sha256
        self.development_sha256 = development_sha256
        self.model_sha256 = model_sha256
        self.bins = {}
        self.blocks = {}
        for group, counts in bins.items():
            if (not isinstance(counts, list) or len(counts) != 20
                    or any(not isinstance(pair, list) or len(pair) != 2
                           or any(type(value) is not int for value in pair)
                           or not 0 <= pair[1] <= pair[0] for pair in counts)):
                raise ValueError('Invalid calibration counts')
            self.bins[group] = [pair[:] for pair in counts]
            blocks = []
            for index, (total, correct) in enumerate(counts):
                if not total:
                    continue
                blocks.append([index, index, total, correct])
                # Pool adjacent violators: empirical accuracy must be monotonic.
                while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
                    right = blocks.pop()
                    left = blocks.pop()
                    blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
            self.blocks[group] = blocks
        if not any(self.blocks.values()):
            raise ValueError('Calibration needs observations')

    @staticmethod
    def score_bin(score):
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Model score must be finite and between zero and one')
        return min(19, int(score * 20))

    @classmethod
    def fit(cls, rows, language, training_sha256, development_sha256, model_sha256):
        bins = {group: [[0, 0] for _ in range(20)] for group in ('known', 'unknown')}
        for score, known, correct in rows:
            if type(known) is not bool or type(correct) is not bool:
                raise ValueError('Calibration labels must be booleans')
            pair = bins['known' if known else 'unknown'][cls.score_bin(score)]
            pair[0] += 1
            pair[1] += correct
        return cls(language, training_sha256, development_sha256, model_sha256, bins)

    def estimate(self, score, known):
        index = self.score_bin(score)
        blocks = self.blocks['known' if known else 'unknown']
        if not blocks:
            return None
        block = next((block for block in blocks if block[1] >= index), blocks[-1])
        return {'accuracy_estimate': block[3] / block[2], 'development_observations': block[2],
                'score_bin_range': [block[0], block[1]],
                'outside_observed_bins': not block[0] <= index <= block[1],
                'development_sha256': self.development_sha256,
                'scope': 'POS conditional on correct token boundaries; gold-word development data',
                'training_sha256': self.training_sha256,
                'model_sha256': self.model_sha256,
                'method': 'binned_isotonic_seen_unseen'}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'schema_version': 1, 'language': self.language,
                                   'scoring_version': SCORING_VERSION,
                                   'training_sha256': self.training_sha256,
                                   'development_sha256': self.development_sha256,
                                   'model_sha256': self.model_sha256,
                                   'bins': self.bins}, indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if (not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1
                or data.get('scoring_version') != SCORING_VERSION):
            raise ValueError('Unsupported calibration schema')
        return cls(data.get('language'), data.get('training_sha256'), data.get('development_sha256'), data.get('model_sha256'), data.get('bins'))
