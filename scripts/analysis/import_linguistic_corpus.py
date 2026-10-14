"""Compile a local CoNLL-U corpus into an auditable LinguaForge resource."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from speech_engine.linguistic_resources import CorpusLexicon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", required=True, help="Declared dataset license; review its terms before use")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lexicon = CorpusLexicon.from_conllu(args.input, args.lang, source_url=args.source_url, license=args.license)
    lexicon.save(args.output)
    print(json.dumps({"language": lexicon.lang, **lexicon.summary}, indent=2))


if __name__ == "__main__":
    main()
