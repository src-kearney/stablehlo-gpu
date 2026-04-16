"""
Live PII obfuscation demo — runs the full pipeline in Python.

Stages (in order, first hit wins):
  1. User table     — known names from identity provider (exact match)
  2. Bloom filter   — fast pre-screen before NER model
  3. NER model      — dslim/bert-base-NER for unknown names
  4. Obfuscate      — stable fake name mapping per session

Usage:
    python3 scripts/demo/obfuscate.py
    python3 scripts/demo/obfuscate.py --text "Call John Smith at Acme."
    python3 scripts/demo/obfuscate.py --mode placeholder
"""

import argparse
import hashlib
import math
import struct
import sys

# ---------------------------------------------------------------------------
# Bloom filter (pure Python, mirrors bloom_filter.cpp)
# ---------------------------------------------------------------------------

def _murmur3_32(key: bytes, seed: int) -> int:
    length = len(key)
    h = seed & 0xFFFFFFFF
    c1, c2 = 0xCC9E2D51, 0x1B873593
    for i in range(length // 4):
        k = struct.unpack_from("<I", key, i * 4)[0]
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    tail = key[length & ~3:]
    k = 0
    if len(tail) >= 3: k ^= tail[2] << 16
    if len(tail) >= 2: k ^= tail[1] << 8
    if len(tail) >= 1:
        k ^= tail[0]
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
    h ^= length
    h ^= h >> 16; h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13; h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


class BloomFilter:
    """Bloom filter over known user name tokens."""

    def __init__(self, capacity: int = 1 << 21, fp_rate: float = 0.001):
        self.capacity = capacity
        self.num_hashes = max(1, int((capacity / max(capacity // 8, 1)) * math.log(2)))
        self._bits = bytearray((capacity + 7) // 8)

    def _positions(self, token: str):
        b = token.lower().encode()
        base = _murmur3_32(b, 0)
        base_b = struct.pack("<I", base)
        for i in range(self.num_hashes):
            yield _murmur3_32(base_b, i) % self.capacity

    def insert(self, token: str):
        for pos in self._positions(token):
            self._bits[pos // 8] |= 1 << (pos % 8)

    def might_be_known_name(self, token: str) -> bool:
        return all(
            (self._bits[pos // 8] >> (pos % 8)) & 1
            for pos in self._positions(token)
        )


# ---------------------------------------------------------------------------
# User table — exact name → user_id mapping
# ---------------------------------------------------------------------------

class UserTable:
    def __init__(self):
        self._by_token:    dict[str, dict] = {}  # "sean" → user entry
        self._by_fullname: dict[str, dict] = {}  # "sean reilly" → user entry

    def add_user(self, user_id: str, first: str, last: str):
        entry = {"user_id": user_id, "first": first, "last": last,
                 "full": f"{first} {last}"}
        self._by_token[first.lower()] = entry
        self._by_token[last.lower()]  = entry
        self._by_fullname[f"{first} {last}".lower()] = entry

    def lookup(self, token: str) -> dict | None:
        return self._by_token.get(token.lower())

    def lookup_bigram(self, first: str, second: str) -> dict | None:
        return self._by_fullname.get(f"{first} {second}".lower())


# ---------------------------------------------------------------------------
# Stable fake name — deterministic from user_id
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Marcus","Elena","Tobias","Priya","Declan","Amara","Felix","Ingrid",
    "Rafael","Yuki","Callum","Nadia","Soren","Leila","Emre","Zara",
    "Hugo","Fatima","Luca","Astrid","Omar","Vera","Finn","Rania",
    "Idris","Cleo","Arjun","Mila","Cian","Sena",
]
LAST_NAMES = [
    "Okafor","Lindqvist","Reyes","Nakamura","Sullivan","Petrov","Mäkinen",
    "Ibrahim","Zhao","Ferreira","Kowalski","Onyekachi","Hartmann","Sato",
    "Diallo","Eriksson","Castellanos","Yilmaz","Bergmann","Osei",
]

def fake_name_for(user_id: str) -> str:
    """Deterministic fake name derived from user_id — globally stable."""
    h = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
    first = FIRST_NAMES[h % len(FIRST_NAMES)]
    last  = LAST_NAMES[(h >> 32) % len(LAST_NAMES)]
    return f"{first} {last}"


# ---------------------------------------------------------------------------
# Session — stable novel-name mapping within a session
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, session_id: str):
        self.id = session_id
        self._map: dict[str, str] = {}
        self._counter = int(hashlib.md5(session_id.encode()).hexdigest(), 16)

    def stable_fake(self, original: str) -> str:
        if original not in self._map:
            self._counter = (self._counter * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
            first = FIRST_NAMES[self._counter % len(FIRST_NAMES)]
            last  = LAST_NAMES[(self._counter >> 32) % len(LAST_NAMES)]
            self._map[original] = f"{first} {last}"
        return self._map[original]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ObfuscationPipeline:
    def __init__(self, users: list[dict], mode: str = "fakename"):
        self.mode = mode
        self.user_table = UserTable()
        self.bloom = BloomFilter()
        self.sessions: dict[str, Session] = {}
        self._ner = None  # lazy-loaded

        for u in users:
            self.user_table.add_user(u["user_id"], u["first"], u["last"])
            self.bloom.insert(u["first"])
            self.bloom.insert(u["last"])
            self.bloom.insert(f"{u['first']} {u['last']}")

    def _get_ner(self):
        if self._ner is None:
            print("Loading dslim/bert-base-NER ...", file=sys.stderr)
            from transformers import pipeline
            self._ner = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
            )
        return self._ner

    def _replacement(self, original: str, user_id: str | None,
                     session: Session) -> str:
        if self.mode == "placeholder":
            return "[PERSON]"
        if self.mode == "redact":
            return "***"
        # fakename mode
        if user_id:
            return fake_name_for(user_id)  # globally stable
        return session.stable_fake(original)  # session-stable for novel names

    def process(self, text: str, session_id: str = "default") -> dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id)
        session = self.sessions[session_id]

        words = text.split()
        result_words = list(words)
        detected = []
        used_model = False
        i = 0

        while i < len(words):
            token = words[i]
            clean = token.strip(".,!?;:\"'")

            # Stage 1 + 2: bloom → user table (bigram first, then single token)
            user_entry = None
            span_len = 1

            if i + 1 < len(words):
                next_clean = words[i + 1].strip(".,!?;:\"'")
                if self.bloom.might_be_known_name(clean) and \
                   self.bloom.might_be_known_name(next_clean):
                    user_entry = self.user_table.lookup_bigram(clean, next_clean)
                    if user_entry:
                        span_len = 2

            if not user_entry and self.bloom.might_be_known_name(clean):
                user_entry = self.user_table.lookup(clean)

            if user_entry:
                original = " ".join(words[i:i+span_len])
                replacement = self._replacement(
                    original, user_entry["user_id"], session)
                detected.append({
                    "text": original, "type": "PER",
                    "source": "user_table",
                    "replacement": replacement,
                })
                # Preserve trailing punctuation on last token of span
                trailing = words[i + span_len - 1][len(
                    words[i + span_len - 1].rstrip(".,!?;:\"'")):]
                result_words[i] = replacement + trailing
                for j in range(1, span_len):
                    result_words[i + j] = ""
                i += span_len
                continue

            i += 1

        # Stage 3+4: NER model for anything not caught by user table.
        # Exclude fake names already inserted by the user table so they
        # don't get obfuscated a second time.
        already_replaced = {d["replacement"] for d in detected}
        remaining_text = " ".join(w for w in result_words if w)
        if any(w and not w.startswith("[") and w != "***"
               for w in result_words):
            used_model = True
            ner = self._get_ner()
            ner_results = ner(remaining_text)
            for entity in ner_results:
                if entity["entity_group"] != "PER":
                    continue
                original = entity["word"]
                if original in already_replaced:
                    continue
                replacement = self._replacement(original, None, session)
                detected.append({
                    "text": original, "type": "PER",
                    "source": "ner_model",
                    "score": round(entity["score"], 3),
                    "replacement": replacement,
                })
                remaining_text = remaining_text.replace(original, replacement, 1)

        return {
            "original": text,
            "obfuscated": remaining_text,
            "detected": detected,
            "used_model": used_model,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DEMO_USERS = [
    {"user_id": "auth0|001", "first": "Sean",  "last": "Kearney"},
    {"user_id": "auth0|002", "first": "Alice",  "last": "Chen"},
    {"user_id": "auth0|003", "first": "Marcus", "last": "Webb"},
]

DEMO_TEXTS = [
    "Sean Kearney approved the deployment at 3pm.",
    "Please send the report to Alice Chen before Friday.",
    "Hi, I'm James Porter and I'd like to schedule a meeting.",
    "The invoice was reviewed by Marcus Webb and forwarded to Sarah Lin.",
    "Call me back when you get a chance — John.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Text to obfuscate (default: run demo)")
    parser.add_argument("--mode", choices=["fakename","placeholder","redact"],
                        default="fakename")
    parser.add_argument("--session", default="demo-session")
    args = parser.parse_args()

    pipeline = ObfuscationPipeline(DEMO_USERS, mode=args.mode)

    texts = [args.text] if args.text else DEMO_TEXTS

    print(f"\n{'─'*60}")
    print(f"  remora obfuscation demo  |  mode: {args.mode}")
    print(f"{'─'*60}\n")

    for text in texts:
        result = pipeline.process(text, session_id=args.session)
        print(f"IN  : {result['original']}")
        print(f"OUT : {result['obfuscated']}")
        for d in result["detected"]:
            src = d["source"]
            rep = d["replacement"]
            score = f"  score={d['score']}" if "score" in d else ""
            print(f"      [{d['type']}] \"{d['text']}\" → \"{rep}\"  ({src}{score})")
        print()


if __name__ == "__main__":
    main()
