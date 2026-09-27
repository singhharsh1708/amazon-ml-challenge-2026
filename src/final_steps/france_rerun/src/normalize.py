"""Copy of a src/ module as used by the France re-run (fixed France normalization). Experiment/validation script normalize.py for this step.

Step context: Copy of a src/ module as used by the France re-run (fixed France normalization). It is put ahead of src/ on sys.path by the France re-run scripts.
"""
import json
import re

from anyascii import anyascii

from config import DATA_DIR

NON_ALNUM = re.compile(r"[^a-z0-9]+")
DOMAIN = re.compile(
    r"\b(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|in|co\.in|fr|biz|info|co|us)\b"
)
DBA = re.compile(r"\b(?:dba|doing business as)\b")
LEET = str.maketrans("013456789", "oleasgtbg")

NAME_STOP = {
    "the", "and", "of", "et",
    "pvt", "private", "ltd", "limited", "llc", "llp", "lp", "inc", "incorporated",
    "corp", "corporation", "co", "company", "plc", "pllc", "pc",
    "ms", "shri", "sri", "services", "service", "center", "centre", "group",
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "gmbh",
    "praivet", "praivhet", "pra", "li", "limitedd",
    "limitet", "praibhet", "elelpi", "piraivet", "limirrd", "praivrr",
    "de", "du", "des", "la", "le", "les",
}

TRANSLIT_FILE = DATA_DIR / "translit_map.json"
TRANSLIT = json.loads(TRANSLIT_FILE.read_text()) if TRANSLIT_FILE.exists() else {}

ADDRESS_STOP = {"none", "null", "nan", "no", "h", "hn", "door", "nos", "c", "o"}

ADDRESS_ABBREV = {
    "road": "rd", "street": "st", "avenue": "ave", "av": "ave", "boulevard": "blvd",
    "bd": "blvd", "drive": "dr", "lane": "ln", "parkway": "pkwy", "highway": "hwy",
    "court": "ct", "place": "pl", "square": "sq", "circle": "cir", "terrace": "ter",
    "trail": "trl", "north": "n", "south": "s", "east": "e", "west": "w",
    "suite": "ste", "apartment": "apt", "floor": "flr", "building": "bldg",
    "nagar": "ngr", "colony": "cly", "coly": "cly", "sector": "sec", "mount": "mt",
    "fort": "ft", "saint": "st", "sainte": "ste", "rue": "r", "allee": "all",
    "impasse": "imp", "chemin": "ch", "route": "rte", "county": "", "city": "",
    "district": "", "dist": "",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
}

STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "district of columbia": "dc", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}

INDIA_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la", "puducherry": "py",
    "pondicherry": "py", "chandigarh": "ch", "tamilnadu": "tn",
    "mharastr": "mh", "dilli": "dl", "uttr prdes": "up", "krnatk": "ka",
    "tmilnatu": "tn", "pscimbng": "wb", "gujrat": "gj", "telmgan": "tg",
    "hriyana": "hr", "rajsthan": "rj", "kerlm": "kl", "mdhy prdes": "mp",
    "amdhrprdes": "ap", "od isa": "od", "pmjab": "pb",
}


def _phrase_pattern(mapping):
    """Compile one regex that matches any phrase in the mapping, longest first."""
    keys = sorted(mapping, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")


STATE_PATTERNS = {
    "us": (_phrase_pattern(STATES), STATES),
    "india": (_phrase_pattern(INDIA_STATES), INDIA_STATES),
}


def to_ascii(text):
    """Transliterate text to lowercase ASCII."""
    if text is None:
        return ""
    text = str(text)
    if not text.isascii():
        text = anyascii(text)
    return text.lower()


def _deleet(token):
    """Undo digit-for-letter substitutions in a token."""
    if token.isalpha() or token.isdigit() or len(token) < 4:
        return token
    letters = sum(ch.isalpha() for ch in token)
    if letters < len(token) - 2:
        return token
    return token.translate(LEET)


def _dedupe(tokens):
    """Drop repeated tokens, keeping the first occurrence."""
    out = []
    for token in tokens:
        if not out or out[-1] != token:
            out.append(token)
    return out


def normalize_name(raw, use_map=True):
    """Return the normalized full name and core name of a business."""
    transliterated = raw is not None and not str(raw).isascii()
    text = to_ascii(raw)
    if text in ("", "none", "null", "nan"):
        return "", ""
    text = text.replace("&", " and ")
    text = DOMAIN.sub(r" \1 ", text)
    text = text.replace(".", "")
    parts = [p for p in DBA.split(text) if p.strip(" :-")]
    if len(parts) > 1:
        text = parts[-1]
    tokens = [_deleet(t) for t in NON_ALNUM.sub(" ", text).split()]
    if transliterated and use_map and TRANSLIT:
        tokens = [TRANSLIT.get(t, t) for t in tokens]
    tokens = _dedupe(tokens)
    core = [t for t in tokens if t not in NAME_STOP and len(t) > 1]
    if not core:
        core = tokens
    return " ".join(tokens), " ".join(core)


def normalize_address(raw, country=""):
    """Return the normalized address for a country."""
    text = to_ascii(raw)
    if text in ("", "none", "null", "nan"):
        return ""
    text = text.replace("&", " and ")
    text = NON_ALNUM.sub(" ", text)
    pattern = STATE_PATTERNS.get((country or "").lower())
    if pattern is not None:
        regex, mapping = pattern
        text = regex.sub(lambda m: mapping[m.group(1)], text)
    tokens = []
    for token in text.split():
        if token in ADDRESS_STOP:
            continue
        if token.isdigit():
            token = token.lstrip("0") or "0"
        token = ADDRESS_ABBREV.get(token, token)
        if token:
            tokens.append(token)
    return " ".join(tokens)
