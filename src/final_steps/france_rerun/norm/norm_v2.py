"""Fixed French address normalization (address_v2) used by the France re-run.. Experiment/validation script norm_v2.py for this step.

Step context: Fixed French address normalization (address_v2) used by the France re-run.
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import re
import sys

sys.path.insert(0, f"{REPO_DIR}/src")
import normalize as N

FR_REGION = {
    "hauts de france", "nouvelle aquitaine", "pays de la loire",
    "nord", "pas de calais", "gironde", "loire atlantique", "france",
}
NUM_SIGN = re.compile(r"\b[Nn]\s*[°º]\s*")
ELISION = re.compile(r"\b(l|d|qu|j|n|s|m|t)['’`´]\s*", re.I)
NUM_SUFFIX = re.compile(r"\b(\d+)(bis|ter|quater|[a-df-z])\b")
AFTER_NUM = {"b": "bis", "t": "ter"}
FR_ABBREV = {
    "crs": "cours", "q": "quai", "psg": "passage", "pass": "passage",
    "res": "residence", "appartement": "apt", "appt": "apt", "app": "apt",
    "bld": "blvd", "bvd": "blvd", "boul": "blvd", "esp": "esplanade",
    "prom": "promenade", "gde": "grande", "ham": "hameau", "dom": "domaine",
}
BRACKET_FRANCE = re.compile(r"[\(\[]\s*france\s*[\)\]]")
FR_NAME_MAP = {"et": "and", "frs": "freres", "cie": "compagnie", "st": "saint", "ste": "sainte"}
FR_NAME_STOP = {"ei", "eirl", "france"}
FR_NAME_STOP_TRADE = {"fils", "associes", "groupe", "compagnie", "freres", "developpement"}

ALL_ADDR = ("region", "ndeg", "suffix", "types", "elision")
ALL_NAME = ("et", "abbr", "stop", "trade", "bracket")


def address_v2(raw, fixes):
    """Return the fixed normalization of an address with the given fixes."""
    if raw is None:
        return ""
    text = str(raw)
    if "ndeg" in fixes:
        text = NUM_SIGN.sub(" ", text)
    if "region" in fixes:
        parts = [p for p in text.split(",")
                 if " ".join(N.NON_ALNUM.sub(" ", N.to_ascii(p)).split()) not in FR_REGION]
        text = ",".join(parts)
    text = N.to_ascii(text)
    if text.strip() in ("", "none", "null", "nan"):
        return ""
    if "elision" in fixes:
        text = ELISION.sub(r"\1", text)
    text = text.replace("&", " and ")
    text = N.NON_ALNUM.sub(" ", text)
    if "suffix" in fixes:
        text = NUM_SUFFIX.sub(r"\1 \2", text)
    tokens = []
    prev_num = False
    for token in text.split():
        if token in N.ADDRESS_STOP:
            continue
        is_num = token.isdigit()
        if is_num:
            token = token.lstrip("0") or "0"
        elif "suffix" in fixes and prev_num and token in AFTER_NUM:
            token = AFTER_NUM[token]
        if "types" in fixes and token in FR_ABBREV:
            token = FR_ABBREV[token]
        token = N.ADDRESS_ABBREV.get(token, token)
        if token:
            tokens.append(token)
        prev_num = is_num
    return " ".join(tokens)


def name_v2(raw, fixes):
    """Return the fixed normalization of a name with the given fixes."""
    transliterated = raw is not None and not str(raw).isascii()
    text = N.to_ascii(raw)
    if text in ("", "none", "null", "nan"):
        return "", ""
    if "bracket" in fixes:
        text = BRACKET_FRANCE.sub(" ", text)
    text = text.replace("&", " and ")
    text = N.DOMAIN.sub(r" \1 ", text)
    text = text.replace(".", "")
    parts = [p for p in N.DBA.split(text) if p.strip(" :-")]
    if len(parts) > 1:
        text = parts[-1]
    tokens = [N._deleet(t) for t in N.NON_ALNUM.sub(" ", text).split()]
    if transliterated and N.TRANSLIT:
        tokens = [N.TRANSLIT.get(t, t) for t in tokens]
    if "et" in fixes:
        tokens = ["and" if t == "et" else t for t in tokens]
    if "abbr" in fixes:
        tokens = [FR_NAME_MAP.get(t, t) if t != "et" else t for t in tokens]
    tokens = N._dedupe(tokens)
    stop = set(N.NAME_STOP)
    if "stop" in fixes:
        stop |= FR_NAME_STOP
    if "trade" in fixes:
        stop |= FR_NAME_STOP_TRADE
    core = [t for t in tokens if t not in stop and len(t) > 1]
    if not core:
        core = tokens
    return " ".join(tokens), " ".join(core)


def name_safe(raw):
    """Return the name normalization with only the safe fixes."""
    full, core = name_v2(raw, {"et", "abbr", "bracket"})
    toks = [t for t in core.split() if t not in ("ei", "eirl")]
    return full, (" ".join(toks) if toks else core)
