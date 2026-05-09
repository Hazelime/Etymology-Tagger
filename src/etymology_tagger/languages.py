"""
Language-code helpers for Wiktionary etymology templates.

Wiktionary uses internal language codes (e.g., 'la' for Latin, 'ang' for Old English).
This file provides a mapping from these codes to human-readable names to facilitate
data extraction and collapsing.

The mapping focuses on the most frequent source languages in the English corpus.
Unknown codes are returned as-is to ensure the pipeline is robust to new additions.
"""

LANGUAGE_CODE_TO_NAME = {
    "aa": "Afar",
    "ab": "Abkhaz",
    "ae": "Avestan",
    "af": "Afrikaans",
    "akk": "Akkadian",
    "ang": "Old English",
    "ar": "Arabic",
    "arc": "Aramaic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "ca": "Catalan",
    "cel-pro": "Proto-Celtic",
    "cop": "Coptic",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dum": "Middle Dutch",
    "el": "Greek",
    "enm": "Middle English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fro": "Old French",
    "fr": "French",
    "frk": "Frankish",
    "frm": "Middle French",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "gem-pro": "Proto-Germanic",
    "gmh": "Middle High German",
    "gmw-pro": "Proto-West Germanic",
    "got": "Gothic",
    "grc": "Ancient Greek",
    "grc-koi": "Koine Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hit": "Hittite",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ine-pro": "Proto-Indo-European",
    "ira-pro": "Proto-Iranian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "ko": "Korean",
    "la": "Latin",
    "la-cla": "Classical Latin",
    "la-lat": "Late Latin",
    "la-med": "Medieval Latin",
    "la-new": "New Latin",
    "la-vul": "Vulgar Latin",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mkh-pro": "Proto-Mon-Khmer",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "ms": "Malay",
    "mul": "Translingual",
    "nan": "Min Nan",
    "nl": "Dutch",
    "non": "Old Norse",
    "no": "Norwegian",
    "xno": "Anglo-Norman",
    "oc": "Occitan",
    "ota": "Ottoman Turkish",
    "peo": "Old Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sga": "Old Irish",
    "sla-pro": "Proto-Slavic",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "yi": "Yiddish",
    "zh": "Chinese",
}

def language_name(code: str | None) -> str | None:
    """
    Returns the human-readable name for a Wiktionary language code.
    
    If the code is unknown, returns the code itself (loss-tolerant behavior).
    """
    if not code:
        return None
    return LANGUAGE_CODE_TO_NAME.get(code, code)
