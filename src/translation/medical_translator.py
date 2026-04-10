Now the translation module:

Click src → translation folder
Click "Add file" → "Create new file"
Type in the filename box:

medical_translator.py

Paste this code:

python"""
translation/medical_translator.py
───────────────────────────────────
LLM-powered medical document translation pipeline.

Translates regulatory and clinical documents across 12 languages
while preserving:
    - Medical terminology accuracy (MedDRA, SNOMED CT, ICD-10)
    - Regulatory language conventions per target market
    - Document structure and formatting
    - Controlled terminology consistency

Supported languages:
    English, French, German, Spanish, Italian, Portuguese,
    Japanese, Chinese (Simplified), Korean, Dutch, Polish, Russian

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openai

logger = logging.getLogger(__name__)


# ── Language registry ─────────────────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    "en": ("English",              "FDA, EMA, PMDA"),
    "fr": ("French",               "EMA, ANSM (France)"),
    "de": ("German",               "EMA, BfArM (Germany)"),
    "es": ("Spanish",              "EMA, AEMPS (Spain), ANMAT (Argentina)"),
    "it": ("Italian",              "EMA, AIFA (Italy)"),
    "pt": ("Portuguese",           "ANVISA (Brazil), INFARMED (Portugal)"),
    "ja": ("Japanese",             "PMDA (Japan)"),
    "zh": ("Chinese (Simplified)", "NMPA (China)"),
    "ko": ("Korean",               "MFDS (Korea)"),
    "nl": ("Dutch",                "EMA, CBG-MEB (Netherlands)"),
    "pl": ("Polish",               "EMA, URPL (Poland)"),
    "ru": ("Russian",              "Roszdravnadzor (Russia)"),
}

# Document types requiring specialist translation approaches
DOCUMENT_TYPES = {
    "csr":          "Clinical Study Report",
    "protocol":     "Clinical Protocol",
    "ib":           "Investigator Brochure",
    "spc":          "Summary of Product Characteristics",
    "pil":          "Patient Information Leaflet",
    "ctr":          "Clinical Trial Report",
    "nonclinical":  "Nonclinical Study Report",
    "label":        "Drug Label / Package Insert",
    "consent":      "Informed Consent Form",
    "overview":     "Clinical/Nonclinical Overview",
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class TranslationRequest:
    """A single document translation request."""
    source_text: str
    source_language: str
    target_language: str
    document_type: str
    drug_name: str = ""
    indication: str = ""
    preserve_formatting: bool = True
    glossary: dict[str, str] = field(default_factory=dict)


@dataclass
class TranslationResult:
    """Result of a single translation."""
    source_language: str
    target_language: str
    document_type: str
    source_text: str
    translated_text: str
    back_translated_text: str = ""
    terminology_flags: list[str] = field(default_factory=list)
    word_count_source: int = 0
    word_count_target: int = 0
    model_used: str = ""

    def __post_init__(self):
        self.word_count_source = len(self.source_text.split())
        self.word_count_target = len(self.translated_text.split())

    def to_dict(self) -> dict:
        return {
            "source_language":    self.source_language,
            "target_language":    self.target_language,
            "document_type":      self.document_type,
            "word_count_source":  self.word_count_source,
            "word_count_target":  self.word_count_target,
            "terminology_flags":  self.terminology_flags,
            "back_translated":    bool(self.back_translated_text),
        }


@dataclass
class BatchTranslationResult:
    """Results from a multi-language batch translation."""
    drug_name: str
    document_type: str
    source_language: str
    results: dict[str, TranslationResult] = field(default_factory=dict)

    def add_result(self, lang_code: str, result: TranslationResult) -> None:
        self.results[lang_code] = result

    def summary(self) -> dict:
        return {
            "drug_name":       self.drug_name,
            "document_type":   self.document_type,
            "source_language": self.source_language,
            "languages_translated": list(self.results.keys()),
            "total_translations":   len(self.results),
        }


# ── Medical Translator ────────────────────────────────────────────────────────

class MedicalTranslator:
    """
    LLM-powered medical document translation with terminology validation.

    Translates regulatory and clinical documents while preserving
    medical terminology accuracy and regulatory language conventions.
    Supports optional back-translation for quality verification.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key.
    model : str
        LLM model. Default 'gpt-4o'.
    enable_back_translation : bool
        If True, back-translates output for QC. Default False.

    Examples
    --------
    >>> translator = MedicalTranslator()
    >>> result = translator.translate(TranslationRequest(
    ...     source_text="The primary endpoint was overall survival.",
    ...     source_language="en",
    ...     target_language="de",
    ...     document_type="csr",
    ...     drug_name="DrugX",
    ... ))
    >>> print(result.translated_text)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        enable_back_translation: bool = False,
    ) -> None:
        self.model                  = model
        self.enable_back_translation = enable_back_translation
        openai.api_key              = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not openai.api_key:
            raise EnvironmentError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key=."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def translate(self, request: TranslationRequest) -> TranslationResult:
        """
        Translate a single document or text segment.

        Parameters
        ----------
        request : TranslationRequest
            Translation request with source text and metadata.

        Returns
        -------
        TranslationResult
            Translation with quality metrics and terminology flags.
        """
        self._validate_languages(request.source_language, request.target_language)

        source_lang_name = SUPPORTED_LANGUAGES[request.source_language][0]
        target_lang_name = SUPPORTED_LANGUAGES[request.target_language][0]
        doc_type_label   = DOCUMENT_TYPES.get(request.document_type, request.document_type)

        logger.info(
            "Translating %s: %s → %s (%d words)",
            doc_type_label, source_lang_name, target_lang_name,
            len(request.source_text.split()),
        )

        translated = self._translate_text(request)
        terminology_flags = self._check_terminology(
            request.source_text, translated, request.target_language
        )

        back_translated = ""
        if self.enable_back_translation:
            back_translated = self._back_translate(
                translated, request.target_language, request.source_language,
                request.document_type,
            )

        result = TranslationResult(
            source_language=request.source_language,
            target_language=request.target_language,
            document_type=request.document_type,
            source_text=request.source_text,
            translated_text=translated,
            back_translated_text=back_translated,
            terminology_flags=terminology_flags,
            model_used=self.model,
        )

        logger.info(
            "Translation complete: %d → %d words, %d terminology flags",
            result.word_count_source, result.word_count_target,
            len(terminology_flags),
        )
        return result

    def translate_batch(
        self,
        source_text: str,
        target_languages: list[str],
        document_type: str,
        source_language: str = "en",
        drug_name: str = "",
        indication: str = "",
    ) -> BatchTranslationResult:
        """
        Translate a document into multiple target languages.

        Parameters
        ----------
        source_text : str
            Source document text.
        target_languages : list[str]
            List of ISO 639-1 language codes.
        document_type : str
            Document type key (e.g. 'csr', 'ib', 'spc').
        source_language : str
            Source language code. Default 'en'.
        drug_name : str
            Drug name for terminology context.
        indication : str
            Indication for terminology context.

        Returns
        -------
        BatchTranslationResult
        """
        batch = BatchTranslationResult(
            drug_name=drug_name,
            document_type=document_type,
            source_language=source_language,
        )

        for lang in target_languages:
            if lang not in SUPPORTED_LANGUAGES:
                logger.warning("Unsupported language code: %s — skipping", lang)
                continue
            if lang == source_language:
                continue

            request = TranslationRequest(
                source_text=source_text,
                source_language=source_language,
                target_language=lang,
                document_type=document_type,
                drug_name=drug_name,
                indication=indication,
            )
            result = self.translate(request)
            batch.add_result(lang, result)

        logger.info(
            "Batch translation complete: %d languages", len(batch.results)
        )
        return batch

    def save_translations(
        self,
        batch: BatchTranslationResult,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Save batch translation results to text files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = {}

        for lang_code, result in batch.results.items():
            lang_name = SUPPORTED_LANGUAGES[lang_code][0]
            fname = (
                f"{batch.drug_name}_{batch.document_type}_{lang_code}.txt"
                if batch.drug_name
                else f"{batch.document_type}_{lang_code}.txt"
            )
            path = output_dir / fname
            content = (
                f"TRANSLATION — {lang_name.upper()}\n"
                f"Document: {DOCUMENT_TYPES.get(batch.document_type, batch.document_type)}\n"
                f"Drug: {batch.drug_name}\n"
                f"Words: {result.word_count_target}\n"
                + (f"Terminology flags: {'; '.join(result.terminology_flags)}\n"
                   if result.terminology_flags else "")
                + f"{'='*60}\n\n"
                + result.translated_text
            )
            if result.back_translated_text:
                content += f"\n\n{'='*60}\nBACK TRANSLATION (QC):\n{result.back_translated_text}"

            path.write_text(content, encoding="utf-8")
            saved[lang_code] = path
            logger.info("Saved translation [%s]: %s", lang_code, path)

        return saved

    # ── Private helpers ───────────────────────────────────────────────────────

    def _translate_text(self, request: TranslationRequest) -> str:
        """Perform the actual translation via LLM."""
        source_lang = SUPPORTED_LANGUAGES[request.source_language][0]
        target_lang = SUPPORTED_LANGUAGES[request.target_language][0]
        target_agency = SUPPORTED_LANGUAGES[request.target_language][1]
        doc_type = DOCUMENT_TYPES.get(request.document_type, request.document_type)

        glossary_text = ""
        if request.glossary:
            glossary_text = "\nUse these specific term translations:\n" + "\n".join(
                f"  '{k}' → '{v}'" for k, v in request.glossary.items()
            )

        system_prompt = f"""
You are a specialist medical translator with expertise in pharmaceutical
regulatory documents. You translate between {source_lang} and {target_lang}
for regulatory submissions to {target_agency}.

Translation requirements:
- Preserve all medical terminology (MedDRA terms, INN names, SNOMED CT)
- Follow regulatory writing conventions for {target_lang}
- Maintain document structure, headings, and paragraph breaks
- Keep all numbers, statistics, and units exactly as written
- Drug names: keep INN names unchanged, translate brand names only if required
- Latin abbreviations (e.g. q.d., b.i.d.): use target language convention
- Do NOT add explanations or translator notes
- Return ONLY the translated text
{glossary_text}
""".strip()

        drug_context = f" for {request.drug_name}" if request.drug_name else ""
        indication_context = f" in {request.indication}" if request.indication else ""

        user_prompt = (
            f"Translate this {doc_type} text{drug_context}{indication_context} "
            f"from {source_lang} to {target_lang}:\n\n"
            f"{request.source_text}"
        )

        response = openai.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _back_translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        document_type: str,
    ) -> str:
        """Back-translate for quality verification."""
        source_name = SUPPORTED_LANGUAGES[source_lang][0]
        target_name = SUPPORTED_LANGUAGES[target_lang][0]
        doc_type    = DOCUMENT_TYPES.get(document_type, document_type)

        response = openai.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a medical translator. Translate the following "
                        f"{doc_type} text from {source_name} back to {target_name}. "
                        f"Return only the translated text."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content.strip()

    def _check_terminology(
        self,
        source: str,
        translated: str,
        target_lang: str,
    ) -> list[str]:
        """Flag potential terminology issues in the translation."""
        flags = []
        target_name = SUPPORTED_LANGUAGES[target_lang][0]

        system_prompt = """
You are a medical terminology validator. Review a translation and identify
any potential terminology issues. Return a JSON array of flag strings,
each describing a specific concern. Return an empty array [] if no issues.
Return ONLY the JSON array, nothing else.
""".strip()

        user_prompt = (
            f"Review this medical translation into {target_name} for terminology issues.\n\n"
            f"SOURCE:\n{source[:500]}\n\n"
            f"TRANSLATION:\n{translated[:500]}\n\n"
            f"Return JSON array of issues, max 5 items."
        )

        try:
            response = openai.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            import json
            content = response.choices[0].message.content.strip()
            flags = json.loads(content)
        except Exception:
            pass

        return flags

    @staticmethod
    def _validate_languages(source: str, target: str) -> None:
        """Validate language codes are supported."""
        if source not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported source language: '{source}'. "
                f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        if target not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported target language: '{target}'. "
                f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        if source == target:
            raise ValueError("Source and target languages must be different")
