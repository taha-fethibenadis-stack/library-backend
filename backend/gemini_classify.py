"""
All Gemini (google-genai) interactions live here:
  1. classify_file()  — turns a freshly uploaded file into {subject, tag, summary}
  2. rank_search()    — turns a natural-language query + the library's records
                         into the 3 best-matching record ids

Model used: gemini-2.5-flash (free tier on Google AI Studio as of writing —
double check current limits at https://ai.google.dev/pricing since free-tier
terms change over time).
"""
import json
import logging

from google import genai
from google.genai import types

from config import GEMINI_API_KEY

logger = logging.getLogger("gemini")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-2.5-flash"

# The exact list of courses for this class/semester. Gemini is forced to pick
# one of these verbatim, so subjects can never drift into near-duplicate
# variants (e.g. "Math" vs "Mathematics" vs "Analyse"). Edit this list
# whenever the curriculum changes (new semester, different promo, etc.) —
# nothing else in the code needs to change.
SUBJECTS = [
    "Analyse 3",
    "Analyse numérique 1",
    "Physique 3",
    "Chimie 3",
    "Mécanique rationnelle",
    "Electricité générale",
    "Mécanique des fluides",
    "Informatique 3",
    "Ingénierie 1",
    "Techniques d'expression 1",
    "Anglais 3",
]
# Fallback used only when a file genuinely doesn't match any course above
# (e.g. someone shares a meme or an unrelated file in the group).
FALLBACK_SUBJECT = "Autre"

# File types Gemini can read directly (multimodal) for a much better summary.
# Anything else falls back to classifying from the filename/caption alone.
MULTIMODAL_MIME_PREFIXES = ("image/", "application/pdf")

_subjects_block = "\n".join(f"- {s}" for s in SUBJECTS)

CLASSIFY_SYSTEM_PROMPT = (
    """You are the librarian for a university class group chat called "Promo Library".
Students dump lecture slides, exercise sheets, and exam papers into the group with no
structure. Your job is to classify each file so it can be organized and searched later.

Always respond with ONLY a single valid JSON object, no markdown fences, no commentary,
in exactly this shape:
{
  "subject": "<one of the exact course names listed below>",
  "tag": "<one of exactly: Lecture, Exercise, Exam>",
  "summary": "<one keyword-dense sentence describing the content, written so a student
               searching with casual language like 'that physics sheet about vectors'
               would match it>"
}

The class follows EXACTLY these courses this semester. "subject" MUST be one of these
strings, copied verbatim (same spelling, accents, and numbering):
"""
    + _subjects_block
    + f"""

If a file genuinely does not belong to any of these courses (e.g. an off-topic file,
meme, or general announcement), use "{FALLBACK_SUBJECT}" instead of forcing it into one of them.

Rules:
- "tag" must be exactly one of: Lecture, Exercise, Exam. If genuinely unclear, infer the
  closest match from the filename/content (e.g. "TD", "worksheet", "homework", "TP" -> Exercise;
  "examen", "interrogation", "rattrapage", "midterm", "final", "EMD", "DS" -> Exam; default to Lecture
  otherwise).
- "summary" should pack in topics, chapter names, or keywords likely to appear in a
  student's search, not a generic description.
- Never invent a subject name that isn't in the list above (other than the fallback).
"""
)


def _safe_parse_json(text: str) -> dict | None:
    text = text.strip()
    # Defensive: strip accidental markdown fences even though we ask Gemini not to use them
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Gemini returned non-JSON output: %r", text[:300])
        return None


def classify_file(
    file_name: str,
    caption: str | None,
    mime_type: str | None,
    file_bytes: bytes | None,
) -> dict:
    """
    Returns {"subject": str, "tag": str, "summary": str}.
    Uses multimodal input (actual file content) when possible, otherwise
    falls back to classifying from the filename + caption alone.
    `subject` is guaranteed to be either one of SUBJECTS or FALLBACK_SUBJECT.
    """
    text_prompt = f"File name: {file_name}"
    if caption:
        text_prompt += f"\nCaption provided by the uploader: {caption}"

    contents: list = []
    can_read_content = (
        file_bytes is not None
        and mime_type is not None
        and mime_type.startswith(MULTIMODAL_MIME_PREFIXES)
        # Gemini free tier inline-data limit is ~20MB per request; stay well under it
        and len(file_bytes) <= 15 * 1024 * 1024
    )

    if can_read_content:
        contents.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
        text_prompt += "\nThe actual file content is attached above — read it to inform your classification."
    else:
        text_prompt += (
            "\n(The file content could not be read; classify based on the filename and "
            "caption only — make a reasonable best guess.)"
        )

    contents.append(text_prompt)

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=CLASSIFY_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        parsed = _safe_parse_json(response.text or "")
    except Exception:
        logger.exception("Gemini classify_file call failed")
        parsed = None

    if not parsed:
        return {"subject": FALLBACK_SUBJECT, "tag": "Lecture", "summary": file_name}

    subject = _snap_to_known_subject(str(parsed.get("subject") or ""))
    tag = str(parsed.get("tag") or "Lecture").strip()
    if tag not in ("Lecture", "Exercise", "Exam"):
        tag = "Lecture"
    summary = str(parsed.get("summary") or file_name).strip()

    return {"subject": subject, "tag": tag, "summary": summary}


_SUBJECTS_BY_LOWER = {s.lower(): s for s in SUBJECTS}


def _snap_to_known_subject(raw_subject: str) -> str:
    """
    Guarantees the returned subject is always exactly one of SUBJECTS or
    FALLBACK_SUBJECT — even if Gemini deviates slightly (extra whitespace,
    wrong casing, or ignores the instruction entirely).
    """
    cleaned = raw_subject.strip()
    if cleaned in SUBJECTS:
        return cleaned
    match = _SUBJECTS_BY_LOWER.get(cleaned.lower())
    return match if match else FALLBACK_SUBJECT


SEARCH_SYSTEM_PROMPT = """You are the search engine for a university class file library called
"Promo Library". You will be given a student's natural-language query and a JSON array of
available files (each with an "id", "file_name", "subject", "tag", and "summary").

Return ONLY a valid JSON object of this exact shape, no markdown fences, no commentary:
{"results": [<id>, <id>, <id>]}

Rules:
- Pick the up-to-3 ids that best match what the student is looking for, ordered best-first.
- Match on meaning, not just exact words — students search casually
  (e.g. "that physics sheet about vectors" should match a summary mentioning vector mechanics).
- If fewer than 3 files are genuinely relevant, return fewer ids. Never invent ids that
  weren't in the input list.
- If nothing is relevant at all, return {"results": []}.
"""


def rank_search(query: str, records: list[dict]) -> list[int]:
    """
    records: list of {"id": int, "file_name": str, "subject": str, "tag": str, "summary": str}
    Returns up to 3 record ids, best match first.
    """
    if not records:
        return []

    compact_records = [
        {
            "id": r["id"],
            "file_name": r.get("file_name", ""),
            "subject": r.get("subject", ""),
            "tag": r.get("tag", ""),
            "summary": r.get("summary", ""),
        }
        for r in records
    ]

    prompt = (
        f"Student query: {query}\n\n"
        f"Available files (JSON):\n{json.dumps(compact_records, ensure_ascii=False)}"
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SEARCH_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        parsed = _safe_parse_json(response.text or "")
    except Exception:
        logger.exception("Gemini rank_search call failed")
        parsed = None

    if not parsed or "results" not in parsed:
        return []

    valid_ids = {r["id"] for r in records}
    results = [rid for rid in parsed["results"] if rid in valid_ids]
    return results[:3]
