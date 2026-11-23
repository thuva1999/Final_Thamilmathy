"""Step 6b: the advisor - retrieval + LLM (Google Gemini) + no-LLM fallback, 3 languages.

get_advice(top, band, language) is the only function the app needs:
  top      : list of (class name, probability), best first (from the classifier)
  band     : "high", "medium" or "low" (confidence rule from Step 5)
  language : "en", "uz" (Uzbek, Latin script) or "ru"

  high   -> four-part advice for the predicted class (what it is, what to do now,
            how to prevent it, when to call an agronomist), written by the LLM
            ONLY from the retrieved knowledge-base rows
  medium -> "Possibly A or B" + how to tell them apart (symptoms rows only)
  low    -> no diagnosis; ask for a clearer photo
  Healthy Leaf -> fixed message (the table has no rows for healthy leaves)

Safety: the LLM answer is rejected (and the table rows are shown instead) if it
contains any number that is not in the retrieved rows - so it cannot add doses.
Without an API key (LLM_API_KEY in .env) the table rows are shown directly.

Run from the project folder:  python src/advisor.py   (prints example answers)
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from knowledge_base import SOURCE, retrieve

ROOT = Path(__file__).resolve().parent.parent

# Tried in this order unless LLM_MODEL is set in .env (free-tier models change often)
DEFAULT_MODELS = ["gemini-3.6-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]

LANGUAGE_NAMES = {"en": "English", "uz": "Uzbek (Latin script)", "ru": "Russian"}

CLASS_NAMES = {
    "Bacterial Blight": {"en": "Bacterial blight", "uz": "Bakterial kuyish (gommoz)",
                         "ru": "Бактериальный ожог (гоммоз)"},
    "Curl Virus": {"en": "Leaf curl virus", "uz": "Barg buralish virusi",
                   "ru": "Вирус скручивания листьев"},
    "Healthy Leaf": {"en": "Healthy leaf", "uz": "Sog'lom barg", "ru": "Здоровый лист"},
    "Herbicide Growth Damage": {"en": "Herbicide damage", "uz": "Gerbitsid shikasti",
                                "ru": "Повреждение гербицидом"},
    "Leaf Hopper Jassids": {"en": "Leafhoppers (jassids)", "uz": "Sikadalar (jassidlar)",
                            "ru": "Цикадки (джассиды)"},
    "Leaf Redding": {"en": "Leaf reddening", "uz": "Barg qizarishi",
                     "ru": "Покраснение листьев"},
    "Leaf Variegation": {"en": "Leaf variegation", "uz": "Barg ola-bulaligi",
                         "ru": "Пестролистность"},
}

TEXT = {
    "en": {
        "what_it_is": "What it is", "do_now": "What to do now",
        "prevent": "How to prevent it", "agronomist": "When to call an agronomist",
        "agronomist_text": "Call an agronomist if the problem spreads to many plants, if it "
                           "does not improve within 1–2 weeks, if you are not sure about the "
                           "diagnosis, and always before using a chemical product you have "
                           "not used before.",
        "healthy": "The leaf looks healthy. No treatment is needed. Keep checking your plants "
                   "every week, especially the underside of the leaves.",
        "retake": "The photo is not clear enough for a diagnosis. Please take a new close-up "
                  "photo of one leaf on the plant, in daylight, with the leaf filling most of "
                  "the picture.",
        "possibly": "Possibly **{a}** or **{b}**. Compare the leaf with these signs:",
        "no_rows": "The knowledge base has no information about this problem.",
        "fallback": "Advice shown directly from the knowledge base (no AI text).",
        "english_source": "",
        "rejected": "The AI answer did not pass the safety check, so the knowledge-base "
                    "text is shown instead.",
    },
    "uz": {
        "what_it_is": "Bu nima", "do_now": "Hozir nima qilish kerak",
        "prevent": "Qanday oldini olish mumkin",
        "agronomist": "Qachon agronomga murojaat qilish kerak",
        "agronomist_text": "Agar muammo ko'plab o'simliklarga tarqalsa, 1–2 hafta ichida "
                           "yaxshilanmasa, tashxisga ishonchingiz komil bo'lmasa, va har doim "
                           "yangi kimyoviy vositani ishlatishdan oldin agronomga murojaat qiling.",
        "healthy": "Barg sog'lom ko'rinadi. Davolash shart emas. O'simliklaringizni har hafta, "
                   "ayniqsa barglarning orqa tomonini tekshirib turing.",
        "retake": "Rasm tashxis qo'yish uchun yetarlicha aniq emas. Iltimos, o'simlikdagi "
                  "bitta bargni kunduzgi yorug'likda yaqindan, barg rasmning katta qismini "
                  "egallaydigan qilib qayta suratga oling.",
        "possibly": "Ehtimol, **{a}** yoki **{b}**. Bargni quyidagi belgilar bilan solishtiring:",
        "no_rows": "Bilimlar bazasida bu muammo haqida ma'lumot yo'q.",
        "fallback": "Maslahat to'g'ridan-to'g'ri bilimlar bazasidan ko'rsatilmoqda "
                    "(sun'iy intellekt matnisiz).",
        "english_source": "Manba matni ingliz tilida.",
        "rejected": "Sun'iy intellekt javobi xavfsizlik tekshiruvidan o'tmadi, shuning uchun "
                    "bilimlar bazasi matni ko'rsatilmoqda.",
    },
    "ru": {
        "what_it_is": "Что это", "do_now": "Что делать сейчас",
        "prevent": "Как предотвратить", "agronomist": "Когда обратиться к агроному",
        "agronomist_text": "Обратитесь к агроному, если проблема распространяется на многие "
                           "растения, если улучшения нет в течение 1–2 недель, если вы не "
                           "уверены в диагнозе, и всегда перед первым применением "
                           "химического препарата.",
        "healthy": "Лист выглядит здоровым. Лечение не требуется. Продолжайте осматривать "
                   "растения каждую неделю, особенно нижнюю сторону листьев.",
        "retake": "Фото недостаточно чёткое для диагноза. Пожалуйста, сделайте новое фото "
                  "одного листа на растении крупным планом, при дневном свете, чтобы лист "
                  "занимал большую часть кадра.",
        "possibly": "Возможно, **{a}** или **{b}**. Сравните лист с этими признаками:",
        "no_rows": "В базе знаний нет информации об этой проблеме.",
        "fallback": "Совет показан напрямую из базы знаний (без текста ИИ).",
        "english_source": "Исходный текст на английском языке.",
        "rejected": "Ответ ИИ не прошёл проверку безопасности, поэтому показан текст из "
                    "базы знаний.",
    },
}


@dataclass
class Advice:
    """What the app shows: the advice text, the source rows, and how it was made."""
    markdown: str
    sources: pd.DataFrame = field(default_factory=pd.DataFrame)
    method: str = "fallback"      # llm / fallback / healthy / retake
    model: str | None = None      # LLM model name, if one was used
    note: str = ""                # e.g. why the fallback was used


# ---------- Small helpers ----------

def class_label(class_name, language):
    """Class name in the chosen language (falls back to the original name)."""
    return CLASS_NAMES.get(class_name, {}).get(language, class_name)


def context_text(rows):
    """Retrieved rows as numbered context for the LLM."""
    return "\n\n".join(f"[{r.row_id}] {r.disease} - {r.feature}:\n{r.text}"
                       for r in rows.itertuples())


def numbers_in(text):
    """All numbers in a text, e.g. '1.25', '20', '300' (row ids like KB012 ignored)."""
    text = re.sub(r"\[?KB\d+\]?", " ", text)
    return {n.replace(",", ".").lstrip("0") or "0" for n in re.findall(r"\d+(?:[.,]\d+)?", text)}


def unsupported_numbers(answer, context):
    """Numbers in the LLM answer that are not in the retrieved text (1-4 allowed: headings,
    and '1-2 weeks' in the agronomist part)."""
    allowed = numbers_in(context) | {"1", "2", "3", "4"}
    return sorted(numbers_in(answer) - allowed)


# ---------- No-LLM fallback ----------

def fallback_advice(class_name, rows, language, note=""):
    """Show the table rows directly, sorted into the four parts."""
    t = TEXT[language]
    lines = [f"### {class_label(class_name, language)}"]
    for part in ["what_it_is", "do_now", "prevent"]:
        part_rows = rows[rows["part"] == part]
        if part_rows.empty:
            continue
        lines.append(f"**{t[part]}**")
        for r in part_rows.itertuples():
            items = "\n".join(f"- {line}" for line in r.text.splitlines())
            lines.append(f"*{r.feature.replace('_', ' ')}* [{r.row_id}]\n{items}")
    lines += [f"**{t['agronomist']}**", t["agronomist_text"]]
    footer = " ".join(x for x in [note, t["fallback"], t["english_source"]] if x)
    lines.append(f"_{footer}_")
    return Advice("\n\n".join(lines), rows, "fallback", None, note)


def fallback_compare(a, b, rows_a, rows_b, language):
    """Medium confidence without LLM: symptoms of both candidates."""
    t = TEXT[language]
    lines = [t["possibly"].format(a=class_label(a, language), b=class_label(b, language))]
    for name, rows in [(a, rows_a), (b, rows_b)]:
        symptoms = rows[rows["feature"] == "Symptoms"]
        if name == "Healthy Leaf":
            text = t["healthy"].split(".")[0] + "."
        elif symptoms.empty:
            text = t["no_rows"]
        else:
            text = "\n".join(f"- {line}" for line in symptoms.iloc[0]["text"].splitlines())
        lines.append(f"**{class_label(name, language)}**\n{text}")
    footer = " ".join(x for x in [t["fallback"], t["english_source"]] if x)
    lines.append(f"_{footer}_")
    sources = pd.concat([rows_a[rows_a["feature"] == "Symptoms"],
                         rows_b[rows_b["feature"] == "Symptoms"]])
    return Advice("\n\n".join(lines), sources, "fallback")


# ---------- LLM (Gemini) ----------

def call_llm(system, prompt):
    """Send one request to Gemini. Returns (text, model). Raises if no key / all models fail."""
    load_dotenv(ROOT / ".env")
    key = os.getenv("LLM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("no LLM_API_KEY in .env")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    models = [os.getenv("LLM_MODEL").strip()] if os.getenv("LLM_MODEL") else DEFAULT_MODELS
    errors = []
    for model in models:
        try:
            response = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.2))
            if response.text and response.text.strip():
                return response.text.strip(), model
            errors.append(f"{model}: empty answer")
        except Exception as e:  # wrong model name, quota, network ...
            errors.append(f"{model}: {str(e)[:120]}")
    raise RuntimeError("; ".join(errors))


def system_prompt(language):
    """Rules for the LLM: only use the context, never add products or doses."""
    return (
        "You are CottonCare, a helper for cotton farmers in Uzbekistan. "
        "Use ONLY the facts in the CONTEXT given by the user. "
        "Never mention any pesticide, product, chemical, active ingredient, dose, rate or "
        "number that is not written in the CONTEXT. If the CONTEXT does not cover something, "
        "say so briefly instead of guessing. "
        f"Write in {LANGUAGE_NAMES[language]}"
        + (" using the Latin alphabet" if language == "uz" else "") + ". "
        "Keep product names exactly as written in the CONTEXT. "
        "Use simple words a farmer understands, short bullet points, at most about 200 words. "
        "After each fact, add the id of the context row it comes from, like [KB012]."
    )


def llm_advice(class_name, rows, language):
    """Four-part advice for one class from its rows."""
    t = TEXT[language]
    headings = [t["what_it_is"], t["do_now"], t["prevent"], t["agronomist"]]
    prompt = (
        f"The photo shows a cotton leaf with: {class_name} "
        f"({class_label(class_name, language)}).\n\n"
        "Write advice with exactly these four Markdown headings (### ...), in this order:\n"
        + "\n".join(f"### {h}" for h in headings) + "\n\n"
        f"Under '{headings[1]}', give the most practical steps first (non-chemical before "
        "chemical) and at most 3 products with their doses copied exactly from the CONTEXT. "
        f"Under '{headings[3]}', give general advice only (spreading, no improvement, "
        "unsure, before using chemicals) - no products.\n\n"
        f"CONTEXT:\n{context_text(rows)}"
    )
    answer, model = call_llm(system_prompt(language), prompt)
    return answer, model


def llm_compare(a, b, rows, language):
    """Medium confidence: how to tell A and B apart, from their symptoms only."""
    t = TEXT[language]
    intro = t["possibly"].format(a=class_label(a, language), b=class_label(b, language))
    healthy_note = (" One of the two options is a healthy leaf: healthy leaves show none of "
                    "these signs." if "Healthy Leaf" in (a, b) else "")
    prompt = (
        f"The classifier is unsure between {a} and {b}.{healthy_note}\n"
        "Write 3-5 short bullet points that help a farmer tell them apart by looking at the "
        "leaf, using only the symptoms in the CONTEXT. Then one sentence: take another photo "
        "or ask an agronomist. No treatments, no products. Do not repeat the question.\n\n"
        f"CONTEXT:\n{context_text(rows)}"
    )
    answer, model = call_llm(system_prompt(language), prompt)
    return f"{intro}\n\n{answer}", model


# ---------- Main entry point ----------

def get_advice(top, band, language="en"):
    """Advice for a classifier result. See the module docstring."""
    t = TEXT[language]
    best = top[0][0]

    if band == "low":
        return Advice(t["retake"], method="retake")

    if band == "medium" and len(top) > 1:
        a, b = top[0][0], top[1][0]
        rows_a, _ = retrieve(a)
        rows_b, _ = retrieve(b)
        symptoms = pd.concat([rows_a[rows_a["feature"] == "Symptoms"],
                              rows_b[rows_b["feature"] == "Symptoms"]])
        if symptoms.empty:
            return fallback_compare(a, b, rows_a, rows_b, language)
        try:
            text, model = llm_compare(a, b, symptoms, language)
        except RuntimeError as e:
            advice = fallback_compare(a, b, rows_a, rows_b, language)
            advice.note = str(e)
            return advice
        if unsupported_numbers(text, context_text(symptoms)):
            advice = fallback_compare(a, b, rows_a, rows_b, language)
            advice.note = t["rejected"]
            return advice
        return Advice(text, symptoms, "llm", model)

    rows, method = retrieve(best)
    if method == "healthy":
        return Advice(t["healthy"], method="healthy")
    if rows.empty:
        return Advice(t["no_rows"] + "\n\n" + t["agronomist_text"], method="fallback")
    try:
        text, model = llm_advice(best, rows, language)
    except RuntimeError as e:  # no key, quota, network ... -> show the table rows
        return _with_note(fallback_advice(best, rows, language), str(e))
    bad = unsupported_numbers(text, context_text(rows))
    if bad:
        return _with_note(fallback_advice(best, rows, language, note=t["rejected"]),
                          f"{t['rejected']} (numbers not in source: {', '.join(bad)})")
    return Advice(text, rows, "llm", model)


def _with_note(advice, note):
    """Attach a technical note (why the fallback was used) without changing the text."""
    advice.note = note
    return advice


def source_citation():
    """One line for the app's source box and About page."""
    return SOURCE


def main():
    """Print example answers for the three confidence bands."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    examples = [
        ([("Leaf Redding", 0.93), ("Leaf Hopper Jassids", 0.04)], "high", "en"),
        ([("Bacterial Blight", 0.62), ("Leaf Hopper Jassids", 0.30)], "medium", "ru"),
        ([("Healthy Leaf", 0.97), ("Curl Virus", 0.02)], "high", "uz"),
        ([("Curl Virus", 0.41), ("Leaf Redding", 0.33)], "low", "uz"),
    ]
    for top, band, lang in examples:
        advice = get_advice(top, band, lang)
        print("=" * 70)
        print(f"{top[0][0]} | band={band} | language={lang} | method={advice.method} "
              f"| model={advice.model}")
        if advice.note:
            print(f"note: {advice.note}")
        print("-" * 70)
        print(advice.markdown)
        if not advice.sources.empty:
            print(f"\nsources: {', '.join(advice.sources['row_id'])} - {SOURCE}")


if __name__ == "__main__":
    main()
