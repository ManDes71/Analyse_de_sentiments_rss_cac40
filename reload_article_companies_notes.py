#!/usr/bin/env python3
"""
Recharge les notes LLM dans public.article_companies a partir des CSV deja exportes.

Exemples:
  uv run reload_article_companies_notes.py
  uv run reload_article_companies_notes.py output/resultats_mistral_20260803_193015.csv
  uv run reload_article_companies_notes.py output/resultats_mistral_*.csv output/resultats_queen_*.csv
  uv run reload_article_companies_notes.py --dry-run
"""

import argparse
import csv
import glob
import os
import re
import sys
import unicodedata

import psycopg2
from dotenv import load_dotenv


MODEL_TO_COLUMNS = {
    "gemini": ("note_gemini", "justification_gemini"),
    "haiku": ("note_haiku", "justification_haiku"),
    "mistral": ("note_mistral", "justification_mistral"),
    "queen": ("note_queen", "justification_queen"),
    "lama": ("note_llama3", "justification_lama"),
}

MODEL_ALIASES = {
    "gemini": "gemini",
    "haiku": "haiku",
    "mistral": "mistral",
    "queen": "queen",
    "qwen": "queen",
    "lama": "lama",
    "llama": "lama",
    "llm": "lama",
}


def sanitize_csv_value(value: str | None) -> str:
    text = (value or "").replace("\x00", " ")
    return text.strip()


def normalize_name(value: str) -> str:
    text = sanitize_csv_value(value).upper()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


def infer_model_from_filename(path: str) -> str | None:
    basename = os.path.basename(path).lower()
    match = re.match(r"resultats?_([a-z0-9]+)_", basename)
    if not match:
        return None
    raw = match.group(1)
    return MODEL_ALIASES.get(raw)


def resolve_input_files(patterns: list[str]) -> list[str]:
    files = []
    for pattern in patterns:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            files.extend(expanded)
        elif os.path.isfile(pattern):
            files.append(pattern)
    seen = set()
    unique_files = []
    for file_path in files:
        abspath = os.path.abspath(file_path)
        if abspath not in seen:
            seen.add(abspath)
            unique_files.append(abspath)
    return unique_files


def build_company_index(cur) -> tuple[dict[str, int], set[str]]:
    cur.execute("SELECT id, name FROM public.companies")
    rows = cur.fetchall()

    index = {}
    duplicates = set()
    for company_id, name in rows:
        key = normalize_name(name)
        if key in index and index[key] != company_id:
            duplicates.add(key)
            continue
        index[key] = company_id
    return index, duplicates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recharge les notes article_companies depuis des CSV resultats_*."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Fichiers CSV ou patterns glob. Defaut: output/resultats_*.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simule les updates sans ecrire en base.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", 5432))
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "postgres")
    db_name = os.getenv("DB_NAME", "finance_db")

    input_patterns = args.files if args.files else ["output/resultats_*.csv"]
    csv_files = resolve_input_files(input_patterns)
    if not csv_files:
        print("Aucun fichier trouve. Exemple: output/resultats_mistral_*.csv")
        return 1

    print(f"{len(csv_files)} fichier(s) a traiter.")
    for path in csv_files:
        print(f"  - {path}")

    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        dbname=db_name,
    )

    total_rows = 0
    total_updated = 0
    total_skipped = 0

    try:
        with conn.cursor() as cur:
            company_index, duplicate_names = build_company_index(cur)
            if duplicate_names:
                print(
                    "Attention: des noms d'entreprise normalises sont dupliques dans companies."
                )
                print("Ces lignes peuvent etre ignorees si ambiguite.")

            for csv_path in csv_files:
                model = infer_model_from_filename(csv_path)
                if not model:
                    print(f"[SKIP] Modele introuvable depuis le nom: {csv_path}")
                    total_skipped += 1
                    continue

                note_col, justif_col = MODEL_TO_COLUMNS[model]
                print(f"\nTraitement {os.path.basename(csv_path)} -> colonnes {note_col}, {justif_col}")

                file_rows = 0
                file_updated = 0
                file_skipped = 0

                with open(csv_path, "r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    expected = {"article_id", "entreprise", "note_llm"}
                    if not reader.fieldnames or not expected.issubset(set(reader.fieldnames)):
                        print(
                            f"[SKIP] Colonnes manquantes dans {csv_path} (attendu: article_id, entreprise, note_llm)"
                        )
                        total_skipped += 1
                        continue

                    for row in reader:
                        file_rows += 1

                        article_raw = sanitize_csv_value(row.get("article_id"))
                        entreprise_raw = sanitize_csv_value(row.get("entreprise"))
                        note_raw = sanitize_csv_value(row.get("note_llm"))
                        justification = sanitize_csv_value(row.get("justification"))

                        if not article_raw or not entreprise_raw or note_raw == "":
                            file_skipped += 1
                            continue

                        try:
                            article_id = int(article_raw)
                            note = int(note_raw)
                        except ValueError:
                            file_skipped += 1
                            continue

                        if note not in (0, 1, 2):
                            file_skipped += 1
                            continue

                        entreprise_key = normalize_name(entreprise_raw)
                        company_id = company_index.get(entreprise_key)
                        if not company_id or entreprise_key in duplicate_names:
                            file_skipped += 1
                            continue

                        if not args.dry_run:
                            cur.execute(
                                f"""
                                UPDATE public.article_companies
                                SET {note_col} = %s, {justif_col} = %s
                                WHERE article_id = %s AND company_id = %s
                                """,
                                (note, justification, article_id, company_id),
                            )
                            if cur.rowcount > 0:
                                file_updated += cur.rowcount
                            else:
                                file_skipped += 1
                        else:
                            file_updated += 1

                if not args.dry_run:
                    conn.commit()

                total_rows += file_rows
                total_updated += file_updated
                total_skipped += file_skipped

                print(
                    f"  Lignes lues: {file_rows} | Maj effectuees: {file_updated} | Ignorees: {file_skipped}"
                )

        print("\nTermine.")
        if args.dry_run:
            print("Mode dry-run: aucune ecriture en base n'a ete faite.")
        print(
            f"Total lignes lues: {total_rows} | Total maj: {total_updated} | Total ignorees: {total_skipped}"
        )
        return 0

    except Exception as exc:
        conn.rollback()
        print(f"Erreur: {exc}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
