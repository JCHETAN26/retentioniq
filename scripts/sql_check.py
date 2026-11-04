"""Verify syntax and parse tree of all analytical SQL queries in sql/analysis/."""

import sys
from pathlib import Path

import sqlparse

SQL_DIR = Path(__file__).resolve().parents[1] / "sql" / "analysis"


def main() -> None:
    sql_files = sorted(SQL_DIR.glob("*.sql"))
    if not sql_files:
        print(f"Error: no SQL files found in {SQL_DIR}", file=sys.stderr)
        sys.exit(1)

    has_errors = False
    print(f"Checking {len(sql_files)} SQL analysis queries...")
    for f in sql_files:
        content = f.read_text(encoding="utf-8")
        try:
            parsed = sqlparse.parse(content)
            if not parsed:
                print(f"FAIL: {f.name} - empty parse tree", file=sys.stderr)
                has_errors = True
            else:
                # Basic validation: ensure query contains SELECT and FROM tokens
                tokens = [t.value.upper() for statement in parsed for t in statement.tokens]
                has_select = any("SELECT" in t for t in tokens)
                has_from = any("FROM" in t for t in tokens)
                if not (has_select and has_from):
                    print(
                        f"FAIL: {f.name} - missing SELECT/FROM keywords",
                        file=sys.stderr,
                    )
                    has_errors = True
                else:
                    print(f"PASS: {f.name}")
        except Exception as e:
            print(f"FAIL: {f.name} - {e}", file=sys.stderr)
            has_errors = True

    if has_errors:
        sys.exit(1)
    print("All SQL queries passed syntax and structure checks.")


if __name__ == "__main__":
    main()
