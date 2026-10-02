"""Chunked readers that turn each source file into DataFrames with raw-table column names.

Values are read as strings where the source may contain malformed data, so that type
coercion (and rejection) happens in one place: the quality contracts.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd

CHUNK_ROWS = 500_000

Reader = Callable[[Path], Iterator[pd.DataFrame]]

ONLINE_RETAIL_COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def read_csv(path: Path) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=CHUNK_ROWS)


def read_online_retail(path: Path) -> Iterator[pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, engine="openpyxl")
    for sheet, df in sheets.items():
        df = df.rename(columns=ONLINE_RETAIL_COLUMNS)
        df["source_sheet"] = sheet
        for start in range(0, len(df), CHUNK_ROWS):
            yield df.iloc[start : start + CHUNK_ROWS]


READERS: dict[str, Reader] = {
    "online_retail_transactions": read_online_retail,
}


def reader_for(table: str) -> Reader:
    return READERS.get(table, read_csv)
