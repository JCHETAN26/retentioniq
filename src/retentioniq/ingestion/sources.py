"""Registry of source datasets, their files, and the raw tables they load into.

Raw tables keep source column semantics (names are only snake_cased); renaming and
business rules happen in dbt staging. Every raw table also gets the metadata columns
in `METADATA_COLUMNS`.
"""

from dataclasses import dataclass, field

METADATA_COLUMNS: dict[str, str] = {
    "_run_id": "uuid NOT NULL",
    "_source_file": "text NOT NULL",
    "_source_row": "bigint NOT NULL",
}


@dataclass(frozen=True)
class SourceFile:
    filename: str
    table: str
    schema_version: str
    columns: dict[str, str]
    max_reject_rate: float = 0.01


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    archive: str | None  # "zip" if the download must be extracted, else None
    files: tuple[SourceFile, ...] = field(default_factory=tuple)
    requires_kaggle_auth: bool = False

    def file(self, filename: str) -> SourceFile:
        for f in self.files:
            if f.filename == filename:
                return f
        raise KeyError(f"{self.name} has no source file {filename!r}")


RETAILROCKET = Dataset(
    name="retailrocket",
    url="https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset",
    archive="zip",
    requires_kaggle_auth=True,
    files=(
        SourceFile(
            "events.csv",
            "retailrocket_events",
            "events.v1",
            {
                "timestamp": "bigint NOT NULL",
                "visitorid": "bigint NOT NULL",
                "event": "text NOT NULL",
                "itemid": "bigint NOT NULL",
                "transactionid": "bigint",
            },
        ),
        *(
            SourceFile(
                f"item_properties_part{i}.csv",
                "retailrocket_item_properties",
                "item_properties.v1",
                {
                    "timestamp": "bigint NOT NULL",
                    "itemid": "bigint NOT NULL",
                    "property": "text NOT NULL",
                    "value": "text NOT NULL",
                },
            )
            for i in (1, 2)
        ),
        SourceFile(
            "category_tree.csv",
            "retailrocket_category_tree",
            "category_tree.v1",
            {"categoryid": "bigint NOT NULL", "parentid": "bigint"},
        ),
    ),
)

ONLINE_RETAIL = Dataset(
    name="online-retail",
    url="https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
    archive="zip",
    files=(
        SourceFile(
            "online_retail_II.xlsx",
            "online_retail_transactions",
            "transactions.v1",
            {
                "invoice": "text NOT NULL",
                "stock_code": "text NOT NULL",
                "description": "text",
                "quantity": "bigint NOT NULL",
                "invoice_date": "timestamp NOT NULL",
                "price": "double precision NOT NULL",
                "customer_id": "bigint",
                "country": "text",
                "source_sheet": "text NOT NULL",
            },
        ),
    ),
)

CRITEO_UPLIFT = Dataset(
    name="criteo-uplift",
    # The original go.criteo.net link returns 404 (checked 2026-11-10); this is Criteo's own,
    # ungated Hugging Face copy of the same v2.1 file (311,422,618 bytes).
    url="https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
    archive=None,
    files=(
        SourceFile(
            "criteo-research-uplift-v2.1.csv.gz",
            "criteo_uplift",
            "criteo_uplift.v2.1",
            {
                **{f"f{i}": "double precision NOT NULL" for i in range(12)},
                "treatment": "smallint NOT NULL",
                "conversion": "smallint NOT NULL",
                "visit": "smallint NOT NULL",
                "exposure": "smallint NOT NULL",
            },
        ),
    ),
)

DATASETS: dict[str, Dataset] = {d.name: d for d in (RETAILROCKET, ONLINE_RETAIL, CRITEO_UPLIFT)}
