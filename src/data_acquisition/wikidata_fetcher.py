import argparse
import gzip
import os
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# --- Configuration for Wikidata Ranks ---
# Wikidata QIDs for the taxonomic ranks
WIKIDATA_RANK_URIS = [
    "http://www.wikidata.org/entity/Q36732",  # kingdom
    "http://www.wikidata.org/entity/Q38348",  # phylum
    "http://www.wikidata.org/entity/Q37517",  # class
    "http://www.wikidata.org/entity/Q36602",  # order
    "http://www.wikidata.org/entity/Q35409",  # family
    "http://www.wikidata.org/entity/Q34740",  # genus
    "http://www.wikidata.org/entity/Q7432",  # species
]

# Mapping from Wikidata URI for ranks to common names (for output columns)
RANK_URI_TO_NAME_MAP = {
    "http://www.wikidata.org/entity/Q36732": "kingdom",
    "http://www.wikidata.org/entity/Q38348": "phylum",
    "http://www.wikidata.org/entity/Q37517": "class",
    "http://www.wikidata.org/entity/Q36602": "order",
    "http://www.wikidata.org/entity/Q35409": "family",
    "http://www.wikidata.org/entity/Q34740": "genus",
    "http://www.wikidata.org/entity/Q7432": "species",
}


# A class to fetch and preprocess taxonomic data from Wikidata SPARQL endpoints.
# Generates files required by the TaxonomyMatcher.
class WikidataDataFetcher:
    QUERY_MAPPING = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT ?WdID ?eol ?gbif ?ncbi ?ott ?itis ?irmng ?col ?nbn ?worms ?bold ?plazi ?apni ?msw3 ?iNat ?eppo ?WdName WHERE {
      ?WdID wdt:P31 wd:Q16521;
            wdt:P225 ?WdName .
      OPTIONAL { ?WdID wdt:P9157 ?ott . }
      OPTIONAL { ?WdID wdt:P685 ?ncbi . }
      OPTIONAL { ?WdID wdt:P846 ?gbif . }
      OPTIONAL { ?WdID wdt:P830 ?eol . }
      OPTIONAL { ?WdID wdt:P815 ?itis . }
      OPTIONAL { ?WdID wdt:P5055 ?irmng . }
      OPTIONAL { ?WdID wdt:P10585 ?col . }
      OPTIONAL { ?WdID wdt:P3240 ?nbn . }
      OPTIONAL { ?WdID wdt:P850 ?worms . }
      OPTIONAL { ?WdID wdt:P3606 ?bold . }
      OPTIONAL { ?WdID wdt:P1992 ?plazi . }
      OPTIONAL { ?WdID wdt:P5984 ?apni . }
      OPTIONAL { ?WdID wdt:P959 ?msw3 . }
      OPTIONAL { ?WdID wdt:P3151 ?iNat . }
      OPTIONAL { ?WdID wdt:P3031 ?eppo . }
    }
    """

    QUERY_LINEAGE = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT ?WdID ?WdName ?hTax ?hTaxName ?hTaxRank WHERE {
      ?WdID wdt:P31 wd:Q16521;
            wdt:P225 ?WdName ;
            wdt:P171* ?hTax .
      ?hTax wdt:P225 ?hTaxName ;
            wdt:P105 ?hTaxRank .
    }
    """

    def __init__(self, output_dir=".", chunk_size=30000000):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.today_str = datetime.today().strftime("%Y%m%d")
        self.chunk_size = chunk_size  # For processing large CSVs in chunks

        self.wd_sparql_output_file = None
        self.wd_lineage_aligned_output_file = None
        self.wd_duplicates_output_file = None

    # Compresses a file to .gz and removes the original.
    @staticmethod
    def _compress_and_remove(file_path: str) -> str:
        gz_path = file_path + ".gz"
        print(f"Compressing {file_path} to {gz_path}...")
        with open(file_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)
        print(f"Original file {file_path} removed.")
        return gz_path

    # Streams SPARQL query results to a file in CSV format.
    @staticmethod
    def _querki_write_file(query: str, output_file: str):
        print(f"Executing SPARQL query and writing to {output_file}...")
        headers = {"Accept": "text/csv"}
        params = {"query": query}
        try:
            with requests.get(
                "https://qlever.dev/api/wikidata",
                headers=headers,
                params=params,
                stream=True,
                timeout=1000,
            ) as response:
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                with open(output_file, "w", encoding="utf-8") as f:
                    for chunk in response.iter_content(
                        chunk_size=8192, decode_unicode=True
                    ):
                        f.write(chunk)
            print(f"Query results written to {output_file}.")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from SPARQL endpoint: {e}")
            raise

    # Runs SPARQL query and returns results as a DataFrame (JSON format).
    @staticmethod
    def _querki_to_dataframe(query: str) -> pd.DataFrame:
        print("Executing SPARQL query and returning DataFrame...")
        endpoint_url = "https://qlever.dev/api/wikidata"
        try:
            response = requests.get(
                endpoint_url, params={"query": query, "format": "json"}, timeout=1000
            )
            response.raise_for_status()
            data = response.json()
            variables = data["head"]["vars"]
            rows = []
            for item in data["results"]["bindings"]:
                row = [item[var]["value"] if var in item else None for var in variables]
                rows.append(row)
            df = pd.DataFrame(rows, columns=variables)
            print("Query results loaded into DataFrame.")
            return df
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from SPARQL endpoint: {e}")
            raise
        except ValueError as e:
            print(f"Error parsing JSON response: {e}")
            raise

    # Fetches Wikidata mapping data and saves it to a gzipped file.
    def _fetch_mapping_data(self) -> str:
        file_mapping = os.path.join(self.output_dir, "wdTax_SPARQL.txt")
        df_mapping = self._querki_to_dataframe(self.QUERY_MAPPING)
        df_mapping.to_csv(file_mapping, index=False)
        return self._compress_and_remove(file_mapping)

    # Fetches Wikidata lineage data and saves it to a gzipped file.
    def _fetch_lineage_data(self) -> str:
        file_lineage = os.path.join(self.output_dir, "wdTax_SPARQL_lineage.txt")
        self._querki_write_file(self.QUERY_LINEAGE, file_lineage)
        return self._compress_and_remove(file_lineage)

    # Filters the lineage file to include only predefined ranks.
    def _filter_lineage_by_ranks(self, input_lineage_file: str) -> str:
        file_lineage_filtered = os.path.join(
            self.output_dir, "wdTax_SPARQL_lineage_filtered.txt.gz"
        )
        print(
            f"Filtering lineage data by predefined ranks to {file_lineage_filtered}..."
        )

        first_chunk = True
        try:
            for chunk in pd.read_csv(
                input_lineage_file,
                compression="gzip",
                chunksize=self.chunk_size,
                dtype=str,
            ):
                required_columns = ["WdID", "WdName", "hTaxRank", "hTaxName"]
                if not all(col in chunk.columns for col in required_columns):
                    print(
                        f"Warning: Missing required columns in chunk. Expected: {required_columns}, Found: {list(chunk.columns)}. Skipping chunk."
                    )
                    continue

                filtered_chunk = chunk[chunk["hTaxRank"].isin(WIKIDATA_RANK_URIS)]
                mode = "w" if first_chunk else "a"
                header = first_chunk
                filtered_chunk.to_csv(
                    file_lineage_filtered,
                    compression="gzip",
                    mode=mode,
                    header=header,
                    index=False,
                )
                first_chunk = False
            print("Lineage data filtered.")
            return file_lineage_filtered
        except Exception as e:
            print(f"Error during lineage filtering: {e}")
            raise

    # Aligns filtered lineage data into columns with ranks as headers.
    def _align_lineage_data(self, filtered_lineage_file: str) -> str:
        file_lineage_filtered_aligned = os.path.join(
            self.output_dir, "wdTax_SPARQL_lineage_filtered_aligned.txt.gz"
        )
        print(f"Aligning filtered lineage data to {file_lineage_filtered_aligned}...")

        first_chunk = True
        # Define the header for the output file using common rank names
        output_header = ["WdID", "WdName"] + [
            RANK_URI_TO_NAME_MAP[uri] for uri in WIKIDATA_RANK_URIS
        ]

        try:
            chunk_iter = pd.read_csv(
                filtered_lineage_file,
                compression="gzip",
                chunksize=self.chunk_size,
                dtype=str,
            )
            for chunk in chunk_iter:
                chunk.columns = chunk.columns.str.strip()
                required_columns = ["WdID", "WdName", "hTaxRank", "hTaxName"]
                if not all(col in chunk.columns for col in required_columns):
                    print(
                        f"Warning: Missing required columns in chunk for alignment. Expected: {required_columns}, Found: {list(chunk.columns)}. Skipping chunk."
                    )
                    continue
                # Create a temporary DataFrame to build the aligned structure for this chunk
                # Using pivot_table can be more efficient than iterrows for large chunks, but the below might malfunction if the chunk-size is increased.
                pivot_df = chunk.pivot_table(
                    index=["WdID", "WdName"],
                    columns="hTaxRank",
                    values="hTaxName",
                    aggfunc="first",  # Use 'first' in case of multiple entries for same rank (shouldn't happen for clean data)
                )
                pivot_df.reset_index(inplace=True)
                # Rename columns from URI to common name
                pivot_df.rename(columns=RANK_URI_TO_NAME_MAP, inplace=True)
                # Ensure all predefined rank columns exist, filling missing with empty string
                for rank_name in [
                    RANK_URI_TO_NAME_MAP[uri] for uri in WIKIDATA_RANK_URIS
                ]:
                    if rank_name not in pivot_df.columns:
                        pivot_df[rank_name] = (
                            np.nan
                        )  # Use NaN first, then fillna later if needed

                # Select and reorder columns according to output_header
                transformed_chunk = pivot_df[output_header]
                mode = "w" if first_chunk else "a"
                header_write = first_chunk  # Write header only for the first chunk
                transformed_chunk.to_csv(
                    file_lineage_filtered_aligned,
                    compression="gzip",
                    mode=mode,
                    header=header_write,
                    index=False,
                )
                first_chunk = False
            print("Lineage data aligned.")
            return file_lineage_filtered_aligned
        except Exception as e:
            print(f"Error during lineage alignment: {e}")
            raise

    # Detects duplicate taxonomic names in the aligned lineage file.
    def _find_lineage_duplicates(self, aligned_lineage_file: str) -> str:
        file_duplicates = os.path.join(self.output_dir, "wdTax_duplicates.txt")
        print(f"Detecting duplicates in aligned lineage to {file_duplicates}...")

        # Read the aligned file (can be large, so process in chunks if necessary,
        # but for duplicates, often need to load all WdName to find true duplicates)
        # For simplicity, loading fully for duplicate detection as it's often a smaller file
        # or pandas handles it well. If memory is an issue, this part needs chunking too. TBD
        try:
            aligned_df = pd.read_csv(
                aligned_lineage_file, compression="gzip", dtype=str
            )
            # Ensure WdName column exists
            if "WdName" not in aligned_df.columns:
                raise ValueError(f"Missing 'WdName' column in {aligned_lineage_file}")

            dup_df = aligned_df[aligned_df["WdName"].duplicated(keep=False)]
            if not dup_df.empty:
                dup_df.sort_values(by="WdName", inplace=True)
                # The header for the duplicates file should match the aligned file's header
                dup_df.to_csv(file_duplicates, header=True, index=False)
                print(f"Duplicates found and saved to {file_duplicates}.")
            else:
                print("No duplicates found.")
                # Create an empty file or a file with just headers if no duplicates
                pd.DataFrame(columns=aligned_df.columns).to_csv(
                    file_duplicates, header=True, index=False
                )

            return file_duplicates
        except Exception as e:
            print(f"Error during duplicate detection: {e}")
            raise

    # Runs the full Wikidata data fetching and initial processing pipeline.
    # Returns a dictionary of paths to the generated files.
    def run_fetch_pipeline(self) -> dict:
        print("Starting Wikidata data fetching pipeline...")
        pipeline_start_time = time.time()

        # Fetch mapping data
        mapping_file_gz = self._fetch_mapping_data()
        print(f"Mapping data saved: {mapping_file_gz}")

        # Fetch lineage data
        lineage_file_gz = self._fetch_lineage_data()
        print(f"Lineage data saved: {lineage_file_gz}")

        # Filter lineage by ranks
        filtered_lineage_file_gz = self._filter_lineage_by_ranks(lineage_file_gz)
        print(f"Filtered lineage data saved: {filtered_lineage_file_gz}")

        # Align lineage data
        filtered_lineage_file_gz = "data/processed/wdTax_SPARQL_lineage_filtered.txt.gz"
        aligned_lineage_file_gz = self._align_lineage_data(filtered_lineage_file_gz)
        print(f"Aligned lineage data saved: {aligned_lineage_file_gz}")

        # Find duplicates in aligned lineage
        duplicates_file = self._find_lineage_duplicates(aligned_lineage_file_gz)
        print(f"Duplicates file saved: {duplicates_file}")

        pipeline_end_time = time.time()
        print(
            f"Wikidata data fetching pipeline finished in {pipeline_end_time - pipeline_start_time:.2f} seconds."
        )

        return {
            "wd_sparql_file": mapping_file_gz,
            "wd_lineage_aligned_file": aligned_lineage_file_gz,
            "wd_repeats_file": duplicates_file,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch and preprocess Wikidata taxonomic data."
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory to save output files."
    )
    # Chunk size has to be atleast the number of lines of the linege_filter_aligned file
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=30000000,
        help="Chunk size for reading large CSV files.",
    )
    args = parser.parse_args()

    fetcher = WikidataDataFetcher(
        output_dir=args.output_dir, chunk_size=args.chunk_size
    )
    generated_files = fetcher.run_fetch_pipeline()
    print("\nGenerated files for TaxonomyMatcher:")
    for key, value in generated_files.items():
        print(f"- {key}: {value}")
