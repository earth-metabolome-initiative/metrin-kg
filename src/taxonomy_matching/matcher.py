import pandas as pd
import numpy as np
import gzip
import os
import argparse
import configparser
from datetime import datetime
from itertools import zip_longest
import time
import csv
import re
from pathlib import Path

# from config import predefined_ranks, prefixes 

predefined_ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
prefixes = {
    1: "EOL:", 2: "GBIF:", 3: "NCBI:", 4: "OTT:", 5: "ITIS:",
    6: "IRMNG:", 7: "COL:", 8: "NBN:", 9: "WORMS:", 10: "BOLD:",
    11: "PLAZI:", 12: "APNI:", 13: "msw3:", 14: "INAT_TAXON:", 15: "EPPO:"
}

# A class to encapsulate the entire taxonomy matching pipeline,
# including both GloBI and TRY-db specific matching logic.
# Manages loading data, preprocessing, and applying matching logic.
class TaxonomyMatcher:
    def __init__(self, config_file="config.txt", args=None):
#                 wd_sparql_file=None, wd_lineage_aligned_file=None, wd_repeats_file=None):
        self.config_file = config_file
        self.args = args

        # File paths - can be passed directly or loaded from config/args
        self.wd_sparql_file = None
        self.verbatim_file = None # Only for GloBI
        self.output_file_globi = None # Only for GloBI
        self.output_file_trydb = None # Only for TRY-db
        self.wd_lineage_file = None # This is the aligned lineage file
        self.wd_repeats_file = None # This is the duplicates file
        self.trydb_species_file = None # Specific to TRY-db

        # Lookup dictionaries and sets for GloBI matching
        self.globi_id_map = {}
        self.globi_id_map_wd = {}
        self.globi_wd_name_to_id_set = set()
        self.globi_wd_name_to_id = {}
        self.globi_wd_lineage_dict = {} # For repeated names in GloBI context
        self.globi_wd_lineage_set = set() # For GloBI's internal duplicate tracking

        # Lookup dictionaries and sets for TRY-db matching
        self.trydb_wd_name_to_id_set = set()
        self.trydb_wd_name_to_id = {}

        # Common configurations
        self.predefined_ranks = predefined_ranks
        self.prefixes = prefixes

        self._load_configuration()

    # Loads file paths from config file or command-line arguments.
    # Prioritize directly passed file paths (from WikidataDataFetcher)
    # Then check config file, then command-line args.
    def _load_configuration(self):
        config_loaded = False
        if os.path.exists(self.config_file):
            config = configparser.ConfigParser()
            config.read(self.config_file)
            self.verbatim_file = config.get('input tsv files', 'globi_verbatim_file', fallback=None)
            self.output_file_globi = config.get('accessory files', 'globi_wd', fallback=None)
            self.output_file_trydb = config.get('accessory files', 'trydb_wd', fallback=None)
            self.trydb_species_file = config.get('input tsv files', 'tryDb_species_file', fallback=None)
            self.wd_sparql_file = config.get('wd files', 'wd_sparql_file', fallback=None)
            self.wd_lineage_file = config.get('wd files', 'wd_lineage_file', fallback=None)
            self.wd_repeats_file = config.get('wd files', 'wd_repeats_file', fallback=None)
            """# If not passed directly, try to get from config. Removed now for simplicity. To Delete later.
            if not self.wd_sparql_file: self.wd_sparql_file = config.get('input tsv files', 'wd_sparql_file', fallback=None)
            if not self.wd_lineage_file: self.wd_lineage_file = config.get('input tsv files', 'wd_lineage_file', fallback=None)
            if not self.wd_repeats_file: self.wd_repeats_file = config.get('input tsv files', 'wd_repeats_file', fallback=None)"""
            config_loaded = True
        
        if self.args and not config_loaded: # Only load from args if config file wasn't found/used
            self.verbatim_file = getattr(self.args, 'verbatim_file', None)
            self.output_file_globi = getattr(self.args, 'output_file_globi', None)
            self.output_file_trydb = getattr(self.args, 'output_file_trydb', None)
            self.trydb_species_file = getattr(self.args, 'inputFile', None) # Renamed for TRY-db script
            self.wd_sparql_file = getattr(self.args, 'wd_sparql_file', None)
            self.wd_lineage_file = getattr(self.args, 'wd_lineage_file', None)
            self.wd_repeats_file = getattr(self.args, 'wd_repeats_file', None)
            """# If not passed directly, try to get from args
            if not self.wd_sparql_file: self.wd_sparql_file = getattr(self.args, 'wd_sparql_file', None)
            if not self.wd_lineage_file: self.wd_lineage_file = getattr(self.args, 'wd_lineage_aligned_file', None)
            if not self.wd_repeats_file: self.wd_repeats_file = getattr(self.args, 'wdLineageRepeats_file', None)"""

        # Basic validation for required files that are common to both pipelines or essential
        if not (self.wd_lineage_file and self.wd_sparql_file and self.wd_repeats_file):
             raise ValueError("Missing essential file paths: wd files are required.")


    # --- GloBI Specific Methods ---

    # Loads and processes the Wikidata SPARQL mapping file for GloBI matching.
    def _load_wd_sparql_data_globi(self):
        if not self.wd_sparql_file or not os.path.exists(self.wd_sparql_file):
            print(f"Warning: Wikidata SPARQL file not found at {self.wd_sparql_file}. Skipping GloBI SPARQL data loading.")
            return

        wd_sparql_df = pd.read_csv(self.wd_sparql_file, sep=",", dtype=str)

        for col_idx, prefix in self.prefixes.items():
            if col_idx < len(wd_sparql_df.columns):
                col_name = wd_sparql_df.columns[col_idx]
                if not wd_sparql_df[col_name].isnull().all():
                    wd_sparql_df[col_name] = prefix + wd_sparql_df[col_name].astype(str)

        wd_sparql_df.replace({"http://www.wikidata.org/entity/": "Wikidata:", '"': ''}, regex=True, inplace=True)

        cols_to_map = wd_sparql_df.columns[:-1]
        self.globi_id_map = (
            wd_sparql_df.melt(id_vars=wd_sparql_df.columns[-1], value_vars=cols_to_map, value_name="key")
            .dropna(subset=["key"])
            .set_index("key")[wd_sparql_df.columns[-1]]
            .to_dict()
        )
        # Assuming first column is WdID and last is WdName
        cols_to_map_WD = [col for col in wd_sparql_df.columns if col not in [wd_sparql_df.columns[0], wd_sparql_df.columns[-1]]]
        self.globi_id_map_wd = (
            wd_sparql_df.melt(id_vars=wd_sparql_df.columns[0], value_vars=cols_to_map_WD, value_name="key")
            .dropna(subset=["key"])
            .set_index("key")[wd_sparql_df.columns[0]]
            .to_dict()
        )

    # Loads and preprocesses the gzipped verbatim interactions file for GloBI matching.
    def _load_verbatim_globi_data(self):
        if not self.verbatim_file or not os.path.exists(self.verbatim_file):
            print(f"Warning: Verbatim GloBI file not found at {self.verbatim_file}. Skipping GloBI data loading.")
            return pd.DataFrame()

        verbatim_globi_df = pd.read_csv(
            self.verbatim_file,
            usecols=[
                'sourceTaxonId', 'sourceTaxonName', 'sourceTaxonPathNames', 'sourceTaxonPathRankNames',
                'targetTaxonId', 'targetTaxonName', 'targetTaxonPathNames', 'targetTaxonPathRankNames'
            ],
            sep="\t",
            lineterminator="\n",
            dtype=str,
            quoting=csv.QUOTE_NONE,
            encoding="iso-8859-1",
            escapechar='\\'
        )

        source_df = verbatim_globi_df[['sourceTaxonId', 'sourceTaxonName', 'sourceTaxonPathNames', 'sourceTaxonPathRankNames']].copy()
        source_df.columns = ["TaxonId", "TaxonName", "TaxonPathName", "TaxonRankName"]

        target_df = verbatim_globi_df[['targetTaxonId', 'targetTaxonName', 'targetTaxonPathNames', 'targetTaxonPathRankNames']].copy()
        target_df.columns = ["TaxonId", "TaxonName", "TaxonPathName", "TaxonRankName"]

        verbatim_globi_df = pd.concat([source_df, target_df], axis=0).reset_index(drop=True)

        verbatim_globi_df.replace({
            "https://www.wikidata.org/wiki/": "Wikidata:",
            "https://www.wikidata.org/entity/": "Wikidata:",
            "urn:lsid:marinespecies.org:taxname": "WORMS",
            "urn:lsid:irmng.org:taxname": "IRMNG",
            "http://www.boldsystems.org/index.php/Public_BarcodeCluster?clusteruri=BOLD": "BOLD",
            "https://www.itis.gov/servlet/SingleRpt/SingleRpt?search_topic=TSN&search_value=": "ITIS:",
            "https://www.inaturalist.org/taxa/": "INAT_TAXON:",
            "https://www.gbif.org/species/": "GBIF:",
            "https://species.nbnatlas.org/species/": "NBN:",
            "https://gd.eppo.int/taxon/": "EPPO:",
            r"^tsn": "ITIS",
            r"GBIF: \+": "GBIF:",
            r"gbif: \+": "GBIF:",
            "gbif:": "GBIF:",
        }, regex=True, inplace=True)
        verbatim_globi_df = verbatim_globi_df.drop_duplicates()

        expanded_verbatim_globi_df = verbatim_globi_df.apply(self._safe_extract_ranks, axis=1, result_type="expand")
        verbatim_globi_df = pd.concat([verbatim_globi_df, expanded_verbatim_globi_df], axis=1)
        return verbatim_globi_df

    # Detects clear mappings with names and IDs using pre-loaded maps for GloBI.
    # Also sets 'Match_Status' value.
    def _initial_tax_match_globi(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        taxon_id = df["TaxonId"].astype(str).str.strip()
        taxon_name = df["TaxonName"].astype(str).str.strip()

        df["Mapped_Value"] = taxon_id.map(self.globi_id_map)
        df["Mapped_ID_WD"] = taxon_id.map(self.globi_id_map_wd)
        df["Mapped_ID"] = taxon_id.where(df["Mapped_Value"].notna())

        df["Match_Status"] = (
            df["Mapped_Value"].str.lower() == taxon_name.str.lower()
        ).map({True: "NAME-MATCH-YES", False: "NAME-MATCH-NO"})

        df["Match_Status"] = df["Match_Status"].where(
            df["Mapped_Value"].notna(), "ID-NOT-FOUND"
        )
        df["Match_Status"] = df["Match_Status"].where(
            taxon_id.notna() & (taxon_id != ""), "ID-NOT-PRESENT"
        )
        return df

    # Loads and processes the Wikidata lineage and repeats files for GloBI matching. 
    def _load_wd_lineage_data_globi(self):
        if not (self.wd_lineage_file and os.path.exists(self.wd_lineage_file) and \
                self.wd_repeats_file and os.path.exists(self.wd_repeats_file)):
            print(f"Warning: Missing GloBI lineage files (aligned: {self.wd_lineage_file}, repeats: {self.wd_repeats_file}). Skipping GloBI lineage data loading.")
            return

        wd_lineage_df = pd.read_csv(self.wd_lineage_file, sep=",", dtype=str)
        wd_lineage_df["WdID"] = wd_lineage_df["WdID"].str.replace("http://www.wikidata.org/entity/", "Wikidata:", regex=False)
        self.globi_wd_name_to_id_set = set(wd_lineage_df["WdName"])

        wd_repeats_lineage = pd.read_csv(self.wd_repeats_file, dtype=str) # Ensure dtype=str
        wd_repeats_lineage = wd_repeats_lineage.fillna("")
        wd_repeats_lineage["WdID"] = wd_repeats_lineage["WdID"].str.replace("http://www.wikidata.org/entity/", "Wikidata:", regex=False)
        self.globi_wd_lineage_set = set(wd_repeats_lineage["WdName"])

        # Create a tuple key for the dictionary based on all relevant columns
        # Ensure all columns used in the key are strings before creating the tuple
        wd_repeats_lineage_keys = wd_repeats_lineage.apply(
            lambda x: (
                str(x["WdName"]),
                str(x.get("family", "")),
                str(x.get("class", "")),
                str(x.get("order", "")),
                str(x.get("phylum", "")),
                str(x.get("kingdom", ""))
            ), axis=1
        )
        self.globi_wd_lineage_dict = {
            key: wd_repeats_lineage.loc[wd_repeats_lineage_keys == key, "WdID"].tolist()
            for key in wd_repeats_lineage_keys.unique()
        }


        mask = ~wd_lineage_df["WdName"].isin(self.globi_wd_lineage_set)
        self.globi_wd_name_to_id = wd_lineage_df.loc[mask].set_index("WdName")[
            ["WdID", "family", "class", "order", "phylum", "kingdom"]
        ].apply(lambda x: tuple(x), axis=1).to_dict()

    # Scores and gets the best Wikidata ID according to rank matching
    # in the case of Wikidata repeats for GloBI matching.
    def _get_best_wikidata_id_globi(self, taxon_name, family, tax_class, order, phylum, kingdom):
        family = str(family) if pd.notna(family) else ""
        tax_class = str(tax_class) if pd.notna(tax_class) else ""
        order = str(order) if pd.notna(order) else ""
        phylum = str(phylum) if pd.notna(phylum) else ""
        kingdom = str(kingdom) if pd.notna(kingdom) else ""

        # Construct the key to look up in globi_wd_lineage_dict
        # The key must match the structure used when creating globi_wd_lineage_dict
        lookup_key_base = (str(taxon_name), family, tax_class, order, phylum, kingdom)

        # Check if this exact key exists in the dictionary
        if lookup_key_base in self.globi_wd_lineage_dict:
            return (lookup_key_base, self.globi_wd_lineage_dict[lookup_key_base])

        # If not an exact match, proceed with scoring logic for partial matches within duplicates
        possible_keys = [k for k in self.globi_wd_lineage_dict.keys() if k[0] == taxon_name and pd.notna(k[0]) and k[0] != ""]
        best_match_key = None
        best_score = -1

        for key in possible_keys:
            score = 0
            # Ensure comparison values are notna and not empty strings
            score += 1 if (pd.notna(key[1]) and key[1] != "" and str(key[1]).lower() == family.lower()) else 0
            score += 1 if (pd.notna(key[2]) and key[2] != "" and str(key[2]).lower() == tax_class.lower()) else 0
            score += 1 if (pd.notna(key[3]) and key[3] != "" and str(key[3]).lower() == order.lower()) else 0
            score += 1 if (pd.notna(key[4]) and key[4] != "" and str(key[4]).lower() == phylum.lower()) else 0
            score += 1 if (pd.notna(key[5]) and key[5] != "" and str(key[5]).lower() == kingdom.lower()) else 0

            if score > best_score:
                best_match_key = key
                best_score = score
        return (best_match_key, self.globi_wd_lineage_dict[best_match_key]) if best_match_key else (None, None)

    # Processes rows with "ID-NOT-FOUND" or "ID-NOT-PRESENT" status for GloBI matching
    # by checking through repeated names, direct matches, or still not found.
    def _process_row_for_id_not_found_globi(self, row: pd.Series) -> pd.Series:
        taxon_name = row["TaxonName"].strip() if pd.notna(row["TaxonName"]) else row["TaxonName"]
        best_wd_id = None
        status = row["Match_Status"]

        # Ensure taxon_name is valid for lookup
        if not pd.notna(taxon_name) or taxon_name == "":
            row["Mapped_ID_WD"] = best_wd_id
            row["Match_Status"] = status
            return row

        # Case 1: Check for repeated names in lineage (using the set of names that have duplicates)
        if taxon_name in self.globi_wd_lineage_set:
            family = row.get("family", "")
            tax_class = row.get("class", "")
            order = row.get("order", "")
            phylum = row.get("phylum", "")
            kingdom = row.get("kingdom", "")

            tempVar, tempVarX = self._get_best_wikidata_id_globi(taxon_name, family, tax_class, order, phylum, kingdom)

            if tempVar:
                best_wd_id = tempVarX[0] if isinstance(tempVarX, list) else tempVarX
                # Update ranks in the row based on the matched Wikidata entry
                row["family"] = tempVar[1] if pd.notna(tempVar[1]) else ""
                row["class"] = tempVar[2] if pd.notna(tempVar[2]) else ""
                row["order"] = tempVar[3] if pd.notna(tempVar[3]) else ""
                row["phylum"] = tempVar[4] if pd.notna(tempVar[4]) else ""
                row["kingdom"] = tempVar[5] if pd.notna(tempVar[5]) else ""
                status = "ID-MATCHED-BY-NAME-DUPL-duplicate"
            else:
                status = "ID-MATCHED-BY-NAME-DUPL-mismatch" # No best match found among duplicates
        
        # Case 2: Check for direct match in non-repeated lineage names
        elif taxon_name in self.globi_wd_name_to_id_set:
            tempVar = self.globi_wd_name_to_id.get(taxon_name, (None, None, None, None, None, None))
            best_wd_id = tempVar[0]
            row["family"] = tempVar[1] if pd.notna(tempVar[1]) else ""
            row["class"] = tempVar[2] if pd.notna(tempVar[2]) else ""
            row["order"] = tempVar[3] if pd.notna(tempVar[3]) else ""
            row["phylum"] = tempVar[4] if pd.notna(tempVar[4]) else ""
            row["kingdom"] = tempVar[5] if pd.notna(tempVar[5]) else ""
            status = "ID-MATCHED-BY-NAME-direct"
        
        # Case 3: Still not found, retain original status or set to specific 'not found'
        else:
            status = row["Match_Status"]

        row["Mapped_ID_WD"] = best_wd_id
        row["Match_Status"] = status
        return row

    # --- TRY-db Specific Methods ---

    # Loads and processes the Wikidata lineage file for TRY-db matching.
    def _load_wd_lineage_data_trydb(self):
        if not self.wd_lineage_file or not os.path.exists(self.wd_lineage_file):
            print(f"Warning: Wikidata lineage file not found at {self.wd_lineage_file} for TRY-db. Skipping TRY-db lineage data loading.")
            return

        wd_lineage_df = pd.read_csv(self.wd_lineage_file, usecols=['WdID','WdName','kingdom'], sep=",", dtype=str)
        wd_lineage_df["WdID"] = wd_lineage_df["WdID"].str.replace("http://www.wikidata.org/entity/", "", regex=False)
        wd_lineage_df["kingdom"] = wd_lineage_df["kingdom"].replace({np.nan: None, pd.NA: None, "": None})

        self.trydb_wd_name_to_id_set = set(wd_lineage_df["WdName"])
        self.trydb_wd_name_to_id = (
            wd_lineage_df.set_index(["WdName", "kingdom"])["WdID"]
            .to_dict()
        )

    # Aligns TRY-db species names directly by name using the pre-loaded
    # Wikidata lineage data specific to TRY-db.
    # Ensure TRY_AccSpeciesName is a string and not NaN
    def _process_trydb_row(self, row: pd.Series) -> pd.Series:
        acc_species_name = str(row['TRY_AccSpeciesName']) if pd.notna(row['TRY_AccSpeciesName']) else None

        best_wd_id = None
        kingdomV = None
        status = "NAME-NOT-MATCHED"

        if acc_species_name and acc_species_name in self.trydb_wd_name_to_id_set:
            # Try to match with 'Plantae' kingdom first
            tempVar = self.trydb_wd_name_to_id.get((acc_species_name, 'Plantae'))
            if tempVar:
                best_wd_id = tempVar
                kingdomV = "Plantae"
            else:
                # If not 'Plantae', try with np.nan (representing None/unspecified kingdom)
                best_wd_id = self.trydb_wd_name_to_id.get((acc_species_name, np.nan))
                kingdomV = "None" # Or keep as None if you want to reflect the lack of kingdom match
            status = "ID-MATCHED-BY-NAME-direct"

        row["WdID"] = best_wd_id
        row["Match_Status"] = status
        row["kingdom"] = kingdomV
        return row

    # Executes the TRY-db specific taxonomy matching pipeline.
    def run_trydb_pipeline(self):
        if not self.trydb_species_file or not os.path.exists(self.trydb_species_file):
            print(f"Warning: TRY-db species file not found at {self.trydb_species_file}. Skipping TRY-db pipeline.")
            return pd.DataFrame()

        start_time = time.time()
        print("Starting TRY-db taxonomy matching pipeline...")

        # Load TRY-db specific lookup data
        print("Loading Wikidata lineage data for TRY-db matching...")
        self._load_wd_lineage_data_trydb()
        print("TRY-db specific lineage data loaded.")

        # Process tryDb file
        print("Processing TRY-db species file...")
        tryDb_df = pd.read_csv(self.trydb_species_file, usecols=['TRY_SpeciesName','TRY_AccSpeciesName'], sep=",", dtype=str, encoding="iso-8859-1")
        # Ensure unique TRY_AccSpeciesName as in original script
        d = { 'TRY_AccSpeciesName' : list(set(tryDb_df['TRY_AccSpeciesName']))}
        tryDb_df_unique_acc_names = pd.DataFrame(d)

        tryDb_df_processed = tryDb_df_unique_acc_names.apply(self._process_trydb_row, axis=1)
        print("TRY-db species file processed.")

        # Save output
        today_str = datetime.today().strftime('%Y%m%d')
        output_file_path = f"{self.output_file_trydb}" # Differentiate output file
        print(f"Saving TRY-db results to {output_file_path}...")
        tryDb_df_processed.to_csv(output_file_path, sep="\t", index=False, compression='gzip')
        print(f"TRY-db pipeline finished in {time.time() - start_time:.2f} seconds.")

        return tryDb_df_processed

    # --- Common Helper Methods (used by both or general) ---
    # Helper function to extract ranks into separate columns.
    def _extract_ranks(self, structure: str, values: str) -> dict:
        rank_list = [r.strip() for r in structure.split("|")]
        value_list = [v.strip() for v in values.split("|")]
        rank_dict = dict(zip_longest(rank_list, value_list, fillvalue=""))
        return {rank: rank_dict.get(rank, "") for rank in self.predefined_ranks}

    # Helper function to safely split taxon paths and rank names.
    def _safe_extract_ranks(self, row: pd.Series) -> pd.Series:
        if pd.notna(row["TaxonPathName"]) and pd.notna(row["TaxonRankName"]):
            return pd.Series(self._extract_ranks(row["TaxonRankName"], row["TaxonPathName"]))
        else:
            return pd.Series({rank: pd.NA for rank in self.predefined_ranks})

    # --- Main Pipeline Orchestration ---
    # Executes the full GloBI specific taxonomy matching pipeline.
    # Check if all required files for GloBI are available
    def run_globi_pipeline(self):
        required_globi_files = [self.wd_sparql_file, self.verbatim_file, self.wd_lineage_file, self.wd_repeats_file]
        if not all(f and os.path.exists(f) for f in required_globi_files):
            missing_files = [f for f in required_globi_files if not (f and os.path.exists(f))]
            print(f"Warning: Missing required files for GloBI pipeline: {missing_files}. Skipping GloBI pipeline.")
            return pd.DataFrame()

        start_time = time.time()
        print("Starting GloBI taxonomy matching pipeline...")

        # Load and process Wikidata SPARQL data
        print("Loading Wikidata SPARQL data for GloBI...")
        self._load_wd_sparql_data_globi()
        print("Wikidata SPARQL data loaded and processed for GloBI.")

        # Load and preprocess Verbatim GloBI data
        print("Loading Verbatim GloBI data...")
        verbatim_globi_df = self._load_verbatim_globi_data()
        print("Verbatim GloBI data loaded and preprocessed.")

        # First layer of matching (ID-based, direct name match)
        print("Performing initial taxonomy matching for GloBI...")
        verbatim_globi_df = self._initial_tax_match_globi(verbatim_globi_df)
        print("Initial taxonomy matching complete for GloBI.")

        # Load Wikidata lineage data for secondary matching
        print("Loading Wikidata lineage data for GloBI...")
        self._load_wd_lineage_data_globi()
        print("Wikidata lineage data loaded for GloBI.")

        # Second layer of matching (for ID-NOT-FOUND/ID-NOT-PRESENT)
        print("Performing secondary taxonomy matching for unmatched GloBI IDs...")
        mask_not_found = verbatim_globi_df["Match_Status"] == "ID-NOT-FOUND"
        mask_not_present = verbatim_globi_df["Match_Status"] == "ID-NOT-PRESENT"

        verbatim_globi_df.loc[mask_not_found | mask_not_present] = \
            verbatim_globi_df.loc[mask_not_found | mask_not_present].apply(self._process_row_for_id_not_found_globi, axis=1)
        print("Secondary taxonomy matching complete for GloBI.")

        # Final output
        #today_str = datetime.today().strftime('%Y%m%d')
        output_file_path = f"{self.output_file_globi}" # add date if required
        print(f"Saving GloBI results to {output_file_path}...")
        verbatim_globi_df.to_csv(output_file_path, index=False)
        print(f"GloBI pipeline finished in {time.time() - start_time:.2f} seconds.")

        return verbatim_globi_df

# Main execution block
if __name__ == "__main__":
    # --- Argument Parsing for both fetcher and matcher ---
    parser = argparse.ArgumentParser(description="Taxonomy matching pipeline.")

    # Arguments for WikidataDataFetcher (optional, if you want to run fetcher from here)
    parser.add_argument('--fetch_data', action='store_true', help="Run Wikidata data fetching pipeline before matching.")
    parser.add_argument('--fetch_output_dir', type=str, default=".", help="Directory for Wikidata fetched files.")
    parser.add_argument('--fetch_chunk_size', type=int, default=30000000, help="Chunk size for fetching large CSVs.") # remember to change chunk size according to number of lines in lineage_aligned file

    # Arguments for TaxonomyMatcher (can be paths from fetcher or direct input)
    parser.add_argument('--wd_sparql_file', type=str, help="Path to Wikidata mappings file (e.g., wdTax_SPARQL_YYYYMMDD.txt.gz).")
    parser.add_argument('--verbatim_file', type=str, help="Path to verbatim interactions file for GloBI.")
    parser.add_argument('--wd_lineage_aligned_file', type=str, help="Path to Wikidata aligned lineage file (e.g., wdTax_SPARQL_lineage_filtered_aligned_YYYYMMDD.txt.gz).")
    parser.add_argument('--wdLineageRepeats_file', type=str, help="Path to file with taxonomic names repeated in Wikidata lineage (e.g., wdTax_duplicates_YYYYMMDD.txt).")
    parser.add_argument('--inputFile', type=str, help="Path to TRY-db gzip species file.")
    parser.add_argument('--output_file_globi', type=str, help="Base output file name (will be suffixed for GloBI/TRY-db).")
    parser.add_argument('--output_file_trydb', type=str, help="Base output file name (will be suffixed for GloBI/TRY-db).")
    parser.add_argument('--config', type=str, default="config.txt", help="Path to the configuration file.")

    args = parser.parse_args()

    # --- Run Wikidata Data Fetcher if requested ---
    generated_files = {}
    if args.fetch_data:
        print("Running Wikidata Data Fetcher...")
        from wikidata_fetcher import WikidataDataFetcher
        fetcher = WikidataDataFetcher(output_dir=args.fetch_output_dir, chunk_size=args.fetch_chunk_size) # remember to change chunk size according to number of l    ines in lineage_aligned file
        generated_files = fetcher.run_fetch_pipeline()
        print("\nWikidata Data Fetcher completed.")

    # --- Prepare arguments for TaxonomyMatcher ---
    # Prioritize files generated by the fetcher
    wd_sparql_file_path = generated_files.get("wd_sparql_file") or args.wd_sparql_file
    wd_lineage_aligned_file_path = generated_files.get("wd_lineage_aligned_file") or args.wd_lineage_aligned_file
    wd_repeats_file_path = generated_files.get("wd_repeats_file") or args.wdLineageRepeats_file

    # Create an instance of the matcher
    matcher = TaxonomyMatcher(
        config_file=args.config,
        args=args, # Pass args so it can fallback to them for other files
        wd_sparql_file=wd_sparql_file_path,
        wd_lineage_aligned_file=wd_lineage_aligned_file_path,
        wd_repeats_file=wd_repeats_file_path
    )

    # --- Run Matching Pipelines ---
    print("\n--- Running Taxonomy Matching Pipelines ---")
    globi_df_result = matcher.run_globi_pipeline()
    trydb_df_result = matcher.run_trydb_pipeline()

    print("\nAll pipeline executions complete.")
