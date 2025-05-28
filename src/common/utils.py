import pandas as pd
import gzip
import re
import os
import datetime
from urllib.parse import quote
from rdflib import URIRef, Literal, Namespace, RDF, RDFS, XSD, DCTERMS, Graph, BNode # Keep imports for functions that use them
import rdflib # For turtle plugin registration


# Register rdflib plugin once when utils is imported
# Assuming 'turtle_custom.serializer' exists and is in sys.path
try:
    rdflib.plugin.register('turtle_custom', rdflib.plugin.Serializer, 'turtle_custom.serializer', 'TurtleSerializerCustom')
except Exception as e:
    print(f"Warning: Could not register rdflib plugin 'turtle_custom'. If this is not intended, check your setup. Error: {e}")


# Function for checking na/none/empty strings
def is_none_na_or_empty(value):
    # Checks if a value is None, NaN, empty string, or specific GloBI placeholder.
    return not (value is None or value == '' or value == "\\N" or value == "no:match" or pd.isna(value) or re.match(r"ĜLOBI:", str(value)))


# Define a function for real-time filtering. Reads a gzipped TSV file in chunks and filters rows based on a key column
def filter_file_runtime(file_path, filter_df, key_column):
    # matching values in a filter DataFrame.
    cs = 10000  # Adjust chunk size. TBD - put in config file
    matching_rows = pd.DataFrame()

    print(f"Filtering file {file_path} by '{key_column}'...")
    try:
        for chunk in pd.read_csv(file_path, compression="gzip", sep="\t", dtype=str, encoding="utf-8", chunksize=cs):
            # Ensure key_column exists in chunk
            if key_column not in chunk.columns:
                print(f"Warning: Key column '{key_column}' not found in chunk. Skipping chunk.")
                continue

            # Filter rows where 'key_column' matches values in filter_df
            # Assuming filter_df[key_column] is a Series/list of values to check against
            filtered_chunk = chunk[chunk['source_WD'].isin(filter_df[key_column]) | chunk['target_WD'].isin(filter_df[key_column])]

            matching_rows = pd.concat([matching_rows, filtered_chunk], ignore_index=True)
        print(f"Filtering complete. Found {len(matching_rows)} matching rows.")
        return matching_rows
    except Exception as e:
        print(f"Error during runtime file filtering: {e}")
        raise


# Define a function for real-time filtering by phylum/kingdom name
# Reads a gzipped TSV file in chunks and filters rows based on
# specific phylum or kingdom names in source/target columns.
def filter_file_runtime_taxonomy(file_path):
    cs = 10000  # Adjust chunk size as needed
    matching_rows = pd.DataFrame()
    phylum_names = ["Arthropoda", "Nematoda"]
    kingdom_names = ["Archaeplastida"]

    print(f"Filtering file {file_path} by taxonomy names...")
    try:
        for chunk in pd.read_csv(file_path, compression="gzip", sep="\t", dtype=str, encoding="utf-8", chunksize=cs):
            # Ensure relevant columns exist
            required_cols = ['targetTaxonKingdomName', 'sourceTaxonKingdomName', 'targetTaxonPhylumName', 'sourceTaxonPhylumName']
            if not all(col in chunk.columns for col in required_cols):
                print(f"Warning: Missing required taxonomy columns in chunk. Expected: {required_cols}, Found: {list(chunk.columns)}. Skipping chunk.")
                continue

            filtered_chunk = chunk[
                chunk['targetTaxonKingdomName'].isin(kingdom_names) |
                chunk['sourceTaxonKingdomName'].isin(kingdom_names) |
                chunk['targetTaxonPhylumName'].isin(phylum_names) |
                chunk['sourceTaxonPhylumName'].isin(phylum_names)
            ]
            matching_rows = pd.concat([matching_rows, filtered_chunk], ignore_index=True)
        print(f"Taxonomy filtering complete. Found {len(matching_rows)} matching rows.")
        return matching_rows
    except Exception as e:
        print(f"Error during runtime taxonomy filtering: {e}")
        raise


# Adds inverse relationships to the RDF graph based on predefined mappings.
def add_inverse_relationships(logFile, graph: Graph, tripCount: int) -> int:
    from src.common.constants import INVERSE_RELATIONS # Import here to avoid circular dependency if constants imports utils

    new_triples = []
    #print_to_log("Adding inverse relationships to the graph...", logFile)
    for subj, pred, obj in graph:
        pred_str = str(pred)
        if pred_str in INVERSE_RELATIONS:
            inverse_pred = URIRef(INVERSE_RELATIONS[pred_str])
            if isinstance(obj, URIRef):  # Only create inverses for URI objects
                new_triples.append((obj, inverse_pred, subj))

    for triple in new_triples:
        graph.add(triple)
        tripCount += 1
    #print_to_log(f"Added {len(new_triples)} inverse relationships.", logFile)
    return tripCount


# Formats the URI part by replacing spaces with underscores and encoding special characters.
def format_uri(uri_part: str) -> str:
    encoded_string = quote(uri_part, safe="")
    return encoded_string


# Creates a dictionary from two columns in a CSV file using pandas.
def create_dict_from_csv(csv_file: str, key_column: str, value_column: str) -> dict:
    if not os.path.exists(csv_file):
        print(f"Warning: Mapping file not found at {csv_file}. Returning empty dictionary.")
        return {}
    try:
        df = pd.read_csv(csv_file, sep=",", dtype=str)
        if key_column not in df.columns or value_column not in df.columns:
            raise ValueError(f"Required columns '{key_column}' or '{value_column}' not found in {csv_file}")
        return dict(zip(df[key_column], df[value_column]))
    except Exception as e:
        print(f"Error loading dictionary from {csv_file}: {e}")
        return {}

# Converts term to lowercase, strips whitespace, and handles simple plural removal.
def preprocess_term(term: str) -> str:
    term = str(term).lower().strip()
    if "mono" not in term and "auto" not in term: # Specific logic from original code
        if term.endswith('s'):
            term = term[:-1]
    return term


# Appends the current date and time followed by the message to the log file.
def print_to_log(message, filename="output.log"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    with open(filename, "a") as log_file:
        log_file.write(log_entry)
