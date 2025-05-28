import pandas as pd
from rdflib import URIRef, Literal, Namespace, RDF, RDFS, XSD, DCTERMS, Graph, BNode
import gzip
import rdflib
import argparse
import sys
import configparser
import os
import re

from src.common.utils import format_uri, is_none_na_or_empty, create_dict_from_csv, add_inverse_relationships, print_to_log

rdflib.plugin.register('turtle_custom', rdflib.plugin.Serializer, 'turtle_custom.serializer', 'TurtleSerializerCustom')

# Namespace declarations
EMI = Namespace("https://purl.org/emi#")
EMI_UNIT = Namespace("https://purl.org/emi/unit#")
EMI_BOX = Namespace("https://purl.org/emi/abox#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
DCTERMS = Namespace("http://purl.org/dc/terms/")
WD = Namespace("http://www.wikidata.org/entity/")
QUDT = Namespace("https://qudt.org/2.1/schema/qudt/")
QUDT_UNIT = Namespace("http://qudt.org/vocab/unit/")

# Generates RDF triples from TRY database data.
class TrydbKGGenerator:
    # Initializes the TryDbRdfGenerator.
    #    Args:
    #        config_file (str): Path to the configuration file.
    def __init__(self, config_file: str):
        self.config = configparser.ConfigParser()
        if os.path.exists(config_file):
            self.config.read(config_file)
            self.trydb_tsv = self.config.get('input tsv files', 'trydb_file')
            self.trydb_wd_map = self.config.get('accessory files', 'trydb_wd')
            self.enpkg_wd_join = self.config.get('accessory files', 'enpkg_wd')
            self.qudt_unit_map_file = self.config.get('knowledge graph files', 'dictFileNameQudt')
            self.emi_unit_map_file = self.config.get('knowledge graph files', 'dictFileNameEmi')
            self.output_ttl = self.config.get('output files', 'trydb_ttl')
        else:
            raise FileNotFoundError(f"Config file not found: {config_file}")

        self.qudt_unit_map = {}
        self.emi_unit_map = {}

    # Loads the unit mappings from CSV files.
    def _load_unit_mappings(self):
        self.qudt_unit_map = create_dict_from_csv(self.qudt_unit_map_file, "origUnit", "mapUnit")
        self.emi_unit_map = create_dict_from_csv(self.emi_unit_map_file, "origUnit", "mapUnit")

    def generate_rdf(self, join_column_species: str, join_column_wd: str, batch_size: int = 10000, filter_with_enpkg: bool = False):
        """
        Generates RDF triples from the TRY database TSV file in batches.

        Args:
            join_column_species (str): Column name in the TRY data for species name to join with WD mapping.
            join_column_wd (str): Column name in the ENPKG data for WD taxon IDs to filter with.
            batch_size (int): Number of rows to process in each batch.
            filter_with_enpkg (bool): Whether to filter the TRY data based on WD IDs in the ENPKG file.
        """
        self._load_unit_mappings()

        trydb_data = pd.read_csv(self.trydb_tsv, compression="gzip", sep="\t", dtype=str, encoding="iso-8859-1")
        wd_mapping_data = pd.read_csv(self.trydb_wd_map, compression="gzip", sep="\t", dtype=str)

        merged_data = pd.merge(trydb_data, wd_mapping_data[[join_column_species, "WdID"]],
                               left_on="AccSpeciesName", right_on=join_column_species, how="left")
        merged_data.drop(columns=[join_column_species], inplace=True)

        if filter_with_enpkg:
            enpkg_data = pd.read_csv(self.enpkg_wd_join, compression="gzip", sep="\t", dtype=str)
            merged_data = merged_data[merged_data['WdID'].isin(enpkg_data[join_column_wd])]

        total_trip_count = 0

        with gzip.open(self.output_ttl, "wt", encoding="utf-8") as out_file:
            out_file.write(f"@prefix emi: <{EMI}> .\n")
            out_file.write(f"@prefix emiUnit: <{EMI_UNIT}> .\n")
            out_file.write(f"@prefix : <{EMI_BOX}> .\n")
            out_file.write(f"@prefix sosa: <{SOSA}> .\n")
            out_file.write(f"@prefix dcterms: <{DCTERMS}> .\n")
            out_file.write(f"@prefix wd: <{WD}> .\n")
            out_file.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
            out_file.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
            out_file.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n")
            out_file.write(f"@prefix qudt: <{QUDT}> .\n")
            out_file.write(f"@prefix qudtUnit: <{QUDT_UNIT}> .\n\n")

        for start_row in range(0, len(merged_data), batch_size):
                end_row = min(start_row + batch_size, len(merged_data))
                batch_data = merged_data[start_row:end_row]
                graph = Graph()
                graph.bind("", EMI_BOX)
                graph.bind("emi", EMI)
                graph.bind("emiUnit", EMI_UNIT)
                graph.bind("sosa", SOSA)
                graph.bind("dcterms", DCTERMS)
                graph.bind("wd", WD)
                graph.bind("qudt", QUDT)
                graph.bind("qudtUnit", QUDT_UNIT)

                batch_trip_count = 0
                for _, row in batch_data.iterrows():
                    sample_uri = EMI_BOX[f"SAMPLE-{format_uri(row['AccSpeciesName'])}-{row['ObservationID']}"]
                    dataset_uri = EMI_BOX[f"DATASET-{format_uri(row['Dataset'])}"] if is_none_na_or_empty(row['Dataset']) else None
                    observation_uri = EMI_BOX[f"OBSERVATION-{format_uri(row['ObservationID'])}"]
                    organism_uri = EMI_BOX[f"ORGANISM-{format_uri(row['AccSpeciesName'])}"]
                    result_bnode = EMI_BOX[f"RESULT-{row['ObsDataID']}"] if is_none_na_or_empty(row['Dataset']) else None

                    graph.add((sample_uri, RDF.type, SOSA.Sample))
                    graph.add((sample_uri, RDFS.label, Literal(row['AccSpeciesName'], datatype=XSD.string)))
                    graph.add((sample_uri, SOSA.isSampleOf, organism_uri))
                    graph.add((sample_uri, SOSA.isFeatureOfInterestOf, observation_uri))
                    batch_trip_count += 4

                    if is_none_na_or_empty(dataset_uri):
                        graph.add((sample_uri, DCTERMS.isPartOf, dataset_uri))
                        graph.add((dataset_uri, DCTERMS.bibliographicCitation, Literal(row['Reference'], datatype=XSD.string)))
                        batch_trip_count += 2

                    graph.add((observation_uri, SOSA.hasResult, result_bnode))
                    if is_none_na_or_empty(result_bnode):
                        if is_none_na_or_empty(row['TraitName']):
                            graph.add((result_bnode, RDF.type, EMI.Trait))
                            batch_trip_count += 1
                            if is_none_na_or_empty(row['OrigValueStr']):
                                pattern = r"-?[0-9]+(\.[0-9]+)?(E[+-][0-9]+)?"
                                if re.fullmatch(pattern, row['OrigValueStr']):
                                    graph.add((result_bnode, RDF.value, Literal(row['OrigValueStr'], datatype=XSD.double)))
                                else:
                                    graph.add((result_bnode, RDF.value, Literal(row['OrigValueStr'], datatype=XSD.string)))
                                batch_trip_count += 1
                        else:
                            graph.add((result_bnode, RDF.type, EMI.NonTrait))
                            batch_trip_count += 1
                            if is_none_na_or_empty(row['OrigValueStr']):
                                graph.add((result_bnode, RDF.value, Literal(row['OrigValueStr'], datatype=XSD.string)))
                                batch_trip_count += 1
                    if is_none_na_or_empty(row['DataName']):
                        graph.add((result_bnode, RDFS.label, Literal(row['DataName'], datatype=XSD.string)))
                        batch_trip_count += 1
                    if is_none_na_or_empty(row['DataID']):
                        graph.add((result_bnode, DCTERMS.identifier, Literal(row['DataID'], datatype=XSD.string)))
                        batch_trip_count += 1

                    if is_none_na_or_empty(row['OrigUnitStr']):
                            unit_str = row['OrigUnitStr']
                            if unit_str in self.qudt_unit_map:
                                graph.add((result_bnode, QUDT.hasUnit, URIRef(QUDT_UNIT[self.qudt_unit_map[unit_str]])))
                                batch_trip_count += 1
                            elif is_none_na_or_empty(row['UnitName']):
                                unit_name = row['UnitName']
                                if unit_name in self.qudt_unit_map:
                                    graph.add((result_bnode, QUDT.hasUnit, URIRef(QUDT_UNIT[self.qudt_unit_map[unit_name]])))
                                    batch_trip_count += 1
                                elif unit_name in self.emi_unit_map:
                                    graph.add((result_bnode, QUDT.hasUnit, URIRef(self.emi_unit_map[unit_name])))
                                    batch_trip_count += 1
                            elif unit_str in self.emi_unit_map:
                                graph.add((result_bnode, QUDT.hasUnit, URIRef(self.emi_unit_map[unit_str])))
                                batch_trip_count += 1
                            graph.add((result_bnode, RDFS.comment, Literal(unit_str.strip(), datatype=XSD.string)))
                            batch_trip_count += 1


                    if pd.notna(row['WdID']):
                        graph.add((organism_uri, EMI.inTaxon, URIRef(WD[format_uri(row['WdID'])])))
                        batch_trip_count += 1

                batch_trip_count = add_inverse_relationships("trydb-kg-log.txt",graph, batch_trip_count)
                total_trip_count += batch_trip_count

                try:
                    serialized_data = graph.serialize(format="turtle_custom")
                    with gzip.open(self.output_ttl, "at", encoding="utf-8") as out_file:
                        out_file.write(serialized_data)
                except Exception as e:
                    print(f"Error serializing batch from row {start_row}: {e}")
                    break
                del graph

        print_to_log(f"{total_trip_count} RDF triples saved to {self.output_ttl}","trydb-kg-log.txt")
