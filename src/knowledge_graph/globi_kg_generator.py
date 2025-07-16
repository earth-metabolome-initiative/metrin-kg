import pandas as pd
from rdflib import URIRef, Literal, Namespace, RDF, RDFS, XSD, DCTERMS, Graph, BNode
import rdflib
import gzip
import argparse
import configparser
import sys
import re
import os

from src.common.utils import format_uri, is_none_na_or_empty, add_inverse_relationships, print_to_log
from src.knowledge_graph.globi_entity_matcher import GlobiEntityMatcher

rdflib.plugin.register('turtle_custom', rdflib.plugin.Serializer, 'turtle_custom.serializer', 'TurtleSerializerCustom')

# Namespace declarations
EMI = Namespace("https://w3id.org/emi#")
EMIBOX = Namespace("https://w3id.org/emi/abox#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
DCTERMS = Namespace("http://purl.org/dc/terms/")
WD = Namespace("http://www.wikidata.org/entity/")
PROV = Namespace("http://www.w3.org/ns/prov#")
WGS84 = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
QUDT = Namespace("http://qudt.org/schema/qudt/")

# Namespaces for specific entity types
AEO = Namespace("http://purl.obolibrary.org/obo/AEO_")
CHEBI = Namespace("http://purl.obolibrary.org/obo/CHEBI_")
CLYH = Namespace("http://purl.obolibrary.org/obo/CLYH_")
ENVO = Namespace("http://purl.obolibrary.org/obo/ENVO_")
FAO = Namespace("http://purl.obolibrary.org/obo/FAO_")
FBDV = Namespace("http://purl.obolibrary.org/obo/FBdv_")
HAO = Namespace("http://purl.obolibrary.org/obo/HAO_")
NCIT = Namespace("http://purl.obolibrary.org/obo/NCIT_")
OMIT = Namespace("http://purl.obolibrary.org/obo/OMIT_")
PATO = Namespace("http://purl.obolibrary.org/obo/PATO_")
PO = Namespace("http://purl.obolibrary.org/obo/PO_")
PORO = Namespace("http://purl.obolibrary.org/obo/PORO_")
RO = Namespace("http://purl.obolibrary.org/obo/RO_")
UBERON = Namespace("http://purl.obolibrary.org/obo/UBERON_")

PREFIX_TO_NAMESPACE = {
    "AEO:": AEO,
    "CHEBI:": CHEBI,
    "CLYH:": CLYH,
    "ENVO:": ENVO,
    "FAO:": FAO,
    "FBdv:": FBDV,
    "HAO:": HAO,
    "NCIT:": NCIT,
    "OMIT:": OMIT,
    "PATO:": PATO,
    "PORO:": PORO,
    "RO:": RO,
    "UBERON:": UBERON,
    "PO:": PO,
    "QUDT:": QUDT
}

# Generates RDF triples from GloBI interaction data.
class GlobiKGGenerator:
    # Initializes the GlobiRdfGenerator.
    #    Args:
    #        config_file (str): Path to the configuration file.
    def __init__(self, config_file: str):
        self.config = configparser.ConfigParser()
        if os.path.exists(config_file):
            self.config.read(config_file)
            self.globi_tsv = self.config.get('input tsv files', 'globi_verbatim_file')
            self.globi_wd_map = self.config.get('accessory files', 'globi_wd')
            self.enpkg_wd_join = self.config.get('accessory files', 'enpkg_wd')
            self.output_ttl = self.config.get('output files', 'globi_ttl')
            self.biological_sex_map = self.config.get('knowledge graph files', 'bs_fileName')
        else:
            raise FileNotFoundError(f"Config file not found: {config_file}")

        self.entity_matcher = GlobiEntityMatcher(config_file, self.biological_sex_map)
        self.intxn_type_set = set()
        self.biological_sex_set = set()
        self.life_stage_set = set()
        self.body_part_set = set()
        self.wd_map_dict_id = {}
        self.wd_map_dict_name = {}
        self.wd_map_set_id = set()
        self.wd_map_set_name = set()

    # Loads the mapping of GloBI IDs and names to Wikidata.
    def _load_wikidata_mappings(self):
        wd_map_df = pd.read_csv(self.globi_wd_map, sep=",", dtype=str, usecols=['TaxonId', 'TaxonName', 'Mapped_ID_WD', 'Mapped_Value'])
        wd_map_df.replace({"Wikidata:": '', '"': ''}, regex=True, inplace=True)
        wd_map_df = wd_map_df.dropna(subset=["Mapped_ID_WD"]).query("Mapped_ID_WD != ''")

        self.wd_map_dict_id = (
            wd_map_df.dropna(subset=["TaxonId"])
            .query("TaxonId != ''")
            .set_index("TaxonId")[["Mapped_ID_WD", "Mapped_Value"]]
            .apply(tuple, axis=1)
            .to_dict()
        )

        self.wd_map_dict_name = (
            wd_map_df.dropna(subset=["TaxonName"])
            .query("TaxonName != ''")
            .set_index("TaxonName")[["Mapped_ID_WD", "Mapped_Value"]]
            .apply(tuple, axis=1)
            .to_dict()
        )

        self.wd_map_set_id = set(wd_map_df["TaxonId"].dropna().replace("", None).dropna())
        self.wd_map_set_name = set(wd_map_df["TaxonName"].dropna().replace("", None).dropna())

    # Adds an entity to the RDF graph.
    def _add_entity_to_graph(self, entity_name: str, entity_id: str, subject: URIRef, predicate: URIRef, rdf_type: URIRef, namespace_prefix: str, graph: Graph, entity_set: set, trip_count: int) -> int:
        if is_none_na_or_empty(entity_id):
            if any(entity_id.startswith(prefix) for prefix in PREFIX_TO_NAMESPACE):
                for prefix, namespace in PREFIX_TO_NAMESPACE.items():
                    if entity_id.startswith(prefix):
                        entity_local_id = entity_id[len(prefix):]
                        entity_uri = namespace[entity_local_id]
                        graph.add((subject, predicate, entity_uri))
                        trip_count += 1
                        if entity_uri not in entity_set:
                            graph.add((entity_uri, RDF.type, rdf_type))
                            graph.add((entity_uri, RDFS.label, Literal(entity_name, datatype=XSD.string)))
                            trip_count += 2
                            entity_set.add(entity_uri)
                        return trip_count
            elif entity_id.startswith("http"):
                entity_uri = URIRef(entity_id)
                graph.add((subject, predicate, entity_uri))
                trip_count += 1
                if entity_uri not in entity_set:
                    graph.add((entity_uri, RDF.type, rdf_type))
                    graph.add((entity_uri, RDFS.label, Literal(entity_name, datatype=XSD.string)))
                    trip_count += 2
                    entity_set.add(entity_uri)
                return trip_count

        self.entity_matcher._lookup_and_add_term(entity_name, subject, predicate, rdf_type, entity_name)
        return self.entity_matcher.get_triple_count() # Update trip count from the matcher

    # Generates RDF triples from the GloBI TSV file in batches.
    def generate_rdf(self, batch_size: int = 100000, checkpoint_file: str = "checkpoint.txt"):
        self._load_wikidata_mappings()

        start_index = 0
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                start_index = int(f.read().strip())
        current_index = start_index
        total_trip_count = 0
        records_count = 0

        with gzip.open(self.output_ttl, "wt", encoding="utf-8") as out_file:
            out_file.write(f"@prefix emi: <{EMI}> .\n")
            out_file.write(f"@prefix : <{EMIBOX}> .\n")
            out_file.write(f"@prefix sosa: <{SOSA}> .\n")
            out_file.write(f"@prefix dcterms: <{DCTERMS}> .\n")
            out_file.write(f"@prefix wd: <{WD}> .\n")
            out_file.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
            out_file.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
            out_file.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n")
            out_file.write(f"@prefix prov: <{PROV}> .\n")
            out_file.write(f"@prefix wgs84: <{WGS84}> .\n")
            out_file.write(f"@prefix qudt: <{QUDT}> .\n\n")

        chunks = pd.read_csv(self.globi_tsv, sep="\t", compression="gzip", chunksize=batch_size, dtype=str, encoding="utf-8")
        for i, batch_data in enumerate(chunks):
                if current_index < start_index:
                    current_index += batch_size
                    continue

                graph = Graph()
                graph.bind("", EMIBOX)
                graph.bind("emi", EMI)
                graph.bind("sosa", SOSA)
                graph.bind("dcterms", DCTERMS)
                graph.bind("wd", WD)
                graph.bind("prov", PROV)
                graph.bind("wgs84", WGS84)
                graph.bind("qudt", QUDT)

                batch_trip_count = 0
                for _, row in batch_data.iterrows():
                    source_taxon_id_mapped = None
                    source_taxon_name_mapped = None
                    target_taxon_id_mapped = None
                    target_taxon_name_mapped = None

                    if row["sourceTaxonId"] in self.wd_map_set_id:
                        source_taxon_id_mapped, source_taxon_name_mapped = self.wd_map_dict_id.get(row["sourceTaxonId"], (None, None))
                        source_taxon_name_mapped = row["sourceTaxonName"] if pd.notna(row["sourceTaxonName"]) else source_taxon_name_mapped
                    elif row["sourceTaxonName"] in self.wd_map_set_name:
                        source_taxon_id_mapped, source_taxon_name_mapped = self.wd_map_dict_name.get(row["sourceTaxonName"], (None, None))
                        source_taxon_name_mapped = row["sourceTaxonName"] if pd.notna(row["sourceTaxonName"]) else source_taxon_name_mapped
                    else:
                        continue

                    if row["targetTaxonId"] in self.wd_map_set_id:
                        target_taxon_id_mapped, target_taxon_name_mapped = self.wd_map_dict_id.get(row["targetTaxonId"], (None, None))
                        target_taxon_name_mapped = row["targetTaxonName"] if pd.notna(row["targetTaxonName"]) else target_taxon_name_mapped
                    elif row["targetTaxonName"] in self.wd_map_set_name:
                        target_taxon_id_mapped, target_taxon_name_mapped = self.wd_map_dict_name.get(row["targetTaxonName"], (None, None))
                        target_taxon_name_mapped = row["targetTaxonName"] if pd.notna(row["targetTaxonName"]) else target_taxon_name_mapped
                    else:
                        continue

                    if not pd.notna(source_taxon_id_mapped) or not pd.notna(target_taxon_id_mapped) or source_taxon_id_mapped == target_taxon_id_mapped:
                        continue
                    else:
                        records_count += 1

                    source_taxon_uri = EMIBOX[f"SAMPLE-{format_uri(source_taxon_id_mapped)}-inRec{i * batch_size + _}"] if is_none_na_or_empty(source_taxon_id_mapped) else None
                    target_taxon_uri = EMIBOX[f"SAMPLE-{format_uri(target_taxon_id_mapped)}-inRec{i * batch_size + _}"] if is_none_na_or_empty(target_taxon_id_mapped) else None
                    intxn_type_uri = EMIBOX[f"{row['interactionTypeName']}"] if is_none_na_or_empty(row['interactionTypeName']) else None
                    intxn_type_id_uri = URIRef(f"{row['interactionTypeId']}") if is_none_na_or_empty(row['interactionTypeId']) else None
                    intxn_rec_uri = EMIBOX[f"inRec{i * batch_size + _}"]

                    graph.add((intxn_rec_uri, RDF.type, EMI.Interaction))
                    batch_trip_count += 1
                    if is_none_na_or_empty(source_taxon_uri):
                        graph.add((intxn_rec_uri, EMI.hasSource, source_taxon_uri))
                        batch_trip_count += 1
                    if is_none_na_or_empty(target_taxon_uri):
                        graph.add((intxn_rec_uri, EMI.hasTarget, target_taxon_uri))
                        batch_trip_count += 1

                    if is_none_na_or_empty(intxn_type_uri) and is_none_na_or_empty(intxn_type_id_uri):
                        graph.add((intxn_rec_uri, EMI.isClassifiedWith, intxn_type_id_uri))
                        batch_trip_count += 1
                        if intxn_type_id_uri not in self.intxn_type_set:
                            graph.add((intxn_type_id_uri, RDF.type, EMI.InteractionType))
                            graph.add((intxn_type_id_uri, RDFS.label, Literal(row['interactionTypeName'], datatype=XSD.string)))
                            self.intxn_type_set.add(intxn_type_id_uri)
                            batch_trip_count += 2
                    if not is_none_na_or_empty(intxn_type_id_uri):
                        graph.add((intxn_rec_uri, EMI.isClassifiedWith, intxn_type_uri))
                        batch_trip_count += 1
                        if intxn_type_uri not in self.intxn_type_set:
                            graph.add((intxn_type_uri, RDF.type, EMI.InteractionType))
                            self.intxn_type_set.add(intxn_type_uri)
                            batch_trip_count += 1

                    if is_none_na_or_empty(row['localityName']):
                        graph.add((intxn_rec_uri, PROV.atLocation, Literal(row['localityName'], datatype=XSD.string)))
                        batch_trip_count += 1
                    if is_none_na_or_empty(row['referenceDoi']):
                        graph.add((intxn_rec_uri, DCTERMS.bibliographicCitation, Literal(row['referenceDoi'], datatype=XSD.string)))
                        batch_trip_count += 1
                    if is_none_na_or_empty(row['sourceDOI']):
                        graph.add((intxn_rec_uri, DCTERMS.bibliographicCitation, Literal(row['sourceDOI'], datatype=XSD.string)))
                        batch_trip_count += 1
                    if is_none_na_or_empty(row['decimalLatitude']):
                        graph.add((intxn_rec_uri, WGS84.lat, Literal(row['decimalLatitude'], datatype=XSD.string)))
                        batch_trip_count += 1
                    if is_none_na_or_empty(row['decimalLongitude']):
                        graph.add((intxn_rec_uri, WGS84.long, Literal(row['decimalLongitude'], datatype=XSD.string)))
                        batch_trip_count += 1

                    if is_none_na_or_empty(source_taxon_name_mapped) and is_none_na_or_empty(source_taxon_uri):
                        source_sample_uri = EMIBOX[f"ORGANISM-{format_uri(source_taxon_name_mapped)}"]
                        graph.add((source_taxon_uri, RDF.type, SOSA.Sample))
                        graph.add((source_taxon_uri, RDFS.label, Literal(source_taxon_name_mapped, datatype=XSD.string)))
                        graph.add((source_taxon_uri, SOSA.isSampleOf, source_sample_uri))
                        batch_trip_count += 3
                    if is_none_na_or_empty(source_taxon_id_mapped) and is_none_na_or_empty(source_taxon_uri):
                        graph.add((source_taxon_uri, EMI.inTaxon, WD[f"{source_taxon_id_mapped}"]))
                        batch_trip_count += 1

                    if is_none_na_or_empty(target_taxon_name_mapped) and is_none_na_or_empty(target_taxon_uri):
                        target_sample_uri = EMIBOX[f"ORGANISM-{format_uri(target_taxon_name_mapped)}"]
                        graph.add((target_taxon_uri, RDF.type, SOSA.Sample))
                        graph.add((target_taxon_uri, RDFS.label, Literal(target_taxon_name_mapped, datatype=XSD.string)))
                        graph.add((target_taxon_uri, SOSA.isSampleOf, target_sample_uri))
                        batch_trip_count += 3
                    if is_none_na_or_empty(target_taxon_id_mapped) and is_none_na_or_empty(target_taxon_uri):
                        graph.add((target_taxon_uri, EMI.inTaxon, WD[f"{target_taxon_id_mapped}"]))
                        batch_trip_count += 1

                    if (is_none_na_or_empty(row['sourceBodyPartName']) or is_none_na_or_empty(row['sourceBodyPartId'])) and is_none_na_or_empty(source_taxon_uri):
                        batch_trip_count = self._add_entity_to_graph(row['sourceBodyPartName'], row['sourceBodyPartId'], source_taxon_uri, EMI.hasAnatomicalEntity, EMI.AnatomicalEntity, "ANATOMICAL_ENTITY", graph, self.body_part_set, batch_trip_count)
                    if (is_none_na_or_empty(row['targetBodyPartName']) or is_none_na_or_empty(row['targetBodyPartId'])) and is_none_na_or_empty(target_taxon_uri):
                        batch_trip_count = self._add_entity_to_graph(row['targetBodyPartName'], row['targetBodyPartId'], target_taxon_uri, EMI.hasAnatomicalEntity, EMI.AnatomicalEntity, "ANATOMICAL_ENTITY", graph, self.body_part_set, batch_trip_count)

                    if (is_none_na_or_empty(row['sourceLifeStageName']) or is_none_na_or_empty(row['sourceLifeStageId'])) and is_none_na_or_empty(source_taxon_uri):
                        batch_trip_count = self._add_entity_to_graph(row['sourceLifeStageName'], row['sourceLifeStageId'], source_taxon_uri, EMI.hasDevelopmentalStage, EMI.DevelopmentalStage, "DEVELOPMENTAL_STAGE", graph, self.life_stage_set, batch_trip_count)
                    if (is_none_na_or_empty(row['targetLifeStageName']) or is_none_na_or_empty(row['targetLifeStageId'])) and is_none_na_or_empty(target_taxon_uri):
                        batch_trip_count = self._add_entity_to_graph(row['targetLifeStageName'], row['targetLifeStageId'], target_taxon_uri, EMI.hasDevelopmentalStage, EMI.DevelopmentalStage, "DEVELOPMENTAL_STAGE", graph, self.life_stage_set, batch_trip_count)

                    if is_none_na_or_empty(row['sourceSexName']) and is_none_na_or_empty(source_taxon_uri):
                        gender_dict = self.entity_matcher.count_biological_sex(row['sourceSexName'])
                        for uri, qty in gender_dict.items():
                            g_data = BNode()
                            graph.add((source_taxon_uri, EMI.hasSex, g_data))
                            graph.add((g_data, QUDT.quantityKind, URIRef(uri)))
                            graph.add((g_data, QUDT.numericValue, Literal(qty, datatype=XSD.integer)))
                            batch_trip_count += 3
                            ent = URIRef(uri)
                            if ent not in self.biological_sex_set:
                                graph.add((ent, RDF.type, EMI.BiologicalSex))
                                self.biological_sex_set.add(ent)
                                batch_trip_count += 1

                    if is_none_na_or_empty(row['targetSexName']) and is_none_na_or_empty(target_taxon_uri):
                        gender_dict = self.entity_matcher.count_biological_sex(row['targetSexName'])
                        for uri, qty in gender_dict.items():
                            g_data = BNode()
                            graph.add((target_taxon_uri, EMI.hasSex, g_data))
                            graph.add((g_data, QUDT.quantityKind, URIRef(uri)))
                            graph.add((g_data, QUDT.numericValue, Literal(qty, datatype=XSD.integer)))
                            batch_trip_count += 3
                            ent = URIRef(uri)
                            if ent not in self.biological_sex_set:
                                graph.add((ent, RDF.type, EMI.BiologicalSex))
                                self.biological_sex_set.add(ent)
                                batch_trip_count += 1

                batch_trip_count = add_inverse_relationships("globi-kg-log.txt",graph, batch_trip_count)
                total_trip_count += batch_trip_count

                with open(checkpoint_file, "w") as f:
                    f.write(str(current_index + batch_size))
                current_index += batch_size

                try:
                    serialized_data = graph.serialize(format="turtle_custom")
                    with gzip.open(self.output_ttl, "at", encoding="utf-8") as out_file:
                        out_file.write(serialized_data)
                except Exception as e:
                    print(f"Error serializing batch {i}: {e}")
                    break
                del graph

        print_to_log(f"{total_trip_count} RDF triples saved to {self.output_ttl}", "globi-kg-log.txt")
        print_to_log(f"{records_count} total records obtained", "globi-kg-log.txt")

