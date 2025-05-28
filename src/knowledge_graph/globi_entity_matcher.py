import pandas as pd
import re
import configparser
from rdflib import URIRef, Literal, Namespace, RDF, RDFS, XSD, DCTERMS, Graph, BNode
from src.common.constants import EMI_BOX_NAMESPACE
from src.common.utils import preprocess_term, format_uri
import os

# Handles entity matching and RDF triple generation for GloBI-specific terms
# like body parts, life stages, and biological sex. Loads entity mappings
# for body parts and life stages.
class GlobiEntityMatcher:

    # Initializes the GlobiEntityMatcher.
    #Args:
    #        config_file (str): Path to the configuration file containing
    #                             file names for body part and life stage mappings.
    #        biological_sex_map_file (str): Path to the TSV file containing
    #                                       mappings for biological sex terms.
    def __init__(self, config_file: str, biological_sex_map_file: str):
        self.graph = Graph()
        self.desig_set = set()
        self.trip_count = 0

        self.eNamesDict = {}
        self.eNamesSet = set()
        self.eURIDict = {}
        self.eURISet = set()
        self.biological_sex_map_dict = {}

        self._load_entity_mappings(config_file)
        self._load_biological_sex_mappings(biological_sex_map_file)

        self.conjunction_patterns1 = re.compile(r'\b(and|y)\b', re.IGNORECASE)
        self.conjunction_patterns2 = re.compile(r'\b(or)\b', re.IGNORECASE)
        self.pre_post_fix = re.compile(r"(adult[as]?|tortere|juvenil[e]?|maybe|\(?torete[s]?\)?)", re.IGNORECASE)
        self.delimiters_regex = re.compile(r"[,;/|&]+", re.IGNORECASE)
        self.delimiters_regex1 = re.compile(r"[\[\]\(\)\?\#:`]+", re.IGNORECASE)
        self.delimiters_regex2 = re.compile(r"[+.,]+", re.IGNORECASE)
        self.delimiters_regex3 = re.compile(r"\s\s", re.IGNORECASE)
        self.number_word_pattern = re.compile(r"(\d+)\s*([\w-]+)|([\w-]+)\s*(\d+)")

        self.fungi_terms = {
            "dematiaceous anamorph": "dematiaceous anamorph",
            "coelomycetous anamorph": "coelomycetous anamorph",
            "anamorph": "anamorph",
            "anamoprh": "anamorph",
            "synnemata": "synnemata",
            "synnema": "synnemata",
            "teleomorph": "teleomorph"
        }

    # Loads entity mappings for body parts and life stages from a config file.
    def _load_entity_mappings(self, config_file: str):
        if os.path.exists(config_file):
            config = configparser.ConfigParser()
            config.read(config_file)
            bp_file_name = config.get('knowledge graph files', 'bp_fileName', fallback=None)
            ls_file_name = config.get('knowledge graph files', 'ls_fileName', fallback=None)

            if bp_file_name and ls_file_name:
                file_names = [bp_file_name, ls_file_name]
                key_col = "InputTerm"
                val_col1 = "BestMatch"
                val_col2 = "URI"
                try:
                    map_df = pd.concat([pd.read_csv(file, sep=",", quoting=0, dtype=str) for file in file_names], ignore_index=True)
                    e_names = map_df.dropna(subset=[val_col1]).query(f"{val_col1}.str.strip() != ''", engine='python')
                    self.eNamesDict = e_names.set_index(key_col)[val_col1].to_dict()
                    self.eNamesSet = set(e_names[key_col])
                    e_uri = map_df.dropna(subset=[val_col2]).query(f"{val_col2}.str.strip() != ''", engine='python')
                    self.eURIDict = e_uri.set_index(key_col)[val_col2].to_dict()
                    self.eURISet = set(e_uri[key_col])
                    print("Body part and life stage entity mappings loaded.")
                except FileNotFoundError:
                    print("Error: One or both body part/life stage mapping files not found.")
                except Exception as e:
                    print(f"Error loading body part/life stage mappings: {e}")
            else:
                print("Warning: Body part or life stage file names not specified in config.")
        else:
            print("Error: config.txt not found.")

    # Loads biological sex mappings from a TSV file.
    def _load_biological_sex_mappings(self, biological_sex_map_file: str):
        if biological_sex_map_file:
            try:
                map_df = pd.read_csv(biological_sex_map_file, sep="\t", quoting=3, dtype=str)
                self.biological_sex_map_dict = dict(zip(map_df['input'].str.lower(), map_df['output']))
                print("Biological sex mappings loaded.")
            except FileNotFoundError:
                print(f"Error: Biological sex mapping file not found at {biological_sex_map_file}")
            except Exception as e:
                print(f"Error loading biological sex mappings: {e}")
        else:
            print("Warning: Biological sex file name not specified.")

    # Adds an entity (with its type and label) to the graph if not already added.
    def _add_entity_to_graph(self, subject: URIRef, predicate: URIRef, rdftype: URIRef, entity_uri: URIRef, entity_name: str) -> None:
        self.graph.add((subject, predicate, entity_uri))
        self.trip_count += 1
        if entity_uri not in self.desig_set:
            self.graph.add((entity_uri, RDF.type, rdftype))
            self.graph.add((entity_uri, RDFS.label, Literal(entity_name, datatype=XSD.string)))
            self.trip_count += 2
            self.desig_set.add(entity_uri)

    # Looks up a term in the entity mappings and adds it to the graph.
    def _lookup_and_add_term(self, original_term: str, graph_subject: URIRef, graph_predicate: URIRef, graph_rdftype: URIRef, term_to_match: str) -> None:
        term_to_match = preprocess_term(term_to_match)

        if term_to_match in self.eURISet:
            mod_entity_uri = URIRef(self.eURIDict[term_to_match])
            mod_entity_name = self.eNamesDict[term_to_match]
            self._add_entity_to_graph(graph_subject, graph_predicate, graph_rdftype, mod_entity_uri, mod_entity_name)
        elif term_to_match in self.eNamesSet:
            mod_entity_name = self.eNamesDict[term_to_match]
            ent = EMI_BOX_NAMESPACE[format_uri(mod_entity_name)]
            self._add_entity_to_graph(graph_subject, graph_predicate, graph_rdftype, ent, mod_entity_name)
        else:
            cleaned_term = preprocess_term(self.pre_post_fix.sub('', term_to_match))
            if cleaned_term in self.eURISet:
                mod_entity_uri = URIRef(self.eURIDict[cleaned_term])
                mod_entity_name = self.eNamesDict[cleaned_term]
                self._add_entity_to_graph(graph_subject, graph_predicate, graph_rdftype, mod_entity_uri, mod_entity_name)
            elif cleaned_term in self.eNamesSet:
                mod_entity_name = self.eNamesDict[cleaned_term]
                ent = EMI_BOX_NAMESPACE[format_uri(mod_entity_name)]
                self._add_entity_to_graph(graph_subject, graph_predicate, graph_rdftype, ent, mod_entity_name)

    def _parse_and_process_terms(self, term_string: str, graph_subject: URIRef, graph_predicate: URIRef, graph_rdftype: URIRef, process_mode: str = 'add_to_graph') -> dict:
        """
        Parses a complex term string, cleans it, and either adds entities to graph
        or counts term occurrences based on process_mode.

        Args:
            term_string (str): The string containing one or more terms to process.
            graph_subject (URIRef): The subject URI for the RDF triple.
            graph_predicate (URIRef): The predicate URI for the RDF triple.
            graph_rdftype (URIRef): The RDF type URI for the entities.
            process_mode (str, optional): 'add_to_graph' to add entities to the graph,
                                         'count_only' to count biological sex terms.
                                         Defaults to 'add_to_graph'.

        Returns:
            dict: A dictionary of biological sex term counts if process_mode is 'count_only',
                  otherwise an empty dictionary.
        """
        term_string_original = term_string
        term_string = str(term_string).lower().strip()

        term_string = self.conjunction_patterns1.sub(',', term_string)
        term_string = self.conjunction_patterns2.sub('', term_string)
        term_string = self.delimiters_regex.sub(',', term_string)
        term_string = self.delimiters_regex1.sub(' ', term_string)
        term_string = self.delimiters_regex3.sub(' ', term_string)

        sub_terms = self.delimiters_regex2.split(term_string)

        result_counts = {value: 0 for value in self.biological_sex_map_dict.values()} if process_mode == 'count_only' else {}
        records = []

        for term in sub_terms:
            cleaned_term_for_match = self.delimiters_regex2.sub(' ', term)
            matches = self.number_word_pattern.findall(cleaned_term_for_match)

            if matches:
                for match in matches:
                    number_str = match[0] if match[0] else match[3]
                    word_term = match[1] if match[1] else match[2]
                    count = int(number_str) if number_str else 1

                    processed_word_term = preprocess_term(word_term.strip())

                    if process_mode == 'add_to_graph':
                        self._lookup_and_add_term(term_string_original, graph_subject, graph_predicate, graph_rdftype, processed_word_term)
                    elif process_mode == 'count_only':
                        mapped_id = self.biological_sex_map_dict.get(processed_word_term)
                        if mapped_id:
                            result_counts[mapped_id] += count
                            records.append({"Term": processed_word_term, "TermID": mapped_id, "Count": count})
                        else:
                            cleaned_word_term_for_count = preprocess_term(self.pre_post_fix.sub('', processed_word_term))
                            mapped_id_cleaned = self.biological_sex_map_dict.get(cleaned_word_term_for_count)
                            if mapped_id_cleaned:
                                result_counts[mapped_id_cleaned] += count
                                records.append({"Term": cleaned_word_term_for_count, "TermID": mapped_id_cleaned, "Count": count})
                            else:
                                if "unknown" in self.biological_sex_map_dict:
                                    result_counts[self.biological_sex_map_dict["unknown"]] += count
                                else:
                                    self.biological_sex_map_dict["unknown"] = "unknown_id"
                                    result_counts[self.biological_sex_map_dict["unknown"]] += count
                                records.append({"Term": processed_word_term, "TermID": "unknown_id", "Count": count})
            else:
                processed_term = preprocess_term(term.strip())
                if process_mode == 'add_to_graph':
                    self._lookup_and_add_term(term_string_original, graph_subject, graph_predicate, graph_rdftype, processed_term)
                elif process_mode == 'count_only':
                    mapped_id = self.biological_sex_map_dict.get(processed_term)
                    if mapped_id:
                        result_counts[mapped_id] += 1
                        records.append({"Term": processed_term, "TermID": mapped_id, "Count": 1})
                    else:
                        cleaned_term_for_count = preprocess_term(self.pre_post_fix.sub('', processed_term))
                        mapped_id_cleaned = self.biological_sex_map_dict.get(cleaned_term_for_count)
                        if mapped_id_cleaned:
                            result_counts[mapped_id_cleaned] += 1
                            records.append({"Term": cleaned_term_for_count, "TermID": mapped_id_cleaned, "Count": 1})
                        else:
                            if "unknown" in self.biological_sex_map_dict:
                                result_counts[self.biological_sex_map_dict["unknown"]] += 1
                            else:
                                self.biological_sex_map_dict["unknown"] = "unknown_id"
                                result_counts[self.biological_sex_map_dict["unknown"]] += 1
                            records.append({"Term": processed_term, "TermID": "unknown_id", "Count": 1})

        if process_mode == 'count_only':
            filtered_dict = {k: v for k, v in result_counts.items() if v != 0}
            return filtered_dict
        else:
            return {}

    # Adds body part entities to the graph.
    def add_body_part_entities(self, subject: URIRef, body_part_string: str) -> None:
        PREDICATE_BODY_PART = URIRef(EMI_BOX_NAMESPACE["hasBodyPart"])
        RDFTYPE_BODY_PART = URIRef(EMI_BOX_NAMESPACE["BodyPart"])
        self._parse_and_process_terms(body_part_string, subject, PREDICATE_BODY_PART, RDFTYPE_BODY_PART, 'add_to_graph')

    # Adds life stage entities to the graph.
    def add_life_stage_entities(self, subject: URIRef, life_stage_string: str) -> None:
        PREDICATE_LIFE_STAGE = URIRef(EMI_BOX_NAMESPACE["hasLifeStage"])
        RDFTYPE_LIFE_STAGE = URIRef(EMI_BOX_NAMESPACE["LifeStage"])
        self._parse_and_process_terms(life_stage_string, subject, PREDICATE_LIFE_STAGE, RDFTYPE_LIFE_STAGE, 'add_to_graph')

    # Counts occurrences of biological sex terms.
    def count_biological_sex(self, sex_string: str) -> dict:
        return self._parse_and_process_terms(sex_string, None, None, None, 'count_only')

    # Returns the RDF graph.
    def get_graph(self) -> Graph:
        return self.graph

    # Returns the current triple count.
    def get_triple_count(self) -> int:
        return self.trip_count
