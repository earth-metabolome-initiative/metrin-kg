import argparse
import configparser
import os
from src.data_acquisition.wikidata_fetcher import WikidataDataFetcher
from src.taxonomy_matching.matcher import TaxonomyMatcher
from src.knowledge_graph.globi_kg_generator import GlobiKGGenerator
from src.knowledge_graph.trydb_kg_generator import TrydbKGGenerator 
from src.ontology_matching.match_names_to_ontology import run_ontology_match



def main():
    parser = argparse.ArgumentParser(description="Full Taxonomy Pipeline.")
    parser.add_argument('--config', type=str, default="config.txt", help="Path to config file.")
    parser.add_argument('--run-wd-fetcher', action='store_true', help="Run Wikidata data fetching.")
    parser.add_argument('--run-globi-match', action='store_true', help="Run GloBI taxonomy matching.")
    parser.add_argument('--run-trydb-match', action='store_true', help="Run TRY-db taxonomy matching.")
    parser.add_argument('--run-globi-kg', action='store_true', help="Build GloBI Knowledge Graph.")
    parser.add_argument('--run-trydb-kg', action='store_true', help="Build TRY-db Knowledge Graph.")
    parser.add_argument('--run-ontology-match', action='store_true', help="Run ontology matcher for terms.")
    args = parser.parse_args()


    config = configparser.ConfigParser()
    config.read(args.config)

    # Run Wikidata Data Fetcher 
    wd_files = {}
    if args.run_wd_fetcher:
        print("--- Running Wikidata Data Fetcher ---")
        fetcher_output_dir = config.get('wd files', 'output_file', fallback='data/processed') # Use output_file as base dir
        fetcher = WikidataDataFetcher(output_dir=fetcher_output_dir)
        wd_files = fetcher.run_fetch_pipeline()
        print("Wikidata Data Fetcher complete.")
    else:
        # If not running fetcher, retrieve paths from config.txt
        # Make sure these are updated in config.txt after a fetcher run
        wd_files = {
            "wd_sparql_file": config.get('wd files', 'wd_sparql_file'),
            "wd_lineage_aligned_file": config.get('wd files', 'wd_lineage_file'),
            "wd_repeats_file": config.get('wd files', 'wd_repeats_file')
        }


    # Run Taxonomy Matcher 
    if args.run_globi_match or args.run_trydb_match:
        print("\n--- Loading arguments from config file for Taxonomy Matching ---")
        matcher_output_file_base_globi = config.get('accessory files', 'globi_wd', fallback='data/processed/matched_taxa_globi')
        matcher_output_file_base_trydb = config.get('accessory files', 'trydb_wd', fallback='data/processed/matched_taxa_globi')
        matcher_args = argparse.Namespace(
            verbatim_file=config.get('input tsv files', 'globi_verbatim_file', fallback=None),
            inputFile=config.get('input tsv files', 'tryDb_species_file', fallback=None),
            output_file_globi=matcher_output_file_base_globi,
            output_file_trydb=matcher_output_file_base_trydb,
            wd_sparql_file=config.get('wd files', 'wd_sparql_file', fallback=None),
            wd_lineage_file=config.get('wd files', 'wd_lineage_file', fallback=None),
            wd_repeats_file=config.get('wd files', 'wd_repeats_file', fallback=None),
            # Pass fetched file paths explicitly to matcher. Only reqd when dates attached to wd_files
            #wd_sparql_file=wd_files.get("wd_sparql_file"),
            #wd_lineage_aligned_file=wd_files.get("wd_lineage_aligned_file"),
            #wdLineageRepeats_file=wd_files.get("wd_repeats_file")
        )
    
        matcher = TaxonomyMatcher(config_file=args.config, args=matcher_args)
            #                  wd_sparql_file=wd_files.get("wd_sparql_file"),
            #                  wd_lineage_aligned_file=wd_files.get("wd_lineage_aligned_file"),
            #                  wd_repeats_file=wd_files.get("wd_repeats_file"))

        globi_matched_df = None
        trydb_matched_df = None

    if args.run_globi_match:
        print("\n--- Running GloBI Taxonomy Matching ---")
        globi_matched_df = matcher.run_globi_pipeline()
        print("GloBI Taxonomy Matching complete.")

    if args.run_trydb_match:
        print("\n--- Running TRY-db Taxonomy Matching ---")
        trydb_matched_df = matcher.run_trydb_pipeline()
        print("TRY-db Taxonomy Matching complete.")


    # Example: Run ontology matcher
    if args.run_ontology_match:
        try:
            input_file = config.get('ontology match', 'input_file') #e.g.: data/processed/globi/verbatim_unmappedLifeStageNamesGlobi_mod.csv
            output_file = config.get('ontology match', 'output_file')#e.g.: data/processed/globi/verbatim_mappedLifeStageNamesGlobi.csv
            print(f"--- Running Ontology Matcher ---")
            run_ontology_match(input_file, output_file)
            print(f"Ontology matching complete: {output_file}")
        except Exception as e:
            print(f"Error matching entities to ontologies: {e}")


    # Build Knowledge Graphs 
    if args.run_globi_kg:
        print("\n--- Building GloBI Knowledge Graph ---")
        try:
            globi_kg_generator = GlobiKGGenerator(config_file=args.config)
            globi_kg_generator.generate_rdf()
            print("GloBI Knowledge Graph building complete.")
        except Exception as e:
            print(f"Error building GloBI KG: {e}")

    if args.run_trydb_kg:
        print("\n--- Building TRY-db Knowledge Graph ---")
        try:
            trydb_generator = TrydbKGGenerator(config_file=args.config)
            trydb_generator.generate_rdf(join_column_species="TRY_AccSpeciesName", join_column_wd="wd_taxon_id")
            print("TRY-db Knowledge Graph building complete.")
        except Exception as e:
            print(f"Error building TRY-db KG: {e}")

    print("\n--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()
