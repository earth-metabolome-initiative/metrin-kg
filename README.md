# emi-trydb-globi-kg
Pipeline for generating the knowledge graph integrating emi, trydb, globi datasets.

**Pipeline Components**

1. Wikidata Data Acquisition
Fetches lineage and taxonomic data for up to **15 taxonomies** from Wikidata using SPARQL.

2. Taxonomy Matching against fetched Wikidata records.
Matches taxa from:
- GloBI (Global Biotic Interactions)
- TRY-db (Plant Trait Database)

3. Knowledge Graph Generation
Generates RDF triples representing taxonomic alignments and traits for:
- GloBI
- TRY-db
- EMI-KG (extension of ENPKG)


**Installation**

Install required dependencies:
pip install -r requirements.txt


**Usage**

Run the pipeline via command-line

`python main.py [OPTIONS]`

Command-Line Options

| Option              | Description                                             |
|---------------------|---------------------------------------------------------|
| `--config`          | Path to config file (default: `config.txt`)             |
| `--run-wd-fetcher`  | Fetch taxonomy data from Wikidata                       |
| `--run-globi-match` | Match GloBI dataset with Wikidata taxonomies            |
| `--run-trydb-match` | Match TRY-db dataset with Wikidata taxonomies           |
| `--run-globi-kg`    | Generate RDF Knowledge Graph for GloBI                  |
| `--run-trydb-kg`    | Generate RDF Knowledge Graph for TRY-db                 |


Run the full pipeline:

`python main.py --run-wd-fetcher --run-globi-match --run-trydb-match --run-globi-kg --run-trydb-kg --config config.txt`


Run only Wikidata fetcher:

`python main.py --run-wd-fetcher --config config.txt`

Run only GloBI/TRY-db taxonomy matching:

`python main.py --run-globi-match --config config.txt`

`python main.py --run-trydb-match --config config.txt`

Generate knowledge graph - GloBI/TRY-db:

`python main.py --run-globi-kg --config config.txt`

`python main.py --run-trydb-kg --config config.txt`


Notes

If you skip --run-wd-fetcher, make sure that the wd_* paths in config.txt point to valid, existing files.

Each part of the pipeline can be run independently—helpful for debugging or incremental updates.



**Outputs**

Fetched taxonomy files from Wikidata (*.json)

Matched taxa files for GloBI and TRY-db (*.tsv)

RDF files representing the final knowledge graphs (*.ttl, *.rdf, etc.)


**Contact**

For bugs, questions, or contributions, please open an issue or submit a pull request.

---

