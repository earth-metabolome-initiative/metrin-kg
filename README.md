# METRIN-KG
Pipeline for generating the knowledge graph integrating [enriched metabolite data originally used for ENPKG](https://zenodo.org/records/10827917), traits data from [TRY-db](https://www.try-db.org/TryWeb/Home.php), and interaction data from [GloBI](https://www.globalbioticinteractions.org/).

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

1. Clone the repository

`git clone https://github.com/earth-metabolome-initiative/metrin-kg.git`


2. Make sure you have pipenv installed. If not, install it via:

`pip install pipenv`


3. Once `pipenv` is installed, install the dependencies:

`pipenv install`
`pipenv shell`




**Usage**


1. Run the pipeline via command-line

`python main.py [OPTIONS]`

Command-Line Options

| Option              | Description                                             |
|---------------------|---------------------------------------------------------|
| `--config`          | Path to config file (default: `config.txt`)             |
| `--run-wd-fetcher`  | Fetch taxonomy data from Wikidata                       |
| `--run-ontology-match` | Match ontologies to GloBI or TRY-db terms            |
| `--run-globi-match` | Match GloBI dataset with Wikidata taxonomies            |
| `--run-trydb-match` | Match TRY-db dataset with Wikidata taxonomies           |
| `--run-globi-kg`    | Generate RDF Knowledge Graph for GloBI                  |
| `--run-trydb-kg`    | Generate RDF Knowledge Graph for TRY-db                 |


3. Run the full pipeline:

`python main.py --run-wd-fetcher --run-globi-match --run-trydb-match --run-globi-kg --run-trydb-kg --config config.txt`


4. Run only Wikidata fetcher:

`python main.py --run-wd-fetcher --config config.txt`


5. Run only GloBI/TRY-db taxonomy matching:

`python main.py --run-globi-match --config config.txt`

`python main.py --run-trydb-match --config config.txt`



6. Run only ontology matching

This can be done for any of the datasets from GloBI (body part, life stages, and biological sex) and TRY-db (unit names). Specify the input and output files under `[ontology]` header in `config.txt`

`python main.py --run-ontology-match --config config.txt`




7. Generate knowledge graph - GloBI/TRY-db:

`python main.py --run-globi-kg --config config.txt`

`python main.py --run-trydb-kg --config config.txt`


_Notes_

If you skip --run-wd-fetcher, make sure that the wd_* paths in config.txt point to valid, existing files.

Each part of the pipeline can be run independently—helpful for debugging or incremental updates.





**Outputs**

Fetched taxonomy files from Wikidata (*.json)

Matched taxa files for GloBI and TRY-db (*.tsv)

RDF files representing the final knowledge graphs (*.ttl, *.rdf, etc.)





**Contribute and Contact**


Have a look at [METRIN-KG wiki](https://github.com/earth-metabolome-initiative/metrin-kg/wiki) for how-to-use and how-to-contribute-to METRIN-KG.

For bugs, questions, or contributions, please open an issue or submit a pull request.

---

