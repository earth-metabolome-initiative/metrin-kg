# METRIN-KG

Pipeline for generating the knowledge graph integrating [enriched metabolite data originally used for ENPKG](https://zenodo.org/records/10827917), traits data from [TRY](https://www.try-db.org/TryWeb/Home.php), and interaction data from [GloBI](https://www.globalbioticinteractions.org/).

> **Notes**
>
> 1. If you want to build the METRIN-KG triples, skip to [installation](https://github.com/earth-metabolome-initiative/metrin-kg#installation)
> 2. If you just want build your own instance METRIN-KG SPARQL endpoint, skip to [querying METRIN-KG](https://github.com/earth-metabolome-initiative/metrin-kg?tab=readme-ov-file#querying-metrin-kg)

## Pipeline Components

1. Wikidata Data Acquisition
   Fetches lineage and taxonomic data for up to 15 taxonomies from Wikidata using SPARQL.

2. Taxonomy Matching against fetched Wikidata records.
   Matches taxa from:

- GloBI (Global Biotic Interactions)
- TRY (Plant Trait Database)

3. Knowledge Graph Generation
   Generates RDF triples representing taxonomic alignments and traits for:

- GloBI
- TRY
- EMI-KG (extension of ENPKG)

## Installation

1. Clone the repository

```bash
git clone https://github.com/earth-metabolome-initiative/metrin-kg.git
```

2. Make sure you have pipenv installed. If not, install it via:

```bash
pip install pipenv
```

The code has been run only with `python-3.12`, but it may work with other versions of `python-3`.

3. Once `pipenv` is installed, install the dependencies:

```bash
pipenv install
pipenv shell
```

## Usage

1. Download associated accessory data from [METRIN-KG zenodo repository](https://doi.org/10.5281/zenodo.15689187) and [verbatim-interactions.tsv.gz](https://zenodo.org/records/14640564/files/verbatim-interactions.tsv.gz?download=1) (only) from [GloBI zenodo repository](https://zenodo.org/records/14640564).

```bash
cd metrin-kg

# download METRIN-KG data
wget https://zenodo.org/records/15689186/files/metrin-kg.tar.gz?download=1
tar -xvf metrin-kg.tar.gz
mv metrin-kg-data data

# download GloBI data
wget https://zenodo.org/records/14640564/files/verbatim-interactions.tsv.gz?download=1
mv verbatim-interactions.tsv.gz data/raw/

# download TRY data
wget https://zenodo.org/records/17079465/files/TRYdb_40340.txt.gz?download=1
mv TRYdb_40340.txt.gz data/raw/
```

2. For supported arguments, run:

```bash
python main.py --help
```

3. Run the pipeline via command-line

```bash
python main.py [OPTIONS]
```

Command-Line Options

| Option                 | Description                                   |
| ---------------------- | --------------------------------------------- |
| `--config`             | Path to config file (default: `config.txt`)   |
| `--run-wd-fetcher`     | Fetch taxonomy data from Wikidata             |
| `--run-ontology-match` | Match ontologies to GloBI or TRY terms     |
| `--run-globi-match`    | Match GloBI dataset with Wikidata taxonomies  |
| `--run-trydb-match`    | Match TRY dataset with Wikidata taxonomies |
| `--run-globi-kg`       | Generate RDF Knowledge Graph for GloBI        |
| `--run-trydb-kg`       | Generate RDF Knowledge Graph for TRY       |

4. Run the full pipeline:

```bash
python main.py --run-wd-fetcher --run-globi-match --run-trydb-match --run-globi-kg --run-trydb-kg --config config.txt
```

Note: This might take a while. If you only want to reproduce the KG, skip to point-8 directly. Note that if you have copied the data from the METRIN-KG zenodo repository, all accessory files are already available.

5. Run only Wikidata fetcher:

```bash
python main.py --run-wd-fetcher --config config.txt
```

Note: If you just want to reproduce the KG, you don't need to perform this step because the data directory already has the relevant files (if the METRIN-KG zenodo contents are copied correctly).

6. Run only GloBI/TRY taxonomy matching:

```bash
python main.py --run-globi-match --config config.txt
```

```bash
python main.py --run-trydb-match --config config.txt
```

Note: If you just want to reproduce the KG, you don't need to perform this step because the data directory already has the relevant files (if the METRIN-KG zenodo contents are copied correctly).

7. Run only ontology matching

This can be done for any of the datasets from GloBI (body part, life stages, and biological sex) and TRY (unit names). Specify the input and output files under `[ontology]` header in `config.txt`

```bash
python main.py --run-ontology-match --config config.txt
```

Note: If you just want to reproduce the KG, you don't need to perform this step because the data directory already has the relevant files (if the METRIN-KG zenodo contents are copied correctly).

8. Generate knowledge graph - GloBI/TRY:

```bash
python main.py --run-globi-kg --config config.txt
```

```bash
python main.py --run-trydb-kg --config config.txt
```

> _Notes_:
>
> 1. For generating the sub knowledge graph of metabolites, [follow the instructions here](https://github.com/earth-metabolome-initiative/earth_metabolome_ontology?tab=readme-ov-file#generating-rdf-triples-based-on-the-emi-ontology-for-the-pf1600-dataset)
> 2. If you skip `--run-wd-fetcher`, make sure that the wd\_\* paths in config.txt point to valid, existing files. Each part of the pipeline can be run independently.
> 3. Outputs
>    a) Fetched taxonomy files from Wikidata (\*.json)
>    b) Matched taxa files for GloBI and TRY (\*.tsv)
>    c) RDF files representing the final knowledge graphs (\*.ttl, \*.rdf, etc.)

## Querying METRIN-KG

For querying METRIN-KG, you can use two methods:

### a) the Qlever powered end-point hosted on [earth-metabolome-initiative.org](https://kg.earthmetabolome.org/metrin/).

_Want to generate your own instance of METRIN-KG SPARQL endpoint?_

Follow the instructions on [qlever-control](https://github.com/qlever-dev/qlever-control) and our [fork of qlever-ui](https://github.com/earth-metabolome-initiative/qlever-ui) to install Qlever.
You can find the [qlever config file](https://github.com/earth-metabolome-initiative/metrin-kg/blob/main/Qlever.metrin_kg) used to index METRIN-KG.
Follow the commands below to generate your own instance of METRIN-KG on localhost.

```bash
qlever --qleverfile Qlever.metrin_kg get-data  # download full METRIN-KG graph
qlever --qleverfile Qlever.metrin_kg index --overwrite-existing --parallel-parsing false  # index KG
qlever --qleverfile Qlever.metrin_kg start  # start the server on local host
```

Once Qlever index is generated and the server started, you can query the endpoint using qlever-ui on your localhost. Once you are done querying METRIN-KG, don't forget to stop the server

```bash
qlever --qleverfile Qlever.metrin_kg stop
```

> _Notes_:
> 1. Note that you will need Docker for running `qlever`. On Linux Docker runs natively and takes up only a small amount of RAM, whereas, on macOS, Docker runs in a virtual machine and thus, takes significant RAM. Therefore, on macOS, `qlever index` may fail sometimes, thus requiring more moemory./home/drishti/.local/bin
> 2. For indexing the METRIN-KG data (`qlever index`), atleast 31 GB RAM will be required - works on Linux, may require more on macOS.
> 3. The shell commands for `qlever get-data` inside the config file have been adapted for Ubuntu's terminal and macOS's iTerm2 default settings. 
> 4. `qlever get-data` command will only download the triple (`ttl.gz` or `ttl`) and not the raw data used to generate the triples. For downloading the full METRIN-KG dataset including the raw data and the triples, please refer to [Usage](https://github.com/earth-metabolome-initiative/metrin-kg?tab=readme-ov-file#usage) point-1.


### b) the [sparql-editor powered endpoint](https://sib-swiss.github.io/sparql-editor/metrin-kg)

This endpoint also provides direct access to class-overview (find the icon at the top-left corner). It also provides a way to suggest example queries to be accepted in the METRIN-KG examples set (find the icon 💾 at the top-left corner).

Note that for some queries, this endpoint might give a `The quota has exceeded` error. We are trying to resolve it. Updates soon...

## Class-overview

For visualization of class overview and data schema, visit the [sparql-editor powered endpoint](https://sib-swiss.github.io/sparql-editor/metrin-kg) and click on the class overview icon at the top-left corner of the page.

You can also open [sparql_editor_metrin-kg.html](https://github.com/earth-metabolome-initiative/metrin-kg/blob/main/sparql_editor_index_metrin-kg.html) in a browser and visualize the class-overview. For instructions on how to generate this file, refer to following github repos: [sparql-editor](https://github.com/sib-swiss/sparql-editor), [sparql-examples](https://github.com/sib-swiss/sparql-examples), and our [fork of void-generator](https://github.com/mdrishti/void-generator-c).

## Contribute and Contact

Have a look at [METRIN-KG wiki](https://github.com/earth-metabolome-initiative/metrin-kg/wiki) for how-to-use and how-to-contribute-to METRIN-KG.

For bugs, questions, or contributions, please open an issue or submit a pull request.

---

## Citation

If you use METRIN-KG in your work, please cite

METRIN-KG: A knowledge graph integrating plant metabolites, traits and biotic interactions
Disha Tandon, Tarcisio Mendes De Farias, Pierre-Marie Allard, Emmanuel Defossez
bioRxiv 2025.08.20.671289; doi: https://doi.org/10.1101/2025.08.20.671289
