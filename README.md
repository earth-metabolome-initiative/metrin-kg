# METRIN-KG
Pipeline for generating the knowledge graph integrating [enriched metabolite data originally used for ENPKG](https://zenodo.org/records/10827917), traits data from [TRY-db](https://www.try-db.org/TryWeb/Home.php), and interaction data from [GloBI](https://www.globalbioticinteractions.org/).

## Pipeline Components

1. Wikidata Data Acquisition
Fetches lineage and taxonomic data for up to 15 taxonomies from Wikidata using SPARQL.

2. Taxonomy Matching against fetched Wikidata records.
Matches taxa from:
- GloBI (Global Biotic Interactions)
- TRY-db (Plant Trait Database)

3. Knowledge Graph Generation
Generates RDF triples representing taxonomic alignments and traits for:
- GloBI
- TRY-db
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


3. Once `pipenv` is installed, install the dependencies:


```bash
pipenv install
pipenv shell
```




## Usage


1. Download associated accessory data from [METRIN-KG zenodo repository](https://doi.org/10.5281/zenodo.15689187) and [verbatim-interactions.tsv.gz](https://zenodo.org/records/14640564/files/verbatim-interactions.tsv.gz?download=1) (only) from [GloBI zenodo repository](https://zenodo.org/records/14640564). 

```bash
cd metrin-kg
wget https://zenodo.org/records/15689187/files/metrin-kg.tar.gz?download=1
wget https://zenodo.org/records/14640564/files/verbatim-interactions.tsv.gz?download=1
tar -xvf metrin-kg.tar.gz
mv metrin-kg-data data
mv verbatim-interactions.tsv.gz data/raw/
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

| Option              | Description                                             |
|---------------------|---------------------------------------------------------|
| `--config`          | Path to config file (default: `config.txt`)             |
| `--run-wd-fetcher`  | Fetch taxonomy data from Wikidata                       |
| `--run-ontology-match` | Match ontologies to GloBI or TRY-db terms            |
| `--run-globi-match` | Match GloBI dataset with Wikidata taxonomies            |
| `--run-trydb-match` | Match TRY-db dataset with Wikidata taxonomies           |
| `--run-globi-kg`    | Generate RDF Knowledge Graph for GloBI                  |
| `--run-trydb-kg`    | Generate RDF Knowledge Graph for TRY-db                 |


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

6. Run only GloBI/TRY-db taxonomy matching:

```bash
python main.py --run-globi-match --config config.txt
```

```bash
python main.py --run-trydb-match --config config.txt
```

Note: If you just want to reproduce the KG, you don't need to perform this step because the data directory already has the relevant files (if the METRIN-KG zenodo contents are copied correctly).


7. Run only ontology matching

This can be done for any of the datasets from GloBI (body part, life stages, and biological sex) and TRY-db (unit names). Specify the input and output files under `[ontology]` header in `config.txt`

```bash
python main.py --run-ontology-match --config config.txt
```

Note: If you just want to reproduce the KG, you don't need to perform this step because the data directory already has the relevant files (if the METRIN-KG zenodo contents are copied correctly).


8. Generate knowledge graph - GloBI/TRY-db:

```bash
python main.py --run-globi-kg --config config.txt
```

```bash
python main.py --run-trydb-kg --config config.txt
```

> _Notes_: 
> 1. For generating the sub knowledge graph of metabolites, [follow the instructions here](https://github.com/earth-metabolome-initiative/earth_metabolome_ontology?tab=readme-ov-file#generating-rdf-triples-based-on-the-emi-ontology-for-the-pf1600-dataset)
> 2. If you skip `--run-wd-fetcher`, make sure that the wd_* paths in config.txt point to valid, existing files. Each part of the pipeline can be run independently.
> 3. Outputs
> a) Fetched taxonomy files from Wikidata (\*.json)
> b) Matched taxa files for GloBI and TRY-db (\*.tsv)
> c) RDF files representing the final knowledge graphs (\*.ttl, \*.rdf, etc.)



## Querying METRIN-KG

For querying METRIN-KG, you can use the Qlever powered end-point hosted on [earth-metabolome-initiative.org](https://kg.earthmetabolome.org/metrin-kg/).

_Want to generate your own instance of METRIN-KG SPARQL endpoint?_

Follow the instructions on [qlever-control](https://github.com/ad-freiburg/qlever-control) and [qlever-ui](https://github.com/earth-metabolome-initiative/qlever-ui) to install Qlever.
You can find the [qlever config file](https://github.com/earth-metabolome-initiative/metrin-kg/blob/main/Qlever.metrin_kg) used to index METRIN-KG. 
Follow the commands below to generate your own instance of METRIN-KG on localhost.

```bash
qleverX --qleverfile Qlever.metrin_kg get-data  # download full METRIN-KG graph
qleverX --qleverfile Qlever.metrin_kg index --overwrite-existing --parallel-parsing false  # index KG
qleverX --qleverfile Qlever.metrin_kg start  # start the server on local host
```
 

Once Qlever index is generated and the server started, you can query the endpoint using qlever-ui on your localhost. Once you are done querying METRIN-KG, don't forget to stop the server
```bash
qleverX --qleverfile Qlever.metrin_kg stop
```




## Contribute and Contact


Have a look at [METRIN-KG wiki](https://github.com/earth-metabolome-initiative/metrin-kg/wiki) for how-to-use and how-to-contribute-to METRIN-KG.

For bugs, questions, or contributions, please open an issue or submit a pull request.

---

