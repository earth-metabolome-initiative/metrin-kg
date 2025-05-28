from rdflib import URIRef, Literal, Namespace, RDF, RDFS, XSD, DCTERMS, Graph

# Predefined ranks for taxonomy matching (from taxonomy_matcher)
predefined_ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

# Prefixes for external taxonomic databases (from taxonomy_matcher)
prefixes = {
    1: "EOL:", 2: "GBIF:", 3: "NCBI:", 4: "OTT:", 5: "ITIS:",
    6: "IRMNG:", 7: "COL:", 8: "NBN:", 9: "WORMS:", 10: "BOLD:",
    11: "PLAZI:", 12: "APNI:", 13: "msw3:", 14: "INAT_TAXON:", 15: "EPPO:"
}

# Wikidata Rank URIs and Names (from wikidata_fetcher)
WIKIDATA_RANK_URIS = [
    "http://www.wikidata.org/entity/Q36732",  # kingdom
    "http://www.wikidata.org/entity/Q38348",  # phylum
    "http://www.wikidata.org/entity/Q37517",  # class
    "http://www.wikidata.org/entity/Q36602",  # order
    "http://www.wikidata.org/entity/Q35409",  # family
    "http://www.wikidata.org/entity/Q34740",  # genus
    "http://www.wikidata.org/entity/Q7432"   # species
]
RANK_URI_TO_NAME_MAP = {
    "http://www.wikidata.org/entity/Q36732": "kingdom",
    "http://www.wikidata.org/entity/Q38348": "phylum",
    "http://www.wikidata.org/entity/Q37517": "class",
    "http://www.wikidata.org/entity/Q36602": "order",
    "http://www.wikidata.org/entity/Q35409": "family",
    "http://www.wikidata.org/entity/Q34740": "genus",
    "http://www.wikidata.org/entity/Q7432": "species"
}

"""# RDFlib Namespaces (from entity matching part)
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")
DCTERMS = Namespace("http://purl.org/dc/terms/")
"""

# Custom namespace for EMI Box (for GloBI entity matching part)
EMI_BOX_NAMESPACE = Namespace("https://purl.org/emi/abox#")

SOSA = Namespace("http://www.w3.org/ns/sosa/") # From your inverse relations
# Inverse relationships for RDF graph (for GloBI entity matching part)
INVERSE_RELATIONS = {
    str(DCTERMS.isPartOf): str(DCTERMS.hasPart),
    str(DCTERMS.hasFormat): str(DCTERMS.isFormatOf),
    str(DCTERMS.hasVersion): str(DCTERMS.isVersionOf),
    str(DCTERMS.references): str(DCTERMS.isReferencedBy),
    str(DCTERMS.replaces): str(DCTERMS.isReplacedBy),
    str(DCTERMS.requires): str(DCTERMS.isRequiredBy),
    str(SOSA.isActedOnBy): str(SOSA.actsOnProperty),
    str(SOSA.isFeatureOfInterestOf): str(SOSA.hasFeatureOfInterest),
    str(SOSA.isResultOf): str(SOSA.hasResult),
    str(SOSA.isSampleOf): str(SOSA.hasSample),
    str(SOSA.isHostedBy): str(SOSA.hosts),
    str(SOSA.actsOnProperty): str(SOSA.isActedOnBy),
    str(SOSA.hasFeatureOfInterest): str(SOSA.isFeatureOfInterestOf),
    str(SOSA.hosts): str(SOSA.isHostedBy),
    str(SOSA.observes): str(SOSA.isObservedBy),
    str(SOSA.hasResult): str(SOSA.isResultOf),
    str(SOSA.hasSample): str(SOSA.isSampleOf),
    str(SOSA.madeByActuator): str(SOSA.madeActuation),
    str(SOSA.madeActuation): str(SOSA.madeByActuator),
    str(SOSA.madeSampling): str(SOSA.madeBySampler),
    str(SOSA.madeObservation): str(SOSA.madeBySensor),
    str(SOSA.madeBySensor): str(SOSA.madeObservation),
    str(SOSA.madeBySampler): str(SOSA.madeSampling),
    str(SOSA.isObservedBy): str(SOSA.observes),
}
