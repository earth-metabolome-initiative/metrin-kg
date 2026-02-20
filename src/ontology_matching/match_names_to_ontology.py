from owlready2 import get_ontology
from sentence_transformers import SentenceTransformer, util
import csv

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_ontologies(ontology_paths):
    ontologies = {}
    for ontology_name, path in ontology_paths.items():
        ontologies[ontology_name] = get_ontology(path).load()
    return ontologies

def extract_terms_from_ontology(ontology):
    ontology_terms = []
    for cls in ontology.classes():
        if cls.label:
            main_label = cls.label[0]
            synonyms = []
            if hasattr(cls, "hasExactSynonym"):
                synonyms.extend(cls.hasExactSynonym)
            if hasattr(cls, "hasBroadSynonym"):
                synonyms.extend(cls.hasBroadSynonym)
            if hasattr(cls, "hasRelatedSynonym"):
                synonyms.extend(cls.hasRelatedSynonym)
            all_labels = [main_label] + synonyms
            for label in all_labels:
                ontology_terms.append((label, cls.iri, main_label))
    return ontology_terms

def generate_ontology_embeddings(ontologies, model):
    all_ontology_terms = []
    for ontology in ontologies.values():
        all_ontology_terms.extend(extract_terms_from_ontology(ontology))
    ontology_labels = [term[0] for term in all_ontology_terms]
    ontology_embeddings = model.encode(ontology_labels, convert_to_tensor=True)
    return all_ontology_terms, ontology_embeddings

def find_best_match(input_term, ontology_terms, ontology_embeddings, model):
    input_embedding = model.encode(input_term, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(input_embedding, ontology_embeddings)
    best_idx = similarities.argmax().item()
    best_label, uri, main_label = ontology_terms[best_idx]
    score = similarities[0][best_idx].item()
    return best_label, main_label, uri, score

def run_ontology_match(input_file, output_file): # e.g. suite
    ontology_paths = {
        "UBERON": "http://purl.obolibrary.org/obo/uberon.owl",
        "PO": "http://purl.obolibrary.org/obo/po.owl",
        "ENVO": "http://purl.obolibrary.org/obo/envo.owl",
        "PATO": "http://purl.obolibrary.org/obo/pato.owl",
        "FAO": "http://purl.obolibrary.org/obo/fao.owl",
        "HAO": "http://purl.obolibrary.org/obo/hao.owl",
        "BTO": "http://purl.obolibrary.org/obo/bto.owl",
        "NCIT": "http://purl.obolibrary.org/obo/ncit.owl",
    }

    ontologies = load_ontologies(ontology_paths)
    ontology_terms, embeddings = generate_ontology_embeddings(ontologies, model)

    with open(input_file, "r") as infile:
        input_terms = [line.strip() for line in infile if line.strip()]

    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Input Term", "Matched Term", "Primary Label", "URI", "Similarity Score"])

        for term_line in input_terms:
            terms = [t.strip() for t in term_line.replace('/', ',').split(',')]
            for term in terms:
                best_label, main_label, uri, score = find_best_match(term, ontology_terms, embeddings, model)
                writer.writerow([term, best_label, main_label, uri, f"{score:.4f}"])
                print(f"Processed: {term} -> {best_label} ({score:.4f})")
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFile', type=str, help="Enter the file name with terms to be matched")
    parser.add_argument('outputFile', type=str, help="Enter the output file name")
    args = parser.parse_args()
    run_ontology_match(args.inputFile, args.outputFile)
