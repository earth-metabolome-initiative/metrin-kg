#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import re
import sys
from pathlib import Path
from rdflib import Graph

def load_prefixes(ttl_file):
    """Load prefix mappings from TTL file"""
    g = Graph()
    g.parse(ttl_file, format='turtle')
    
    prefixes = {}
    
    # Query for sh:prefix and sh:namespace pairs
    query = """
    PREFIX sh: <http://www.w3.org/ns/shacl#>
    SELECT ?prefix ?namespace
    WHERE {
        ?decl sh:prefix ?prefix ;
              sh:namespace ?namespace .
    }
    """
    
    for row in g.query(query):
        prefix = str(row.prefix)
        namespace = str(row.namespace)
        prefixes[prefix] = namespace
    
    # Add common prefixes that might not be in the file
    default_prefixes = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        'owl': 'http://www.w3.org/2002/07/owl#',
        'xsd': 'http://www.w3.org/2001/XMLSchema#',
        'dcterms': 'http://purl.org/dc/terms/',
        'qudt': 'http://qudt.org/schema/qudt/',
        'foaf': 'http://xmlns.com/foaf/0.1/',
        'wgs84': 'http://www.w3.org/2003/01/geo/wgs84_pos#',
    }
    
    for prefix, namespace in default_prefixes.items():
        if prefix not in prefixes:
            prefixes[prefix] = namespace
    
    return prefixes

def find_and_replace_entities(text, prefixes):
    """Find entities in text and create hyperlinks"""
    # Pattern to match prefix:localname
    entity_pattern = r'\b([a-zA-Z][a-zA-Z0-9_-]*):([a-zA-Z][a-zA-Z0-9_-]*(?:\s*\([^)]*\))?)\b'
    
    links_added = 0
    replacements = []
    
    for match in re.finditer(entity_pattern, text):
        full_match = match.group(0)
        prefix = match.group(1)
        local_part = match.group(2)
        
        # Remove optional marker for URL
        local_part_clean = re.sub(r'\s*\(optional\)\s*$', '', local_part).strip()
        
        if prefix in prefixes:
            href = f"{prefixes[prefix]}{local_part_clean}"
            replacements.append((full_match, href, match.start(), match.end()))
            links_added += 1
    
    return replacements, links_added

def add_links_to_value(value, prefixes):
    """Add hyperlinks to entities within a value, preserving structure"""
    if not value:
        return value, 0
    
    # Skip if already has links
    if '<a href=' in value:
        return value, 0
    
    links_added = 0
    
    # Strategy: Find all text content, check for entities, wrap them in <a> tags
    # Handle nested tags properly
    
    def process_tag_content(match):
        nonlocal links_added
        tag_name = match.group(1)  # font, b, i, div, span, etc.
        tag_attrs = match.group(2)  # attributes
        content = match.group(3)  # Content inside tag
        
        # Extract pure text without any tags
        text = re.sub('<[^>]+>', '', content).strip()
        
        # Check if it's an entity (has colon and is not empty)
        if ':' in text and text and not text.startswith('http'):
            parts = text.split(':', 1)
            prefix = parts[0].strip()
            local_part = parts[1].strip()
            
            # Handle edge cases like "emi:ChemicalTaxon(npc:Class)"
            # Extract just the first entity for now
            if '(' in local_part and ')' in local_part:
                # This might contain multiple entities, handle the main one
                local_part = local_part.split('(')[0].strip()
            
            # Remove optional marker
            local_part_clean = re.sub(r'\s*\(optional\)\s*$', '', local_part)
            
            if prefix in prefixes:
                href = f"{prefixes[prefix]}{local_part_clean}"
                
                # Wrap the text in <a> tag while preserving inner formatting
                # If content has nested tags, preserve them
                if '<' in content:
                    # Has nested tags, wrap carefully
                    new_content = f'<a href="{href}">{content}</a>'
                else:
                    # Plain text
                    new_content = f'<a href="{href}">{text}</a>'
                
                links_added += 1
                print(f"  ✓ {text} -> {href}")
                return f'<{tag_name}{tag_attrs}>{new_content}</{tag_name}>'
        
        # Return original if not an entity
        return match.group(0)
    
    # Process common wrapper tags: font, b, i, div, span
    new_value = re.sub(
        r'<(font|b|i|div|span)([^>]*)>(.*?)</\1>',
        process_tag_content,
        value,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    return new_value, links_added

def add_links_to_drawio(drawio_file, prefixes, output_file):
    tree = ET.parse(drawio_file)
    root = tree.getroot()
    total_links_added = 0
    
    for cell in root.iter('mxCell'):
        value = cell.get('value')
        if value:
            new_value, links_added = add_links_to_value(value, prefixes)
            if links_added > 0:
                cell.set('value', new_value)
                total_links_added += links_added
    
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    print(f"\nTotal links added: {total_links_added}")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python script.py <drawio_file> [prefix_ttl_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    prefix_file = sys.argv[2] if len(sys.argv) == 3 else None
    
    if not Path(input_file).exists():
        print(f"Error: File '{input_file}' not found!")
        sys.exit(1)
    
    # Load prefixes
    if prefix_file:
        print(f"Loading prefixes from {prefix_file}...")
        prefixes = load_prefixes(prefix_file)
    else:
        print("No prefix file provided, using default prefixes...")
        prefixes = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'owl': 'http://www.w3.org/2002/07/owl#',
            'xsd': 'http://www.w3.org/2001/XMLSchema#',
            'dcterms': 'http://purl.org/dc/terms/',
            'qudt': 'http://qudt.org/schema/qudt/',
            'emi': 'https://w3id.org/emi#',
            'wgs84': 'http://www.w3.org/2003/01/geo/wgs84_pos#',
        }
    
    print(f"Loaded {len(prefixes)} prefixes:")
    for p, ns in sorted(prefixes.items()):
        print(f"  {p}: {ns}")
    
    path = Path(input_file)
    output_file = path.parent / f"{path.stem}_withLinks.xml"
    
    print(f"\nProcessing {input_file}...")
    add_links_to_drawio(input_file, prefixes, str(output_file))
