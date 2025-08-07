#!/usr/bin/env python3
"""
Test script to verify the dynamic extra_attrs implementation for study entities.
"""

import json
import sys
import os
from pathlib import Path

# Add the hypergraph_extractor package to the path
sys.path.insert(0, 'hypergraph_extractor')

# Import the necessary modules
from models import NTKG, Entity, Fact, FactRole
from typedb_exporter import export_typeql_schema, l

def test_extra_attrs():
    """Test that extra_attrs are generated correctly for study entities."""
    
    # Create a simpler test case focused on the extra_attrs functionality
    data = {
        "graph_type": "N-tuple Hyper-Relational Temporal Knowledge Graph",
        "metadata": {
            "source_pmid": "12345",
            "publication_year": 2023
        },
        "entities": [
            { "id": "E1", "name": "Vemurafenib", "type": "Drug" },
            { "id": "E2", "name": "PMID:12345", "type": "Study" }
        ],
        "facts": [
            {
                "id": "F1",
                "predicate": "therapy_effect",
                "timestamp": "2023-01-01",
                "truth_value": 0.90,
                "sentence_id": "SENT_1",
                "tuple": [
                    { "entity": "E1", "role": "agent" },
                    { "entity": "E2", "role": "evidence_source" }
                ]
            }
        ]
    }
    
    # Create NTKG instance
    ntkg = NTKG.model_validate(data)
    
    # Test the l() function behavior
    print("Testing l() function:")
    print(f"l('source-pmid') = '{l('source-pmid')}'")
    print(f"l('publication-year') = '{l('publication-year')}'")
    
    # Generate schema and check the output
    schema_path = Path('test_schema.tql')
    export_typeql_schema(ntkg, schema_path)
    
    # Read the generated schema
    schema_content = schema_path.read_text()
    
    print("\nGenerated schema content:")
    print("=" * 50)
    print(schema_content)
    print("=" * 50)
    
    # Check if the study entity has the expected extra attributes
    study_line = None
    for line in schema_content.split('\n'):
        if 'entity study sub data-entity' in line:
            study_line = line
            break
    
    if study_line:
        print(f"\nStudy entity line: {study_line}")
        
        # Check if it contains the expected attributes (after _safe() conversion)
        expected_attrs = ['source_pmid', 'publication_year']
        
        for attr in expected_attrs:
            if f'owns {attr}' in study_line:
                print(f"✓ Found expected attribute: {attr}")
            else:
                print(f"✗ Missing expected attribute: {attr}")
        
        # Verify that the attributes are present in the attribute definitions
        print("\nChecking attribute definitions:")
        for attr in expected_attrs:
            if f'attribute {attr} value string;' in schema_content:
                print(f"✓ Found attribute definition: {attr}")
            else:
                print(f"✗ Missing attribute definition: {attr}")
    else:
        print("✗ Study entity not found in schema")
    
    # Clean up
    if schema_path.exists():
        schema_path.unlink()
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_extra_attrs()
