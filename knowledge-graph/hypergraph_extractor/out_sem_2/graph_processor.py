# file: hypergraph_extractor/out_sem_2/graph_processor.py
from __future__ import annotations

__requires__ = ["Setuptools<81"]
import importlib_resources, importlib_metadata

import json
import textwrap
from pathlib import Path
from typing import Dict, List

import networkx as nx
from pyvis.network import Network
from networkx.readwrite import json_graph

# ════════════════════════════════════════════════════════════════
#  🧩 Core Data Loading and Processing Class (V14)
# ════════════════════════════════════════════════════════════════

class GraphProcessor:
    """
    Loads, processes, and visualizes graphs from HRKG and AtomSpace JSON files.
    This final version includes:
    1. Data sanitization for HG sentence nodes.
    2. A Document Hub to connect all sentences.
    3. Full sentence text in HG visualization tooltips.
    4. Exporting final graphs to Gpickle and JSON formats.
    """

    def __init__(self, source_path: Path):
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        self.source_path = source_path
        self.data = json.loads(source_path.read_text())
        self.graph_type = self.data.get("graph_type", "UNKNOWN").lower()
        self.graph: nx.MultiDiGraph = None

    def process(self):
        """Orchestrates the parsing based on the graph type."""
        print(f"Processing '{self.source_path.name}' with the Unified Reified Model (V14).")
        if "hyper-relational" in self.graph_type:
            self.graph = self._parse_hrkg_reified_model_with_hub()
        elif "atomspace hypergraph" in self.graph_type:
            self.graph = self._parse_hg_reified_model_with_hub()
        else:
            raise ValueError(f"Unsupported graph type: {self.graph_type}")
        return self

    # ==========================================================
    #  HRKG Parser
    # ==========================================================
    def _parse_hrkg_reified_model_with_hub(self) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        entities = self.data.get("entities", [])
        facts = self.data.get("facts", [])
        sentences = self.data.get("metadata", {}).get("source_sentences", [])
        doc_id = self.data.get("metadata", {}).get("document_id", "Source Document")

        G.add_node(doc_id, label=doc_id, node_class='Document', color='gold', size=25, shape='star')

        for entity in entities:
            G.add_node(entity["id"], label=entity.get("name", entity["id"]),
                       title=f"Entity: {entity.get('name')}<br>Type: {entity.get('type', 'N/A')}",
                       node_class='Entity', color='skyblue', size=15)
        for fact in facts:
            G.add_node(fact["id"], label=f"{fact['id']}: {fact.get('predicate','fact')}",
                       title=f"Fact: {fact.get('predicate')}<br>Truth: {fact.get('truth_value')}",
                       node_class='Fact', color='lightcoral', shape='square', size=10)
        for sentence in sentences:
            G.add_node(sentence["id"], label=f"Sentence {sentence['id'].split('_')[-1]}",
                       title=textwrap.fill(sentence.get('text', ''), 80),
                       node_class='Sentence', color='lightgreen', shape='box', size=20)
            G.add_edge(doc_id, sentence['id'])

        entity_name_map = {e['name']: e['id'] for e in entities}
        
        for sentence in sentences:
            for entity_name, entity_id in entity_name_map.items():
                if entity_name in sentence.get('text', ''):
                    G.add_edge(sentence['id'], entity_id, title='mentions', color='#e0e0e0')

        for fact in facts:
            fact_id, sentence_id = fact['id'], fact.get('sentence_id')
            if sentence_id and G.has_node(sentence_id):
                G.add_edge(sentence_id, fact_id, title='contains_fact')
            if fact.get("tuple"):
                for element in fact["tuple"]:
                    if element.get("entity") and G.has_node(element.get("entity")):
                        G.add_edge(fact_id, element["entity"], title=element.get("role", "member"))
            if fact.get("arguments"):
                for arg_fact_id in fact["arguments"]:
                    if G.has_node(arg_fact_id):
                        G.add_edge(fact_id, arg_fact_id, label=fact.get('predicate', 'implies'), color='purple')
        return G

    # ==========================================================
    #  HG Parser
    # ==========================================================
    def _sanitize_hg_nodes(self, atoms: List[Dict]) -> List[Dict]:
        """Pre-processes the list of atoms to standardize sentence nodes."""
        sanitized_atoms = []
        for atom in atoms:
            atom_name = atom.get("name", "")
            if "SENT_" in atom_name:
                clean_atom = atom.copy()
                clean_atom["atom_type"] = "SentenceNode"
                sent_index = atom_name.find("SENT_")
                clean_atom["name"] = atom_name[sent_index:]
                sanitized_atoms.append(clean_atom)
            else:
                sanitized_atoms.append(atom)
        return sanitized_atoms

    def _parse_hg_reified_model_with_hub(self) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        original_atoms = self.data.get("nodes", [])
        links = self.data.get("links", [])
        sentences_metadata = self.data.get("metadata", {}).get("source_sentences", [])
        doc_id = self.data.get("metadata", {}).get("document_id", "Source Document")
        
        atoms = self._sanitize_hg_nodes(original_atoms)
        sentence_text_map = {s['id']: s.get('text', '') for s in sentences_metadata}

        G.add_node(doc_id, label=doc_id, node_class='Document', color='gold', size=25, shape='star')
        timestamp_node_id = next((a['id'] for a in atoms if a.get('atom_type') == 'TimestampNode'), None)

        for atom in atoms:
            if atom['id'] == timestamp_node_id: continue
            
            is_sentence = atom.get('atom_type') == 'SentenceNode'
            # FIX: Get full sentence text for the title if it's a sentence node
            node_title = (textwrap.fill(sentence_text_map.get(atom.get('name'), ''), 80) if is_sentence 
                          else f"Atom: {atom.get('name')}<br>Type: {atom.get('atom_type', 'N/A')}")
            
            G.add_node(atom["id"], label=atom.get("name", atom["id"]), title=node_title,
                       node_class=atom.get('atom_type', 'Atom'),
                       color='lightgreen' if is_sentence else 'skyblue',
                       shape='box' if is_sentence else 'ellipse',
                       size=20 if is_sentence else 15)
            if is_sentence:
                G.add_edge(doc_id, atom['id'])
                
        for link in links:
            G.add_node(link["id"], label=link['id'],
                       title=f"Link: {link['id']}<br>Type: {link.get('link_type')}",
                       node_class='Link', color='lightcoral', shape='square', size=10)

        for link in links:
            arguments = link.get("arguments", [])
            if link.get("predicate"): arguments.append(link["predicate"])
            for arg_id in arguments:
                if arg_id != timestamp_node_id and G.has_node(arg_id):
                    G.add_edge(link["id"], arg_id, title=link.get('link_type', 'connects'))
        return G

    # ==========================================================
    #  Saving and Visualization
    # ==========================================================
    def save_gpickle(self, output_dir: Path = Path("./output_files")):
        if self.graph is None: return
        output_dir.mkdir(parents=True, exist_ok=True)
        nx.write_gpickle(self.graph, output_dir / f"{self.source_path.stem}.gpickle")
        print(f"💾 Saved gpickle file to {output_dir / f'{self.source_path.stem}.gpickle'}")

    def save_json(self, output_dir: Path = Path("./output_files")):
        """Saves the processed NetworkX graph to a JSON file."""
        if self.graph is None: return
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.source_path.stem}_graph.json"
        
        # Convert graph to a serializable format (node-link is standard)
        data = json_graph.node_link_data(self.graph)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved JSON graph file to {output_path}")

    def visualize_pyvis(self, output_dir: Path = Path("./output_visuals"), hide_isolates=False):
        if self.graph is None: return
        
        graph_to_viz = self.graph
        if hide_isolates:
            graph_copy = self.graph.to_undirected() if self.graph.is_directed() else self.graph.copy()
            isolates = list(nx.isolates(graph_copy))
            if isolates:
                viz_graph_copy = self.graph.copy()
                viz_graph_copy.remove_nodes_from(isolates)
                graph_to_viz = viz_graph_copy
                print(f"Hiding {len(isolates)} disconnected nodes for cleaner visualization.")

        output_dir.mkdir(parents=True, exist_ok=True)
        net = Network(height="900px", width="100%", notebook=True, directed=True, cdn_resources='remote')
        net.from_nx(graph_to_viz)
        net.show_buttons(filter_=['physics'])
        net.save_graph(str(output_dir / f"{self.source_path.stem}_interactive.html"))
        print(f"🌐 Saved interactive visualization to {output_dir / f'{self.source_path.stem}_interactive.html'}")

    def summary(self):
        if self.graph is None: return
        
        graph_copy = self.graph.to_undirected() if self.graph.is_directed() else self.graph.copy()
        isolates = len(list(nx.isolates(graph_copy)))
        
        print("\n" + "="*50)
        print(f"Graph Summary for: {self.source_path.name}")
        
        node_classes = [d.get('node_class', 'Unknown') for _, d in self.graph.nodes(data=True)]
        unique_classes = sorted(list(set(node_classes)))
        
        print(f"  - Model: Unified Reified Graph with Sanitization (V14)")
        for cls in unique_classes:
            print(f"  - {cls} Nodes: {node_classes.count(cls)}")
        
        print(f"  - Total Edges: {self.graph.number_of_edges()}")
        print(f"  - Disconnected (Isolated) Nodes Found: {isolates}")
        print("="*50 + "\n")


# ==============================================================================
#  🚀 Main execution block
# ==============================================================================

if __name__ == '__main__':
    base = Path.cwd()
    hrkg_file = base / "HRKG_merged/final_merged_hrkg.json"
    hg_file = base/ "HG_merged/final_merged_hg.json"

    if not hrkg_file.exists() or not hg_file.exists():
        print("Please make sure 'final_merged_hrkg.json' and 'final_merged_hg.json' are in the same directory.")
    else:
        for file_path in [hrkg_file, hg_file]:
            try:
                processor = GraphProcessor(file_path).process()
                processor.summary()
                processor.save_gpickle()
                processor.save_json()
                processor.visualize_pyvis(hide_isolates=True)
            except Exception as e:
                print(f"An error occurred while processing {file_path.name}: {e}")
                import traceback
                traceback.print_exc()