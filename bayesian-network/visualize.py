import json

import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


def format_cpd_data(cpd):
    """
    Converts a pgmpy TabularCPD object into structured data for JavaScript.
    """
    variable = cpd.variable
    evidence = cpd.variables[1:]

    if not evidence:
        states = cpd.state_names[variable]
        probabilities = cpd.values.flatten()
        return {
            "variable": variable,
            "type": "marginal",
            "data": [
                {"state": f"{variable} = {state}", "probability": f"{prob:.3f}"}
                for state, prob in zip(states, probabilities)
            ],
        }
    else:
        evidence_states = [cpd.state_names[e] for e in evidence]
        evidence_combinations = list(
            pd.MultiIndex.from_product(evidence_states, names=evidence)
        )

        columns = []
        for combo in evidence_combinations:
            if len(evidence) == 1:
                columns.append(f"P({variable} | {evidence[0]} = {combo})")
            else:
                evidence_str = ", ".join(
                    [f"{ev} = {val}" for ev, val in zip(evidence, combo)]
                )
                columns.append(f"P({variable} | {evidence_str})")

        reshaped_values = cpd.values.reshape(cpd.values.shape[0], -1)

        rows = []
        for i, state in enumerate(cpd.state_names[variable]):
            row = {"state": f"{variable} = {state}"}
            for j, col in enumerate(columns):
                row[col] = f"{reshaped_values[i][j]:.3f}"
            rows.append(row)

        return {
            "variable": variable,
            "type": "conditional",
            "columns": columns,
            "data": rows,
        }


model = DiscreteBayesianNetwork(
    [
        ("Alcohol_Abuse", "AST_ALT_Ratio"),
        ("Alcohol_Abuse", "GGT"),
        ("Liver_Disease", "AST_ALT_Ratio"),
        ("Liver_Disease", "GGT"),
        ("AST_ALT_Ratio", "Diagnosis"),
        ("GGT", "Diagnosis"),
    ]
)

cpd_alcohol = TabularCPD(
    variable="Alcohol_Abuse",
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={"Alcohol_Abuse": ["No", "Yes"]},
)

cpd_liver_disease = TabularCPD(
    variable="Liver_Disease",
    variable_card=2,
    values=[[0.8], [0.2]],
    state_names={"Liver_Disease": ["No", "Yes"]},
)

cpd_ast_alt = TabularCPD(
    variable="AST_ALT_Ratio",
    variable_card=3,
    values=[[0.8, 0.4, 0.3, 0.1], [0.1, 0.4, 0.4, 0.3], [0.1, 0.2, 0.3, 0.6]],
    evidence=["Alcohol_Abuse", "Liver_Disease"],
    evidence_card=[2, 2],
    state_names={
        "AST_ALT_Ratio": ["< 1.0", "1.0 - 2.0", "> 2.0"],
        "Alcohol_Abuse": ["No", "Yes"],
        "Liver_Disease": ["No", "Yes"],
    },
)

cpd_ggt = TabularCPD(
    variable="GGT",
    variable_card=2,
    values=[[0.9, 0.3, 0.2, 0.1], [0.1, 0.7, 0.8, 0.9]],
    evidence=["Alcohol_Abuse", "Liver_Disease"],
    evidence_card=[2, 2],
    state_names={
        "GGT": ["Normal", "Elevated"],
        "Alcohol_Abuse": ["No", "Yes"],
        "Liver_Disease": ["No", "Yes"],
    },
)

cpd_diagnosis = TabularCPD(
    variable="Diagnosis",
    variable_card=2,
    values=[[0.95, 0.8, 0.6, 0.2, 0.1, 0.05], [0.05, 0.2, 0.4, 0.8, 0.9, 0.95]],
    evidence=["AST_ALT_Ratio", "GGT"],
    evidence_card=[3, 2],
    state_names={
        "Diagnosis": ["Healthy", "Disease"],
        "AST_ALT_Ratio": ["< 1.0", "1.0 - 2.0", "> 2.0"],
        "GGT": ["Normal", "Elevated"],
    },
)

model.add_cpds(cpd_alcohol, cpd_liver_disease, cpd_ast_alt, cpd_ggt, cpd_diagnosis)

nodes_data = []
edges_data = []
cpd_data = {}

node_configs = {
    "Alcohol_Abuse": {
        "x": 200,
        "y": 100,
        "color": "#ff6b6b",
        "icon": "🍺",
        "label": "Alcohol Abuse",
        "type": "risk-factor",
        "shape": "circle",
    },
    "Liver_Disease": {
        "x": 600,
        "y": 100,
        "color": "#ff9800",
        "icon": "🏥",
        "label": "Liver Disease",
        "type": "risk-factor",
        "shape": "circle",
    },
    "AST_ALT_Ratio": {
        "x": 200,
        "y": 300,
        "color": "#4caf50",
        "icon": "📊",
        "label": "AST/ALT Ratio",
        "type": "biomarker",
        "shape": "rect",
    },
    "GGT": {
        "x": 600,
        "y": 300,
        "color": "#2196f3",
        "icon": "🧪",
        "label": "GGT Level",
        "type": "biomarker",
        "shape": "rect",
    },
    "Diagnosis": {
        "x": 400,
        "y": 500,
        "color": "#9c27b0",
        "icon": "⚕️",
        "label": "Final Diagnosis",
        "type": "outcome",
        "shape": "diamond",
    },
}

for node_name in model.nodes():
    cpd = model.get_cpds(node_name)
    cpd_data[node_name] = format_cpd_data(cpd)

    config = node_configs[node_name]
    nodes_data.append(
        {
            "id": node_name,
            "x": config["x"],
            "y": config["y"],
            "color": config["color"],
            "icon": config["icon"],
            "label": config["label"],
            "type": config["type"],
            "shape": config["shape"],
        }
    )

for edge in model.edges():
    edges_data.append({"from": edge[0], "to": edge[1]})

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Liver Disease Diagnosis - Bayesian Network</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: white;
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 30px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            backdrop-filter: blur(15px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            background: linear-gradient(45deg, #fff, #e8eaff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
            margin: 5px 0;
        }}

        .legend {{
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 15px;
            margin: 20px auto;
            max-width: 900px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            color: #2d3748;
        }}

        .legend h3 {{
            margin: 0 0 20px 0;
            text-align: center;
            font-size: 1.3em;
        }}

        .legend-items {{
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }}

        .legend-item {{
            padding: 12px 20px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .risk-factor {{ background: linear-gradient(135deg, #ffebee, #ffcdd2); color: #c62828; }}
        .biomarker {{ background: linear-gradient(135deg, #e8f5e8, #c8e6c9); color: #2e7d32; }}
        .outcome {{ background: linear-gradient(135deg, #f3e5f5, #e1bee7); color: #7b1fa2; }}

        .network-container {{
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.2);
            margin: 0 auto;
            max-width: 1200px;
            overflow: hidden;
            position: relative;
            width: 100%;
            height: 700px;
        }}

        #network {{
            width: 100%;
            height: 100%;
            display: block;
        }}

        .tooltip {{
            position: absolute;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 15px 50px rgba(0,0,0,0.3);
            color: white;
            min-width: 300px;
            max-width: 90vw;
            max-height: 80vh;
            z-index: 1000;
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.3s ease;
            pointer-events: none;
            backdrop-filter: blur(10px);
            overflow: auto;
        }}

        .tooltip.show {{
            opacity: 1;
            transform: translateY(0);
        }}

        .tooltip h3 {{
            margin: 0 0 15px 0;
            font-size: 18px;
            text-align: center;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }}

        .tooltip h4 {{
            margin: 0 0 15px 0;
            font-size: 16px;
            text-align: center;
            color: #e8eaff;
        }}

        .cpd-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            overflow: hidden;
            table-layout: auto;
        }}

        .cpd-table th {{
            background: rgba(255,255,255,0.2);
            color: white;
            font-weight: 600;
            padding: 8px 6px;
            text-align: center;
            font-size: 10px;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            white-space: normal;
            word-wrap: break-word;
            line-height: 1.2;
            min-width: 80px;
        }}

        .cpd-table td {{
            padding: 8px 6px;
            text-align: center;
            font-size: 11px;
            font-weight: 500;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            background: rgba(255,255,255,0.05);
            white-space: nowrap;
        }}

        .cpd-table.wide-table th {{
            font-size: 9px;
            padding: 6px 4px;
        }}

        .cpd-table.wide-table td {{
            font-size: 10px;
            padding: 6px 4px;
        }}

        .cpd-table tr:hover td {{
            background: rgba(255,255,255,0.15);
        }}

        .cpd-table tbody tr:last-child td {{
            border-bottom: none;
        }}

        .instructions {{
            text-align: center;
            margin-top: 30px;
            font-size: 1.1em;
            opacity: 0.9;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }}

        /* Modal Styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(5px);
            animation: fadeIn 0.3s ease;
        }}

        .modal.show {{
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .modal-content {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 30px;
            max-width: 95vw;
            max-height: 90vh;
            overflow: auto;
            box-shadow: 0 25px 80px rgba(0,0,0,0.4);
            color: white;
            position: relative;
            animation: slideIn 0.3s ease;
        }}

        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 15px;
        }}

        .modal-title {{
            font-size: 24px;
            margin: 0;
            background: linear-gradient(45deg, #fff, #e8eaff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .close-button {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            font-size: 24px;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }}

        .close-button:hover {{
            background: rgba(255,255,255,0.3);
            transform: scale(1.1);
        }}

        .modal-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            overflow: hidden;
            margin-top: 20px;
        }}

        .modal-table th {{
            background: rgba(255,255,255,0.2);
            color: white;
            font-weight: 600;
            padding: 15px 12px;
            text-align: center;
            font-size: 14px;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            white-space: normal;
            word-wrap: break-word;
            line-height: 1.3;
            vertical-align: middle;
        }}

        .modal-table td {{
            padding: 12px;
            text-align: center;
            font-size: 14px;
            font-weight: 500;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            background: rgba(255,255,255,0.05);
            white-space: nowrap;
        }}

        .modal-table tr:hover td {{
            background: rgba(255,255,255,0.15);
        }}

        .modal-table tbody tr:last-child td {{
            border-bottom: none;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        @keyframes slideIn {{
            from {{ 
                opacity: 0; 
                transform: translateY(-50px) scale(0.9); 
            }}
            to {{ 
                opacity: 1; 
                transform: translateY(0) scale(1); 
            }}
        }}

        .instructions strong {{
            color: #fff;
        }}

        .node {{
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .node-circle {{
            r: 40;
        }}

        .node-rect {{
            width: 100;
            height: 70;
            rx: 10;
            ry: 10;
        }}

        .node-diamond {{
            transform-origin: center;
        }}

        .node-text {{
            text-anchor: middle;
            dominant-baseline: middle;
            fill: white;
            font-weight: 600;
            font-size: 12px;
            pointer-events: none;
        }}

        .node-icon {{
            text-anchor: middle;
            dominant-baseline: middle;
            fill: white;
            font-size: 20px;
            pointer-events: none;
        }}

        .edge {{
            stroke: #4a5568;
            stroke-width: 3;
            fill: none;
            marker-end: url(#arrowhead);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Liver Disease Diagnosis</h1>
        <p>Interactive Bayesian Network Visualization</p>
        <p style="font-size: 1em; opacity: 0.8;">Professional Medical Decision Support System</p>
    </div>

    <div class="legend">
        <h3>📋 Network Components</h3>
        <div class="legend-items">
            <div class="legend-item risk-factor">🍺 Risk Factors</div>
            <div class="legend-item biomarker">🧪 Laboratory Biomarkers</div>
            <div class="legend-item outcome">⚕️ Clinical Diagnosis</div>
        </div>
    </div>

    <div class="network-container">
        <svg id="network" viewBox="0 0 800 600" preserveAspectRatio="xMidYMid meet">
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                        refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#4a5568"/>
                </marker>
                
                <!-- Gradients for nodes -->
                <linearGradient id="alcoholGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#ff6b6b;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#ff5252;stop-opacity:1" />
                </linearGradient>
                
                <linearGradient id="liverGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#ff9800;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#f57c00;stop-opacity:1" />
                </linearGradient>
                
                <linearGradient id="astGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#4caf50;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#388e3c;stop-opacity:1" />
                </linearGradient>
                
                <linearGradient id="ggtGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#2196f3;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#1976d2;stop-opacity:1" />
                </linearGradient>
                
                <linearGradient id="diagnosisGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#9c27b0;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#7b1fa2;stop-opacity:1" />
                </linearGradient>
            </defs>
        </svg>
        <div id="tooltip" class="tooltip"></div>
    </div>

    <div class="instructions">
        <p>💡 <strong>Interactive Features:</strong> Hover over nodes for quick probability view • Click nodes for detailed probability tables • Drag nodes to rearrange the network</p>
    </div>

    <!-- Modal for detailed probability tables -->
    <div id="probabilityModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title" id="modalTitle"></h2>
                <button class="close-button" onclick="closeModal()">&times;</button>
            </div>
            <div id="modalTableContainer"></div>
        </div>
    </div>

    <script>
        // Data from Python - using normalized coordinates (0-1 range)
        const nodes = [
            {{
                'id': 'Alcohol_Abuse',
                'x': 0.25, 'y': 0.2,
                'color': 'url(#alcoholGradient)',
                'icon': '🍺',
                'label': 'Alcohol Abuse',
                'type': 'risk-factor',
                'shape': 'circle'
            }},
            {{
                'id': 'Liver_Disease',
                'x': 0.75, 'y': 0.2,
                'color': 'url(#liverGradient)',
                'icon': '🏥',
                'label': 'Liver Disease',
                'type': 'risk-factor',
                'shape': 'circle'
            }},
            {{
                'id': 'AST_ALT_Ratio',
                'x': 0.25, 'y': 0.5,
                'color': 'url(#astGradient)',
                'icon': '📊',
                'label': 'AST/ALT Ratio',
                'type': 'biomarker',
                'shape': 'rect'
            }},
            {{
                'id': 'GGT',
                'x': 0.75, 'y': 0.5,
                'color': 'url(#ggtGradient)',
                'icon': '🧪',
                'label': 'GGT Level',
                'type': 'biomarker',
                'shape': 'rect'
            }},
            {{
                'id': 'Diagnosis',
                'x': 0.5, 'y': 0.8,
                'color': 'url(#diagnosisGradient)',
                'icon': '⚕️',
                'label': 'Final Diagnosis',
                'type': 'outcome',
                'shape': 'diamond'
            }}
        ];
        
        const edges = {json.dumps(edges_data, indent=2)};
        const cpdData = {json.dumps(cpd_data, indent=2)};

        // Initialize the network
        const svg = document.getElementById('network');
        const tooltip = document.getElementById('tooltip');
        let isDragging = false;
        let currentNode = null;
        
        // Get SVG dimensions
        function getSVGDimensions() {{
            const rect = svg.getBoundingClientRect();
            return {{ width: rect.width, height: rect.height }};
        }}
        
        // Convert normalized coordinates to SVG coordinates
        function normalizedToSVG(normalizedX, normalizedY) {{
            return {{
                x: normalizedX * 800,
                y: normalizedY * 600
            }};
        }}
        
        // Convert SVG coordinates to normalized coordinates
        function svgToNormalized(svgX, svgY) {{
            return {{
                x: svgX / 800,
                y: svgY / 600
            }};
        }}

        // Create edges first (so they appear behind nodes)
        function createEdges() {{
            edges.forEach(edge => {{
                const fromNode = nodes.find(n => n.id === edge.from);
                const toNode = nodes.find(n => n.id === edge.to);
                
                const fromSVG = normalizedToSVG(fromNode.x, fromNode.y);
                const toSVG = normalizedToSVG(toNode.x, toNode.y);
                
                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                const controlX = (fromSVG.x + toSVG.x) / 2;
                const controlY = (fromSVG.y + toSVG.y) / 2 - 40;
                const d = `M ${{fromSVG.x}} ${{fromSVG.y}} Q ${{controlX}} ${{controlY}} ${{toSVG.x}} ${{toSVG.y}}`;
                
                path.setAttribute('d', d);
                path.setAttribute('class', 'edge');
                path.setAttribute('data-from', edge.from);
                path.setAttribute('data-to', edge.to);
                svg.appendChild(path);
            }});
        }}

        // Create nodes
        function createNodes() {{
            nodes.forEach(node => {{
                const svgCoords = normalizedToSVG(node.x, node.y);
                const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                group.setAttribute('class', 'node');
                group.setAttribute('data-id', node.id);
                group.setAttribute('transform', `translate(${{svgCoords.x}}, ${{svgCoords.y}})`);

                let shape;
                if (node.shape === 'circle') {{
                    shape = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    shape.setAttribute('r', '40');
                    shape.setAttribute('fill', node.color);
                    shape.setAttribute('stroke', 'rgba(255,255,255,0.3)');
                    shape.setAttribute('stroke-width', '3');
                    shape.setAttribute('filter', 'drop-shadow(0 4px 8px rgba(0,0,0,0.3))');
                }} else if (node.shape === 'rect') {{
                    shape = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    shape.setAttribute('x', '-50');
                    shape.setAttribute('y', '-35');
                    shape.setAttribute('width', '100');
                    shape.setAttribute('height', '70');
                    shape.setAttribute('rx', '10');
                    shape.setAttribute('ry', '10');
                    shape.setAttribute('fill', node.color);
                    shape.setAttribute('stroke', 'rgba(255,255,255,0.3)');
                    shape.setAttribute('stroke-width', '3');
                    shape.setAttribute('filter', 'drop-shadow(0 4px 8px rgba(0,0,0,0.3))');
                }} else if (node.shape === 'diamond') {{
                    shape = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    shape.setAttribute('x', '-45');
                    shape.setAttribute('y', '-45');
                    shape.setAttribute('width', '90');
                    shape.setAttribute('height', '90');
                    shape.setAttribute('rx', '15');
                    shape.setAttribute('ry', '15');
                    shape.setAttribute('fill', node.color);
                    shape.setAttribute('stroke', 'rgba(255,255,255,0.3)');
                    shape.setAttribute('stroke-width', '3');
                    shape.setAttribute('transform', 'rotate(45)');
                    shape.setAttribute('filter', 'drop-shadow(0 4px 8px rgba(0,0,0,0.3))');
                }}

                const iconText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                iconText.setAttribute('class', 'node-icon');
                iconText.setAttribute('y', '-8');
                iconText.textContent = node.icon;

                const labelText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                labelText.setAttribute('class', 'node-text');
                labelText.setAttribute('y', '8');
                labelText.textContent = node.label;

                group.appendChild(shape);
                group.appendChild(iconText);
                group.appendChild(labelText);

                // Add event listeners
                group.addEventListener('mouseenter', (e) => {{
                    showTooltip(e, node.id);
                    // Scale up the node using SVG transform
                    const currentTransform = group.getAttribute('transform');
                    if (!currentTransform.includes('scale')) {{
                        group.setAttribute('transform', currentTransform + ' scale(1.1)');
                    }}
                }});
                group.addEventListener('mouseleave', (e) => {{
                    hideTooltip();
                    // Reset scale by removing it from transform
                    const currentTransform = group.getAttribute('transform');
                    const newTransform = currentTransform.replace(/ scale\([^)]*\)/, '');
                    group.setAttribute('transform', newTransform);
                }});
                group.addEventListener('click', (e) => {{
                    e.stopPropagation();
                    openModal(node.id);
                }});
                group.addEventListener('mousedown', (e) => startDrag(e, node));

                svg.appendChild(group);
            }});
        }}

        // Tooltip functions
        function showTooltip(event, nodeId) {{
            const cpd = cpdData[nodeId];
            let tableHtml = '';

            if (cpd.type === 'marginal') {{
                tableHtml = `
                    <table class="cpd-table">
                        <thead>
                            <tr><th>State</th><th>Probability</th></tr>
                        </thead>
                        <tbody>
                            ${{cpd.data.map(row => 
                                `<tr><td>${{row.state}}</td><td>${{row.probability}}</td></tr>`
                            ).join('')}}
                        </tbody>
                    </table>
                `;
            }} else {{
                const headers = ['State', ...cpd.columns];
                const isWideTable = headers.length > 4;
                const tableClass = isWideTable ? 'cpd-table wide-table' : 'cpd-table';
                tableHtml = `
                    <table class="${{tableClass}}">
                        <thead>
                            <tr>${{headers.map(h => `<th title="${{h}}">${{h.length > 25 ? h.substring(0, 25) + '...' : h}}</th>`).join('')}}</tr>
                        </thead>
                        <tbody>
                            ${{cpd.data.map(row => {{
                                const cells = [row.state, ...cpd.columns.map(col => row[col])];
                                return `<tr>${{cells.map(cell => `<td>${{cell}}</td>`).join('')}}</tr>`;
                            }}).join('')}}
                        </tbody>
                    </table>
                `;
            }}

            tooltip.innerHTML = `
                <h3>📊 Conditional Probability Distribution</h3>
                <h4>${{cpd.variable}}</h4>
                <div style="overflow-x: auto; max-width: 100%;">
                    ${{tableHtml}}
                </div>
            `;

            tooltip.classList.add('show');
            positionTooltip(event);
        }}

        function hideTooltip() {{
            tooltip.classList.remove('show');
        }}

        function positionTooltip(event) {{
            const containerRect = document.querySelector('.network-container').getBoundingClientRect();
            const svgRect = svg.getBoundingClientRect();
            
            // Get mouse position relative to container
            const mouseX = event.clientX - containerRect.left;
            const mouseY = event.clientY - containerRect.top;
            
            // Calculate tooltip position with overflow prevention
            let tooltipX = mouseX + 20;
            let tooltipY = mouseY - tooltip.offsetHeight - 10;
            
            // Prevent horizontal overflow
            if (tooltipX + tooltip.offsetWidth > containerRect.width) {{
                tooltipX = mouseX - tooltip.offsetWidth - 20;
            }}
            
            // Prevent vertical overflow
            if (tooltipY < 0) {{
                tooltipY = mouseY + 20;
            }}
            
            // Ensure tooltip stays within container bounds
            tooltipX = Math.max(10, Math.min(tooltipX, containerRect.width - tooltip.offsetWidth - 10));
            tooltipY = Math.max(10, Math.min(tooltipY, containerRect.height - tooltip.offsetHeight - 10));
            
            tooltip.style.left = tooltipX + 'px';
            tooltip.style.top = tooltipY + 'px';
        }}

        // Drag functionality
        function startDrag(event, node) {{
            isDragging = true;
            currentNode = node;
            event.preventDefault();
        }}

        svg.addEventListener('mousemove', (event) => {{
            if (isDragging && currentNode) {{
                const svgRect = svg.getBoundingClientRect();
                const svgX = ((event.clientX - svgRect.left) / svgRect.width) * 800;
                const svgY = ((event.clientY - svgRect.top) / svgRect.height) * 600;
                
                // Update normalized coordinates
                currentNode.x = svgX / 800;
                currentNode.y = svgY / 600;
                
                // Update node position
                const nodeElement = svg.querySelector(`[data-id="${{currentNode.id}}"]`);
                nodeElement.setAttribute('transform', `translate(${{svgX}}, ${{svgY}})`);
                // Clear any CSS transform when dragging to avoid conflicts
                nodeElement.style.transform = '';
                
                updateEdges();
            }}
        }});

        svg.addEventListener('mouseup', () => {{
            isDragging = false;
            currentNode = null;
        }});

        svg.addEventListener('mouseleave', () => {{
            isDragging = false;
            currentNode = null;
        }});

        function updateEdges() {{
            edges.forEach(edge => {{
                const fromNode = nodes.find(n => n.id === edge.from);
                const toNode = nodes.find(n => n.id === edge.to);
                const edgeElement = svg.querySelector(`[data-from="${{edge.from}}"][data-to="${{edge.to}}"]`);
                
                if (edgeElement && fromNode && toNode) {{
                    const fromSVG = normalizedToSVG(fromNode.x, fromNode.y);
                    const toSVG = normalizedToSVG(toNode.x, toNode.y);
                    const controlX = (fromSVG.x + toSVG.x) / 2;
                    const controlY = (fromSVG.y + toSVG.y) / 2 - 40;
                    const d = `M ${{fromSVG.x}} ${{fromSVG.y}} Q ${{controlX}} ${{controlY}} ${{toSVG.x}} ${{toSVG.y}}`;
                    edgeElement.setAttribute('d', d);
                }}
            }});
        }}

        // Modal functions
        function openModal(nodeId) {{
            const modal = document.getElementById('probabilityModal');
            const modalTitle = document.getElementById('modalTitle');
            const modalTableContainer = document.getElementById('modalTableContainer');
            
            const cpd = cpdData[nodeId];
            const nodeInfo = nodes.find(n => n.id === nodeId);
            
            modalTitle.textContent = `${{nodeInfo.icon}} ${{nodeInfo.label}} - Probability Distribution`;
            
            let tableHtml = '';
            if (cpd.type === 'marginal') {{
                tableHtml = `
                    <table class="modal-table">
                        <thead>
                            <tr><th>State</th><th>Probability</th></tr>
                        </thead>
                        <tbody>
                            ${{cpd.data.map(row => 
                                `<tr><td>${{row.state}}</td><td>${{row.probability}}</td></tr>`
                            ).join('')}}
                        </tbody>
                    </table>
                `;
            }} else {{
                const headers = ['State', ...cpd.columns];
                tableHtml = `
                    <table class="modal-table">
                        <thead>
                            <tr>${{headers.map(h => `<th>${{h}}</th>`).join('')}}</tr>
                        </thead>
                        <tbody>
                            ${{cpd.data.map(row => {{
                                const cells = [row.state, ...cpd.columns.map(col => row[col])];
                                return `<tr>${{cells.map(cell => `<td>${{cell}}</td>`).join('')}}</tr>`;
                            }}).join('')}}
                        </tbody>
                    </table>
                `;
            }}
            
            modalTableContainer.innerHTML = tableHtml;
            modal.classList.add('show');
            
            // Prevent body scroll when modal is open
            document.body.style.overflow = 'hidden';
        }}

        function closeModal() {{
            const modal = document.getElementById('probabilityModal');
            modal.classList.remove('show');
            
            // Restore body scroll
            document.body.style.overflow = '';
        }}

        // Close modal when clicking outside of it
        window.addEventListener('click', (event) => {{
            const modal = document.getElementById('probabilityModal');
            if (event.target === modal) {{
                closeModal();
            }}
        }});

        // Close modal with Escape key
        window.addEventListener('keydown', (event) => {{
            if (event.key === 'Escape') {{
                closeModal();
            }}
        }});

        // Initialize the visualization
        createEdges();
        createNodes();
    </script>
</body>
</html>
"""

file_name = "liver_disease_network_with_cpds.html"
with open(file_name, "w", encoding="utf-8") as f:
    f.write(html_content)

print("🎉 Successfully generated custom professional interactive visualization!")
print(f"📁 File saved as: {file_name}")
print("🌐 Open the file in your browser to view the interactive Bayesian network.")
print()
print("✨ Custom Features:")
print("   • Pure HTML/CSS/JavaScript - no external dependencies")
print("   • Beautiful gradient backgrounds and modern styling")
print("   • Properly formatted probability tables in tooltips")
print("   • Smooth drag-and-drop functionality for nodes")
print("   • Curved edges with arrow indicators")
print("   • Responsive design with professional typography")
print("   • Color-coded node types with icons")
print("   • Hover effects and smooth animations")
print("   • Professional medical dashboard appearance")
