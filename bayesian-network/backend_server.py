import copy
import json

from flask import Flask, jsonify, request
from flask_cors import CORS
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

app = Flask(__name__)
CORS(app)


# Initialize the Bayesian Network (same as in visualize.py)
def create_base_model():
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

    # Base CPDs
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
    return model


# Global model instance
base_model = create_base_model()


def format_inference_result(variable, result):
    """Format inference results for frontend consumption"""
    if variable not in result.variables:
        return None

    states = result.state_names[variable]
    probabilities = result.values

    return {
        "variable": variable,
        "states": states,
        "probabilities": [float(p) for p in probabilities],
    }


@app.route("/api/inference", methods=["POST"])
def perform_inference():
    """Perform inference with given evidence"""
    try:
        data = request.get_json()
        evidence = data.get("evidence", {})

        filtered_evidence = {}
        for var, value in evidence.items():
            if value is not None and value != "" and value != "None":
                filtered_evidence[var] = value

        print(f"Received evidence: {evidence}")
        print(f"Filtered evidence: {filtered_evidence}")

        model = copy.deepcopy(base_model)

        priors = data.get("priors", {})
        if priors:
            if "Alcohol_Abuse" in priors:
                alcohol_prob = priors["Alcohol_Abuse"]
                cpd_alcohol = TabularCPD(
                    variable="Alcohol_Abuse",
                    variable_card=2,
                    values=[[1 - alcohol_prob], [alcohol_prob]],
                    state_names={"Alcohol_Abuse": ["No", "Yes"]},
                )
                model.remove_cpds("Alcohol_Abuse")
                model.add_cpds(cpd_alcohol)

            if "Liver_Disease" in priors:
                liver_prob = priors["Liver_Disease"]
                cpd_liver = TabularCPD(
                    variable="Liver_Disease",
                    variable_card=2,
                    values=[[1 - liver_prob], [liver_prob]],
                    state_names={"Liver_Disease": ["No", "Yes"]},
                )
                model.remove_cpds("Liver_Disease")
                model.add_cpds(cpd_liver)

        inference = VariableElimination(model)

        results = {}
        for variable in model.nodes():
            if variable not in filtered_evidence:
                try:
                    if filtered_evidence:
                        result = inference.query(
                            variables=[variable], evidence=filtered_evidence
                        )
                    else:
                        result = inference.query(variables=[variable])
                    results[variable] = format_inference_result(variable, result)
                except Exception as e:
                    print(f"Error inferring {variable}: {e}")
                    continue
            else:
                observed_state = filtered_evidence[variable]
                states = model.get_cpds(variable).state_names[variable]
                probabilities = [
                    1.0 if state == observed_state else 0.0 for state in states
                ]
                results[variable] = {
                    "variable": variable,
                    "states": states,
                    "probabilities": probabilities,
                }

        return jsonify(
            {
                "success": True,
                "results": results,
                "evidence": filtered_evidence,
                "priors_used": priors,
            }
        )

    except Exception as e:
        print(f"Inference error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/marginals", methods=["GET"])
def get_marginals():
    """Get marginal probabilities for all variables"""
    try:
        model = copy.deepcopy(base_model)
        inference = VariableElimination(model)

        results = {}
        for variable in model.nodes():
            result = inference.query(variables=[variable])
            results[variable] = format_inference_result(variable, result)

        return jsonify({"success": True, "results": results})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/controllable_variables", methods=["GET"])
def get_controllable_variables():
    """Get variables that can be controlled via sliders"""
    return jsonify(
        {
            "priors": [
                {
                    "id": "Alcohol_Abuse",
                    "name": "Alcohol Abuse Prior",
                    "description": "Prior probability of alcohol abuse",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.3,
                    "step": 0.01,
                },
                {
                    "id": "Liver_Disease",
                    "name": "Liver Disease Prior",
                    "description": "Prior probability of liver disease",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.2,
                    "step": 0.01,
                },
            ],
            "evidence": [
                {
                    "id": "AST_ALT_Ratio",
                    "name": "AST/ALT Ratio",
                    "description": "Set observed AST/ALT ratio level",
                    "states": ["< 1.0", "1.0 - 2.0", "> 2.0"],
                    "default": None,
                },
                {
                    "id": "GGT",
                    "name": "GGT Level",
                    "description": "Set observed GGT enzyme level",
                    "states": ["Normal", "Elevated"],
                    "default": None,
                },
                {
                    "id": "Diagnosis",
                    "name": "Final Diagnosis",
                    "description": "Set observed final diagnosis",
                    "states": ["Healthy", "Disease"],
                    "default": None,
                },
            ],
        }
    )


if __name__ == "__main__":
    print("🚀 Starting Bayesian Network Inference Server...")
    print("📊 Interactive sliders available for:")
    print("   • Alcohol Abuse Prior Probability")
    print("   • Liver Disease Prior Probability")
    print("🌐 Server running on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
