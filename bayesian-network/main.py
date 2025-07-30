from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

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
        "AST_ALT_Ratio": ["<1", "1-2", ">2"],
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
        "Diagnosis": ["No Disease", "Disease"],
        "AST_ALT_Ratio": ["<1", "1-2", ">2"],
        "GGT": ["Normal", "Elevated"],
    },
)

model.add_cpds(cpd_alcohol, cpd_liver_disease, cpd_ast_alt, cpd_ggt, cpd_diagnosis)

print(f"Model is valid: {model.check_model()}")

inference = VariableElimination(model)

# --- Query 1: Probability of Liver Disease given high AST/ALT ratio and elevated GGT ---
query1 = inference.query(
    variables=["Diagnosis"], evidence={"AST_ALT_Ratio": ">2", "GGT": "Elevated"}
)
print(
    "\n--- Query 1: Probability of Liver Disease given high AST/ALT ratio (>2) and elevated GGT ---"
)
print(query1)

# --- Query 2: Probability of Liver Disease given no alcohol abuse, but elevated GGT ---
query2 = inference.query(
    variables=["Diagnosis"], evidence={"Alcohol_Abuse": "No", "GGT": "Elevated"}
)
print(
    "\n--- Query 2: Probability of Liver Disease given no alcohol abuse, but elevated GGT ---"
)
print(query2)

# --- Query 3: Probability of having a high AST/ALT ratio given Liver Disease ---
query3 = inference.query(variables=["AST_ALT_Ratio"], evidence={"Liver_Disease": "Yes"})
print(
    "\n--- Query 3: Probability of having a high AST/ALT ratio given Liver Disease ---"
)
print(query3)
