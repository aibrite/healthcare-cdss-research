import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

data = pd.DataFrame(
    {
        "ChestPain": [
            "typical",
            "atypical",
            "non-anginal",
            "asymptomatic",
            "typical",
            "atypical",
            "typical",
            "asymptomatic",
        ],
        "ExerciseAngina": ["no", "yes", "no", "no", "yes", "no", "yes", "no"],
        "RestECG": [
            "normal",
            "abnormal",
            "normal",
            "normal",
            "abnormal",
            "normal",
            "abnormal",
            "normal",
        ],
        "HeartDisease": [
            "absent",
            "present",
            "absent",
            "absent",
            "present",
            "absent",
            "present",
            "absent",
        ],
    }
)

data["ChestPain"] = data["ChestPain"].map(
    {"typical": 0, "atypical": 1, "non-anginal": 2, "asymptomatic": 3}
)
data["ExerciseAngina"] = data["ExerciseAngina"].map({"no": 0, "yes": 1})
data["RestECG"] = data["RestECG"].map(
    {"normal": 0, "abnormal": 1, "left_ventricular_hypertrophy": 2}
)
data["HeartDisease"] = data["HeartDisease"].map({"absent": 0, "present": 1})

model = DiscreteBayesianNetwork()

model.add_nodes_from(["ChestPain", "ExerciseAngina", "RestECG", "HeartDisease"])

model.add_edges_from(
    [
        ("ChestPain", "HeartDisease"),
        ("ExerciseAngina", "HeartDisease"),
        ("RestECG", "HeartDisease"),
    ]
)

model.fit(data, estimator=MaximumLikelihoodEstimator)

print("CPD for ChestPain:")
print(model.get_cpds("ChestPain"))
print("\nCPD for ExerciseAngina:")
print(model.get_cpds("ExerciseAngina"))
print("\nCPD for RestECG:")
print(model.get_cpds("RestECG"))
print("\nCPD for HeartDisease (given parents):")
print(model.get_cpds("HeartDisease"))

infer = VariableElimination(model)

print("\n--- Inference Queries ---")
query_result_1 = infer.query(
    variables=["HeartDisease"],
    evidence={"ChestPain": 0, "ExerciseAngina": 0, "RestECG": 0},
)
print(
    "\nProbability of HeartDisease given typical angina, no exercise angina, normal ECG:"
)
print(query_result_1)

query_result_2 = infer.query(
    variables=["HeartDisease"],
    evidence={"ChestPain": 1, "ExerciseAngina": 1, "RestECG": 1},
)
print(
    "\nProbability of HeartDisease given atypical angina, yes exercise angina, abnormal ECG:"
)
print(query_result_2)

query_result_3 = infer.query(variables=["ChestPain"], evidence={"HeartDisease": 1})
print("\nProbability of ChestPain given HeartDisease is present:")
print(query_result_3)
