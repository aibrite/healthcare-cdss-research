import pandas as pd

# Biomarker Thresholds Data -----------------------------------------------------------------------------------------------------------------

df1 = pd.read_csv("all_biomarker_thresholds.csv")
df2 = pd.read_csv("extended_biomarker_reference.csv")

merged_df = pd.concat([df1, df2], ignore_index=True)
merged_df = merged_df.drop_duplicates(subset='Test Name', keep='first')
merged_df.to_csv("biomarker_thresholds.csv", index=False)

mapping = {
    "Alanin Aminotransferaz\n(alt) (serum/plazma)": "ALT",
    "Albümin\n(serum/plazma)": "Albumin",
    "Alkalen Fosfataz\n(serum/plazma)": "ALP",
    "Aspartat\nAminotransferaz (ast)\n(serum/plazma)": "AST",
    "BASO#": "Basophils",
    "BASO%": "BASO %",
    "Bakır (Cu), Serum": "Serum Copper (Cu)",
    "Bakır (Cu), Serum µg/L": "Serum Copper (Cu) µg/L",
    "Bilirubin, Direkt\n(serum/plazma)": "Bilirubin Direct",
    "Bilirubin, Direkt (vücut\nSıvıları)": "Bilirubin Direct (Body Fluids)",
    "Bilirubin, Total\n(serum/plazma)": "Total Bilirubin",
    "C Reaktif Protein (crp)": "CRP (high-sens)",
    "DNI": "DNI",
    "Demir (serum/plazma)": "Serum Iron",
    "Doymamış Demir\nBağlama Kapasitesi": "Unsaturated Iron Binding Capacity (UIBC)",
    "EOS#": "Eosinophils",
    "EOS%": "EOS %",
    "Ferritin (serum/plazma)": "Ferritin",
    "Folat (serum/plazma)": "Serum Folate",
    "Fosfor (serum/plazma)": "Phosphate (Serum)",
    "Gamma Glutamil\nTransferaz (ggt)\n(serum/plazma)": "Gamma-glutamyl Transferase (GGT)",
    "Glukoz (serum/plazma)": "Fasting Glucose",
    "HCT": "Hematocrit",
    "HDL kolesterol": "HDL-C",
    "HGB": "Hemoglobin",
    "HbA1c (Elektroforez)": "HbA1c",
    "Kalsiyum\n(serum/plazma)": "Calcium (total)",
    "Klorür (serum/plazma)": "Chloride",
    "Kreatinin": "Creatinine",
    "LUC#": "LUC",
    "LUC%": "LUC %",
    "LYM#": "Lymphocytes",
    "LYM%": "LYM %",
    "Ldl Kolesterol (direkt)": "LDL-C",
    "MCH": "MCH",
    "MCHC": "MCHC",
    "MCV": "MCV",
    "MONO#": "Monocytes",
    "MONO%": "MONO %",
    "MPV": "MPV",
    "Magnezyum\n(serum/plazma)": "Magnesium (Serum)",
    "NEU#": "Neutrophils (absolute)",
    "NEU%": "NEU %",
    "Non-HDL Kolesterol": "Non-HDL Cholesterol",
    "PCT": "Plateletcrit (PCT)",
    "PDW": "PDW",
    "PLT": "Platelets",
    "Potasyum\n(serum/plazma)": "Potassium",
    "Protein (serum/plazma)": "Total Protein",
    "RBC": "RBC",
    "RDW": "RDW",
    "Serbest T3": "Free T3",
    "Serbest T4": "Free T4",
    "Sodyum\n(serum/plazma)": "Sodium",
    "TSH": "TSH",
    "Total Kolesterol": "Total Cholesterol",
    "Trigliserid": "Triglycerides",
    "VLDL Kolesterol": "VLDL Cholesterol (direct)",
    "Vitamin B12": "Vitamin B12",
    "WBC": "WBC",
    "eGFR": "eGFR",
    "Çinko (Zn), Serum": "Serum Zinc",
    "Üre (serum/plazma)": "Serum Urea",
    "Ürik Asit\n(serum/plazma)": "Uric Acid (Serum)",
}


df = pd.read_csv("Consolidated_Results.csv")
df['Test'] = df['Test'].map(mapping)
df = df.dropna(subset=['Test'])
df = df.reset_index(drop=True)
df.to_csv("test_results_renamed.csv", index=False)

df_results = pd.read_csv("test_results_renamed.csv")
df_normal = pd.read_csv("biomarker_thresholds.csv")
merged_df = pd.merge(df_results, df_normal, left_on='Test', right_on='Test Name', how='inner')

# Creating a file containing all relevant biomarker thresholds consistent with the patient's test results
final_df = merged_df[['Test', 'Unit', 'Reference Range', 'Normal Range', 'Normal Change', 'Sharp Change']]
final_df = final_df.rename(columns={'Test': 'test name'})
final_df.to_csv("merged_test_results.csv", index=False)


# Patient Data--------------------------------------------------------------------------------------------------------------

import pandas as pd

df = pd.read_csv("new_data.csv") # A manually edited (added synthetic data/removed non-numeric results) version of the test_results_renamed file

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df_resampled = df.resample("30D").mean()

df_interpolated = df_resampled.interpolate(method="linear")
df_filled = df_interpolated.ffill().bfill()

# Yeni dosyayı oku
df_filled.rename(columns=mapping, inplace=True)
df_filled = df_filled[[col for col in df_filled.columns if col in mapping.values()]]
df_filled = df_filled.dropna(how='all', subset=[col for col in df_filled.columns if col != 'Date'])

df_filled.reset_index(inplace=True)
df_filled.to_csv("interpolated_filled_output.csv", index=False)

print(df_filled)
