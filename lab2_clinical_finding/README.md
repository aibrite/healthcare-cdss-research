# 🧬 Clinical Laboratory Data Analysis & Biomarker Intelligence System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![PDFPlumber](https://img.shields.io/badge/pdfplumber-0.9+-orange.svg)](https://github.com/jsvine/pdfplumber)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-red.svg)](https://numpy.org/)

> **Transform raw laboratory reports into intelligent clinical insights with advanced biomarker analysis and trend detection**

A sophisticated clinical data processing system that converts PDF laboratory reports into structured data, performs comprehensive biomarker analysis, and provides intelligent clinical insights for healthcare decision support. The system features robust trend analysis, critical value detection, and automated clinical ratio calculations.

## 🌟 Key Features

### 📄 **PDF Laboratory Report Processing**
- **Multi-format PDF extraction** with intelligent table detection
- **Turkish-English medical terminology mapping** for international compatibility
- **Automated data normalization** and reference range parsing
- **Temporal data consolidation** across multiple report dates

### 🔬 **Advanced Biomarker Analysis**
- **Comprehensive biomarker metadata database** with 99+ clinical parameters
- **Sex-specific reference ranges** for accurate clinical interpretation
- **Intelligent trend classification** using Theil-Sen robust slope estimation
- **Critical value detection** with configurable alert thresholds
- **Rate-of-change analysis** with clinical significance assessment

### 📊 **Clinical Intelligence Features**
- **Automated clinical ratio calculations** (BUN/Creatinine, Albumin/Globulin, Total/HDL)
- **Temporal context analysis** (Current vs. Historical trends)
- **Multi-unit conversion system** for standardized comparisons
- **Sharp change detection** with percentage and absolute thresholds
- **Measurement status classification** (Low/Normal/High/Critical)

## 🏗️ System Architecture

```
lab2_clinical_finding/
├── 📄 PDF Reports/
│   ├── 03.03.2025.pdf          # Latest laboratory report
│   ├── 26.08.2024.pdf          # Historical report #1
│   └── 03.12.2024.pdf          # Historical report #2
├── 🔧 Processing Tools/
│   ├── PDF2Excel.py            # PDF extraction & Excel conversion
│   └── biomarker_analysis.py   # Advanced clinical analysis engine
├── 📊 Structured Data/
│   ├── Results_Single.csv      # Single timepoint results
│   ├── Results_Multiple.csv    # Multi-timepoint longitudinal data
│   └── Biomarker_Data.csv      # Clinical metadata & reference ranges
└── 📈 Output Files/
    └── Consolidated_Results.xlsx # Formatted Excel output
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy pdfplumber xlsxwriter
```

### 1. PDF to Excel Conversion

Convert laboratory PDF reports to structured Excel format:

```bash
cd lab2_clinical_finding
python PDF2Excel.py
```

**Output**: `Consolidated_Results.xlsx` with:
- Tests as rows, dates as columns
- Automatic unit and reference range mapping
- Formatted, readable layout with auto-fitted columns

### 2. Advanced Biomarker Analysis

Perform comprehensive clinical analysis:

```bash
python biomarker_analysis.py --single Results_Single.csv \
                            --multiple Results_Multiple.csv \
                            --biomarker Biomarker_Data.csv \
                            --sex Female \
                            --output analysis_results.json
```

**Parameters**:
- `--single`: Single timepoint results file
- `--multiple`: Multi-timepoint longitudinal data
- `--biomarker`: Clinical metadata database
- `--sex`: Patient sex for sex-specific ranges (Male/Female)
- `--output`: JSON output file path

## 📋 Data Formats

### Input Files

#### PDF Laboratory Reports
- **Format**: Standard laboratory PDF reports
- **Languages**: Turkish and English terminology supported
- **Content**: Test results, reference ranges, dates, units

#### Biomarker Metadata (`Biomarker_Data.csv`)
```csv
Test Name,sex,normal_low,normal_high,normal_unit,normal_comp,
normal_change_low,normal_change_high,normal_change_unit,
normal_change_time_days,normal_change_per_day_low,
normal_change_per_day_high,sharp_change_parsed,
sharp_change_aggregated_low,sharp_change_aggregated_high
```

### Output Structure

#### Analysis Results (JSON)
```json
{
  "ALT": {
    "test_name": "ALT",
    "values": [70, 53, 72],
    "dates": ["2024-03-12", "2024-08-26", "2025-03-03"],
    "unit": "U/L",
    "temporal_context": "Last 12 Months",
    "trend": "Fluctuating",
    "measurement_status": "High",
    "critical_status": null,
    "rate_of_change": 0.0067,
    "rate_status": "Normal"
  }
}
```

## 🔬 Clinical Analysis Features

### Trend Classification Algorithm

The system uses the **Theil-Sen estimator** for robust trend analysis:

- **Up**: Consistent increasing pattern
- **Down**: Consistent decreasing pattern  
- **Stable**: Minimal variation (<5% relative change)
- **Fluctuating**: Variable pattern with significant changes
- **No Trend**: Insufficient data points

### Critical Value Detection

Multi-layered approach for clinical alerts:

1. **Absolute Thresholds**: Values exceeding critical limits
2. **Rate-of-Change Rules**: Rapid changes indicating clinical concern
3. **Percentage Changes**: Relative changes beyond normal variation
4. **Persistent Abnormalities**: Sustained out-of-range values

### Unit Conversion System

Automatic conversion between common laboratory units:
- `mg/dL ↔ g/dL ↔ mg/L ↔ µg/dL`
- `mmol/L ↔ mg/dL` (glucose, creatinine)
- `/µL ↔ x10^9/L ↔ x10^12/L` (cell counts)
- `µg/L ↔ µg/dL` (trace elements)

## 🧪 Supported Biomarkers

### Complete Blood Count (CBC)
- WBC, RBC, Hemoglobin, Hematocrit
- Platelets, MCV, MCH, MCHC, RDW
- Neutrophils, Lymphocytes, Monocytes, Eosinophils, Basophils

### Liver Function Tests
- ALT, AST, ALP, GGT
- Total Bilirubin, Direct Bilirubin
- Albumin, Total Protein

### Lipid Panel
- Total Cholesterol, HDL-C, LDL-C, VLDL
- Triglycerides, Non-HDL Cholesterol
- Calculated ratios (Total/HDL)

### Renal Function
- Creatinine, BUN, eGFR
- Uric Acid, Electrolytes (Na, K, Cl)

### Endocrine Markers
- TSH, Free T3, Free T4
- Glucose, HbA1c

### Minerals & Vitamins
- Iron, TIBC, Ferritin
- Vitamin B12, Folate
- Calcium, Phosphate, Magnesium
- Copper, Zinc

## 🔧 Advanced Configuration

### Turkish-English Mapping

The system includes comprehensive medical terminology mapping:

```python
TURKISH_TO_ENGLISH = {
    "Alanin Aminotransferaz (alt)": "ALT",
    "Kolesterol (serum/plazma)": "Total Cholesterol",
    "Kreatinin (serum/plazma)": "Creatinine",
    # ... 150+ mappings
}
```

### Custom Reference Ranges

Sex-specific and age-specific ranges supported:

```csv
Test Name,sex,normal_low,normal_high,normal_unit
Hemoglobin,Female,11.5,15.5,g/dL
Hemoglobin,Male,13,17,g/dL
```

## 📊 Clinical Applications

### 1. **Longitudinal Monitoring**
- Track biomarker trends over time
- Identify gradual changes requiring intervention
- Monitor treatment response

### 2. **Critical Value Alerts**
- Immediate flagging of dangerous values
- Rate-of-change based early warnings
- Configurable alert thresholds

### 3. **Clinical Decision Support**
- Automated ratio calculations
- Reference range comparisons
- Trend-based recommendations

### 4. **Quality Assurance**
- Data validation and normalization
- Unit standardization
- Missing value handling

## 🛠️ Technical Implementation

### PDF Processing Engine
- **pdfplumber**: Robust table extraction
- **Configurable strategies**: Lines, text, explicit
- **Multi-page support**: Automatic concatenation
- **Error handling**: Graceful failure recovery

### Data Processing Pipeline
- **Pandas-based**: Efficient data manipulation
- **Type safety**: Comprehensive validation
- **Memory efficient**: Streaming processing for large datasets
- **Extensible**: Modular design for easy enhancement

### Statistical Methods
- **Theil-Sen estimator**: Robust slope calculation
- **Outlier detection**: Statistical anomaly identification
- **Time series analysis**: Temporal pattern recognition
- **Clinical correlation**: Medical knowledge integration

## 🔍 Example Use Cases

### Case 1: Liver Function Monitoring
```python
# Monitor ALT trends in hepatitis patient
results = process_results("Results_Single.csv", 
                         "Results_Multiple.csv", 
                         "Biomarker_Data.csv", 
                         sex="Male")

alt_data = results["ALT"]
print(f"ALT Trend: {alt_data['trend']}")
print(f"Current Status: {alt_data['measurement_status']}")
```

### Case 2: Diabetes Management
```python
# Track glucose and HbA1c patterns
glucose_trend = results["Fasting Glucose"]["trend"]
hba1c_status = results["HbA1c"]["measurement_status"]
```

## 🚨 Clinical Alerts & Warnings

The system generates intelligent alerts based on:

- **Critical absolute values**: Life-threatening levels
- **Rapid changes**: >25% change in 24 hours
- **Persistent abnormalities**: Sustained out-of-range values
- **Clinical correlations**: Multi-parameter patterns

## 📈 Future Enhancements

- [ ] **Machine Learning Integration**: Predictive modeling
- [ ] **FHIR Compatibility**: Healthcare interoperability
- [ ] **Real-time Processing**: Live data streams
- [ ] **Mobile Interface**: Point-of-care access
- [ ] **Multi-language Support**: Additional medical terminologies

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style standards
- Testing requirements
- Documentation updates
- Feature requests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Clinical expertise**: Medical professionals who provided domain knowledge
- **Open source libraries**: pandas, numpy, pdfplumber communities
- **Healthcare standards**: HL7 FHIR, LOINC terminology systems

---

*Built with ❤️ for advancing healthcare through intelligent data analysis*
