import pandas as pd
import numpy as np
import os

# --- Core DC4F Algorithm Components ---

def get_universe_of_discourse(data):
    """
    Determines the universe of discourse (U) from the time series data.
    The paper defines U as [D_min - D_1, D_max + D_2], where D_1 and D_2
    are positive numbers. For simplicity, we'll add a 10% buffer.

    Args:
        data (pd.Series): The input time series data.

    Returns:
        tuple: A tuple containing the minimum and maximum values of the universe.
    """
    d_min = data.min()
    d_max = data.max()
    buffer = (d_max - d_min) * 0.10  # 10% buffer
    return (d_min - buffer, d_max + buffer)

def dynamic_cut(data):
    """
    Calculates the dynamic cut points based on the statistical properties of the data.
    These points are used to partition the universe of discourse into fuzzy sets.
    The paper defines 7 partitions based on mean and standard deviation.

    Args:
        data (pd.Series): The input time series data.

    Returns:
        list: A list of cut points for partitioning the data.
    """
    mean = data.mean()
    std_dev = data.std()

    # Define the cut points based on the formulas in the paper (Section 2.2)
    # A_1 = (-inf, mean - 2*std_dev]
    # A_2 = (mean - 2*std_dev, mean - std_dev]
    # A_3 = (mean - std_dev, mean - 0.5*std_dev]
    # A_4 = (mean - 0.5*std_dev, mean + 0.5*std_dev]
    # A_5 = (mean + 0.5*std_dev, mean + std_dev]
    # A_6 = (mean + std_dev, mean + 2*std_dev]
    # A_7 = (mean + 2*std_dev, +inf)
    
    cut_points = [
        mean - 2 * std_dev,
        mean - std_dev,
        mean - 0.5 * std_dev,
        mean + 0.5 * std_dev,
        mean + std_dev,
        mean + 2 * std_dev,
    ]
    
    return cut_points

def get_fuzzy_sets(cut_points, universe):
    """
    Defines the fuzzy sets based on the cut points.
    Each fuzzy set is represented by a triangular membership function.

    Args:
        cut_points (list): The list of points partitioning the universe.
        universe (tuple): The min and max of the universe of discourse.

    Returns:
        dict: A dictionary where keys are fuzzy set names (e.g., 'A1') and
              values are the parameters [a, b, c] of their triangular
              membership function.
    """
    u_min, u_max = universe
    
    # Add universe boundaries to the cut points for easier processing
    extended_cuts = [u_min] + cut_points + [u_max]
    
    fuzzy_sets = {}
    # According to the paper, we have 7 fuzzy sets
    for i in range(7):
        # Triangular membership function parameters [a, b, c]
        # a = left foot, b = peak, c = right foot
        a = extended_cuts[i-1] if i > 0 else u_min
        b = extended_cuts[i]
        c = extended_cuts[i+1] if i < 6 else u_max

        # For the first and last sets, we create trapezoidal functions
        # by setting the peak equal to the foot at the boundary.
        if i == 0:
            fuzzy_sets[f'A{i+1}'] = [u_min, u_min, extended_cuts[1]]
        elif i == 6:
             fuzzy_sets[f'A{i+1}'] = [extended_cuts[5], u_max, u_max]
        else:
             fuzzy_sets[f'A{i+1}'] = [extended_cuts[i-1], extended_cuts[i], extended_cuts[i+1]]

    # Let's adjust the centers for better representation
    fuzzy_sets['A1'][1] = (fuzzy_sets['A1'][0] + fuzzy_sets['A1'][2]) / 2
    fuzzy_sets['A2'][1] = (fuzzy_sets['A2'][0] + fuzzy_sets['A2'][2]) / 2
    fuzzy_sets['A3'][1] = (fuzzy_sets['A3'][0] + fuzzy_sets['A3'][2]) / 2
    fuzzy_sets['A4'][1] = (fuzzy_sets['A4'][0] + fuzzy_sets['A4'][2]) / 2
    fuzzy_sets['A5'][1] = (fuzzy_sets['A5'][0] + fuzzy_sets['A5'][2]) / 2
    fuzzy_sets['A6'][1] = (fuzzy_sets['A6'][0] + fuzzy_sets['A6'][2]) / 2
    fuzzy_sets['A7'][1] = (fuzzy_sets['A7'][0] + fuzzy_sets['A7'][2]) / 2


    return fuzzy_sets


def fuzzify(value, fuzzy_sets):
    """
    Fuzzifies a crisp value, finding the fuzzy set it belongs to with the highest membership degree.

    Args:
        value (float): The crisp input value.
        fuzzy_sets (dict): The dictionary of defined fuzzy sets.

    Returns:
        str: The name of the fuzzy set with the highest membership (e.g., 'A3').
    """
    memberships = {}
    for name, params in fuzzy_sets.items():
        a, b, c = params
        # Triangular/Trapezoidal membership function calculation
        if a == b: # Left trapezoid
             membership = max(0, min(1, (c - value) / (c - a))) if value <= c else 0
        elif b == c: # Right trapezoid
             membership = max(0, min(1, (value - a) / (c - a))) if value >= a else 0
        else: # Triangle
            membership = max(0, min((value - a) / (b - a), (c - value) / (c - b)))
        memberships[name] = membership

    # Return the set with the maximum membership value
    return max(memberships, key=memberships.get)

def establish_fuzzy_rule_base(fuzzified_data):
    """
    Establishes the fuzzy logical relationships (FLR) or rules from the fuzzified data.
    The rule base is a dictionary where 'LHS -> RHS'.

    Args:
        fuzzified_data (list): A list of fuzzy set names representing the time series.

    Returns:
        dict: A dictionary where keys are the LHS (current state) and values are lists of RHS (next states).
    """
    rule_base = {}
    for i in range(len(fuzzified_data) - 1):
        lhs = fuzzified_data[i]
        rhs = fuzzified_data[i+1]
        if lhs not in rule_base:
            rule_base[lhs] = []
        rule_base[lhs].append(rhs)
    return rule_base

def forecast(current_fuzzified_value, rule_base, fuzzy_sets):
    """
    Performs the forecasting using the established rule base.

    Args:
        current_fuzzified_value (str): The fuzzy representation of the current data point.
        rule_base (dict): The fuzzy rule base.
        fuzzy_sets (dict): The dictionary of defined fuzzy sets.

    Returns:
        float: The defuzzified, crisp forecast value. Returns None if no rule exists.
    """
    if current_fuzzified_value not in rule_base:
        print(f"Warning: No rule found for LHS '{current_fuzzified_value}'. Cannot forecast.")
        return None

    # Get all possible outcomes (RHS) for the current state (LHS)
    possible_outcomes = rule_base[current_fuzzified_value]
    
    # The paper uses the centers of the fuzzy sets for defuzzification.
    # We will average the centers of all possible outcomes.
    forecast_values = []
    for outcome in possible_outcomes:
        # The center of the triangular membership function is the 'b' parameter
        center = fuzzy_sets[outcome][1]
        forecast_values.append(center)
        
    # Defuzzify by taking the mean of the centers of the resulting fuzzy sets
    return np.mean(forecast_values)


def predict_trend(forecasted_value, last_actual_value):
    """
    Determines the forecasted trend based on Section 3 of the DC4F paper.

    Args:
    forecasted_value (float): The value predicted by the model.
    last_actual_value (float): The last known data point in the training series.

    Returns:
    str: A string indicating the trend ('Upward', 'Downward', or 'Unchanged').
    """
    if forecasted_value > last_actual_value:
        return "Upward"
    elif forecasted_value < last_actual_value:
        return "Downward"
    else:
        return "Unchanged"

def dc4f_pipeline(file_path, column_name):
    """
    Runs the complete DC4F pipeline from reading data to forecasting.

    Args:
        file_path (str): The path to the input CSV file.
        column_name (str): The name of the column containing the time series data.

    Returns:
        tuple: A tuple containing the forecast value and the actual next value for comparison.
               Returns (None, None) if forecasting is not possible.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in the CSV file.")
            return None, None
        time_series = df[column_name]
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return None, None

    # For this example, we use all but the last data point for training
    # and the last data point for forecasting.
    training_data = time_series.iloc[:-1]
    last_known_value = training_data.iloc[-1]
    actual_next_value = time_series.iloc[-1]

    # 2. Get Universe of Discourse
    universe = get_universe_of_discourse(training_data)
    print(f"Universe of Discourse (U): {np.round(universe, 2)}")

    # 3. Perform Dynamic Cut
    cut_points = dynamic_cut(training_data)
    print(f"Dynamic Cut Points: {np.round(cut_points, 2)}")

    # 4. Define Fuzzy Sets
    fuzzy_sets = get_fuzzy_sets(cut_points, universe)
    print("Fuzzy Sets (Triangular/Trapezoidal params [a, b, c]):")
    for name, params in fuzzy_sets.items():
        print(f"  {name}: {np.round(params, 2)}")

    # 5. Fuzzify the entire training dataset
    fuzzified_training_data = [fuzzify(value, fuzzy_sets) for value in training_data]
    print(f"\nFuzzified Training Data (first 10): {fuzzified_training_data[:10]}")

    # 6. Establish Fuzzy Rule Base
    rule_base = establish_fuzzy_rule_base(fuzzified_training_data)
    print("\nEstablished Fuzzy Rule Base (LHS -> [RHS]):")
    for lhs, rhs_list in rule_base.items():
        print(f"  {lhs} -> {list(set(rhs_list))}") # Use set to show unique outcomes

    # 7. Forecast the next value
    # Fuzzify the last known value to get the current state
    current_fuzzified_value = fuzzify(last_known_value, fuzzy_sets)
    print(f"\nLast known value: {last_known_value:.2f} -> Fuzzified as: {current_fuzzified_value}")
    
    # Perform the forecast
    predicted_value = forecast(current_fuzzified_value, rule_base, fuzzy_sets)

    if predicted_value is not None:
        print(f"\n---> The Current Value Regarding DCF4 Spline: {predicted_value:.2f}")
        print(f"---> The Real Current Value:    {actual_next_value:.2f}")
        # Return last_known_value as well
        return predicted_value, actual_next_value, last_known_value
    else:
        return None, None, None


def detect_abnormality(forecasted_value, actual_value, training_data, threshold_multiplier=2.0):
    """
    Detects an abnormality if the forecast error is unusually large.
    This is NOT part of the paper but is a common-sense implementation.
    """
    error = abs(forecasted_value - actual_value)
    # Threshold is defined as a multiple of the training data's standard deviation
    threshold = training_data.std() * threshold_multiplier

    if error > threshold:
        return f"Abnormal Event ❗ (Error {error:.2f} > Threshold {threshold:.2f})"
    else:
        return f"Normal Event ✔️ (Error {error:.2f} <= Threshold {threshold:.2f})"


# --- MODIFIED: Pipeline Function ---
def dc4f_pipeline(file_path, column_name):
    """
    Runs the complete DC4F pipeline from reading data to forecasting.
    Now returns the training data for abnormality detection.
    """
    try:
        df = pd.read_csv(file_path)
        time_series = df[column_name]
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

    training_data = time_series.iloc[:-1]
    last_known_value = training_data.iloc[-1]
    actual_next_value = time_series.iloc[-1]

    # ... (rest of pipeline is the same)
    universe = get_universe_of_discourse(training_data)
    cut_points = dynamic_cut(training_data)
    fuzzy_sets = get_fuzzy_sets(cut_points, universe)
    fuzzified_training_data = [fuzzify(value, fuzzy_sets) for value in training_data]
    rule_base = establish_fuzzy_rule_base(fuzzified_training_data)
    current_fuzzified_value = fuzzify(last_known_value, fuzzy_sets)
    predicted_value = forecast(current_fuzzified_value, rule_base, fuzzy_sets)

    if predicted_value is not None:
        print(f"\n--- Results ---")
        print(f"Last known value:         {last_known_value:.2f} (Fuzzified as: {current_fuzzified_value})")
        print(f"Forecasted Next Value:    {predicted_value:.2f}")
        print(f"Actual Next Value:        {actual_next_value:.2f}")
        # Return all necessary values for analysis
        return predicted_value, actual_next_value, last_known_value, training_data
    else:
        return None, None, None, None


# --- MODIFIED: Main Execution Block ---
if __name__ == "__main__":
    print("--- DC4F Forecasting Demo with Trend and Abnormality Detection ---")

    # The user should specify their file and column name here.
    csv_file = 'sample_time_series.csv'
    value_column = 'Value'
    
    if not os.path.exists(csv_file):
        print(f"\nError: The file '{csv_file}' was not found.")
        print("Please create the file or change the 'csv_file' variable in the script.")
    else:
        # The pipeline now returns four values
        forecasted, actual, last_known, training_data = dc4f_pipeline(
            file_path=csv_file,
            column_name=value_column
        )

        if forecasted is not None and actual is not None:
            # --- Analysis Block ---
            trend = predict_trend(forecasted, last_known)
            abnormality_status = detect_abnormality(forecasted, actual, training_data)
            
            print(f"Forecasted Trend:         {trend}")
            print(f"Event Status:             {abnormality_status}")

            error = abs(forecasted - actual)
            mape = (error / actual) * 100 if actual != 0 else float('inf')
            print(f"\nAbsolute Error: {error:.2f}")
            print(f"MAPE: {mape:.2f}%")

