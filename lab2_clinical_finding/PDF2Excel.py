import pdfplumber
import pandas as pd
import os
from os import listdir

def extract_tables_from_pdf(path, v_strat, h_strat):
    try:
        with pdfplumber.open(path) as pdf:
            table_settings = {
                "vertical_strategy": v_strat,
                "horizontal_strategy": h_strat
            }

            df_list = []

            for page in pdf.pages:
                tables = page.find_tables(table_settings)
                for table in tables:
                    # Extract table and convert to DataFrame
                    tb = table.extract()
                    df = pd.DataFrame(tb[1:], columns=tb[0])

                    # Normalize headers
                    header_translation = {
                        'Tarih': 'Date',
                        'Test': 'Test',
                        'Değer': 'Value',
                        'Birim': 'Unit',
                        'Referans Aralığı': 'RefRange',
                        'Date': 'Date',
                        'Test Name': 'Test',
                        'Result': 'Value',
                        'Result Unit': 'Unit',
                        'Reference\nValue': 'RefRange',
                        'Reference Value': 'RefRange',
                        'Value': 'Value',
                        'Unit': 'Unit',
                        'RefRange': 'RefRange'
                    }
                    df.columns = [header_translation.get(col, col) for col in df.columns]

                    df_list.append(df)

            # Concatenate all DataFrames
            result_df = pd.concat(df_list, ignore_index=True)
            
            # Normalize date column
            if 'Date' in result_df.columns:
                # Forward-fill the Date column where value is "-" or blank
                result_df['Date'] = result_df['Date'].replace(['-', ''], pd.NA)
                result_df['Date'] = result_df['Date'].ffill()
                
                # Parse to datetime with errors="ignore" to keep original formats intact if parsing fails
                result_df['Date'] = pd.to_datetime(result_df['Date'], errors='ignore')
            
            return result_df
    except Exception as e:
        print(f"Error processing PDF {path}: {str(e)}")
        return None

def main():
    dir_path = os.getcwd()
    print(f'Processing PDF files in: {dir_path}')
    try:
        any_pdf = pdfs_to_excels(dir_path)
        if any_pdf:
            print(f'✓ Consolidated_Results.xlsx created successfully in {dir_path}')
        else:
            print('No PDF files found in the current directory.')
    except Exception as e:
        print(f'Error: {e}')

def pdfs_to_excels(dir_path):
    try:
        files_full_path = list_files_full_path(dir_path, 'pdf')
        if len(files_full_path) > 0:
            all_data = []
            unit_refrange_lookup = {}
            
            output_path = os.path.join(dir_path, 'Consolidated_Results.xlsx')
            
            # Process each PDF
            for f in files_full_path:
                pdf_df = extract_tables_from_pdf(f, "lines", "lines")
                if pdf_df is not None and not pdf_df.empty:
                    # Collect all data
                    if all(col in pdf_df.columns for col in ['Date', 'Test', 'Value']):
                        all_data.append(pdf_df[['Date', 'Test', 'Value', 'Unit', 'RefRange']])
                    
                    # Build unit and reference range lookup
                    if 'Test' in pdf_df.columns:
                        for _, row in pdf_df.iterrows():
                            test_name = row.get('Test', '')
                            if test_name and test_name not in unit_refrange_lookup:
                                unit_refrange_lookup[test_name] = {
                                    'Unit': row.get('Unit', ''),
                                    'RefRange': row.get('RefRange', '')
                                }
            
            if all_data:
                # Combine all data
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # Pivot to get tests as rows and dates as columns
                pivot_df = combined_df.pivot_table(
                    index='Test', 
                    columns='Date', 
                    values='Value', 
                    aggfunc='first'
                )
                
                # Reset index to make Test a column
                pivot_df = pivot_df.reset_index()
                
                # Sort columns to have Test first, then dates in order
                date_columns = [col for col in pivot_df.columns if col != 'Test']
                date_columns.sort()
                pivot_df = pivot_df[['Test'] + date_columns]
                
                # Format date columns to be more readable
                new_columns = ['Test']
                for col in pivot_df.columns[1:]:
                    if pd.api.types.is_datetime64_any_dtype(pivot_df[col]) or isinstance(col, pd.Timestamp):
                        new_columns.append(col.strftime('%Y-%m-%d %H:%M'))
                    else:
                        new_columns.append(str(col))
                pivot_df.columns = new_columns
                
                # Add Unit and Reference Range columns
                pivot_df['Unit'] = pivot_df['Test'].map(lambda x: unit_refrange_lookup.get(x, {}).get('Unit', ''))
                pivot_df['Reference Range'] = pivot_df['Test'].map(lambda x: unit_refrange_lookup.get(x, {}).get('RefRange', ''))
                
                # Write to Excel
                with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                    pivot_df.to_excel(writer, sheet_name='Results', index=False)
                    
                    # Format the Excel sheet
                    workbook = writer.book
                    worksheet = writer.sheets['Results']
                    
                    # Auto-fit columns
                    for i, col in enumerate(pivot_df.columns):
                        max_len = max(
                            pivot_df[col].astype(str).map(len).max(),
                            len(str(col))
                        ) + 2
                        worksheet.set_column(i, i, max_len)
                
                return True
        else:
            return False
    except Exception as e:
        print(f"Failed to create Excel files in {dir_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    

def list_files(directory, extension):
  return [f for f in listdir(directory) if f.endswith('.' + extension)]

def list_files_full_path(directory, extension):  
    files = list_files(directory, extension)
    return [f'{directory}/{f}' for f in files]

if __name__ == '__main__':
    main()
