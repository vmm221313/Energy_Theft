class FilePaths:
    def __init__(self):
        
        self.raw_data = 'data/raw/EnergyConsumption_Theft.csv' # Original file
        
        # Impute
        self.imputation_raw = 'data/processed/imputation/imp_raw.csv' # Processed data ready for imputation
        self.edtwbi_imputed = 'data/processed/imputation/edtwbi.pkl' # eDTWBI imputation done

fp = FilePaths()