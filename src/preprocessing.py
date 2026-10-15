import re
import numpy as np
import pandas as pd


class Preprocess():
    def __init__(self, df, t=0.9) -> None:
        self.df=df
        self.t=t
        
    def do_preprocess(self):
        # First make pure int/float objects to int/float
        self.df = self.df.applymap(self.robust_convert)
        
        for col in self.df.columns:
            if 'id' in col.lower(): self.df[col] = self.df[col].astype(str)
            else:
                numeric_count = self.df[col].apply(lambda x: isinstance(x, (int, float))).sum()
                total_count = len(self.df[col])

                if numeric_count / total_count >= self.t:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce') 
                    self.df[col] = self.df[col].astype(float) 

        # Second make semantic int/float objects to float
        for col in self.df.columns: 
            if self.df[col].dtype == 'object': self.df[col]=self.df[col].apply(self.convert_to_float)
        
        return self.df

    def robust_convert(self, value):
        if isinstance(value, str):
            if re.match(r'^\d+$', value):  
                return np.float64(value)
            elif re.match(r'^\d+(\.\d+)?$', value): 
                return np.float64(value)
        return value
        
    def convert_to_float(self, value):
        try:
            if pd.isna(value) or value in ['NaN', 'nan', 'N/A']:
                return np.nan
            
            # Handle percentages
            if isinstance(value, str) and value.endswith('%'):
                return np.float64(value.strip('%'))/ 100
            
            # Handle currency (USD, GBP, EUR, JPY)
            if isinstance(value, str) and ('$' in value or '£' in value or '€' in value or '¥' in value):
                value = value.replace('$', '').replace('£', '').replace('€', '').replace('¥', '').replace(',', '')
                return np.float64(value)
            
            # Handle special strings for infinities
            if value in ['Infinity', 'inf']:
                return np.float64('inf')
            if value in ['-Infinity', '-inf']:
                return np.float64('-inf')
            
            # Handle dates by converting them to timestamps
            if isinstance(value, str):
                try:
                    return value.timestamp()
                except:
                    return value
            
        except (ValueError, TypeError):
            return value
