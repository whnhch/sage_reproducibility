from typing import Tuple, List, Union
import re

class PivotTableFormatter:
    def __init__(self, pivot_table, value, aggfunc, unique_dict):
        self.pt = pivot_table
        self.aggfunc = aggfunc
        self.value = value
        self.unique_dict = unique_dict

    def transform_values_in_categorization(self, attribute: Union[str, List[str]], values: Union[str, float, List[str | float]]):
        """
        Transforms values into bin labels based on unique_dict[attribute].
        - Handles values like '51.0|74.0' by checking each and returning shared or joined bin strings.
        - Leaves values untouched if bins are not defined.
        """
#         print(f"\n[transform_values_in_categorization] attribute: {attribute}")
#         print(f"[Input] values: {values}")

        unique_values = list(self.unique_dict.get(attribute, []))
        is_bin_format = all(isinstance(s, str) and re.match(r'\(([-\d.]+), ([-\d.]+)\]', s) for s in unique_values)

        def get_bin_label(val):
            str_val = str(val)
            if str_val in unique_values:
#                 print(f"  - Direct match for '{str_val}' → {str_val}")
                return str_val

            if is_bin_format:
                try:
                    num_val = float(val)
                    for bin_str in unique_values:
                        match = re.match(r'\(([-\d.]+), ([-\d.]+)\]', bin_str)
                        if match:
                            lower = float(match.group(1))
                            upper = float(match.group(2))
                            if lower < num_val <= upper:
#                                 print(f"  - Numeric match: {num_val} ∈ {bin_str}")
                                return bin_str
                    # print(f"  - No matching bin for {val}")
                    return "0"
                except ValueError:
                    # print(f"  - ValueError parsing {val}")
                    return "0"
            else:
                # print(f"  - Not bin format. Returning as-is: {str_val}")
                return str_val

        def transform_single(value):
            if isinstance(value, str) and '|' in value:
                parts = value.split('|')
                bins = [get_bin_label(p) for p in parts]
#                 print(f"  - Pipe-separated: {value} → bins: {bins}")
                if all(b == bins[0] for b in bins if b is not None):
                    return bins[0] if bins[0] is not None else value
                else:
                    return '|'.join([b if b is not None else 'UNK' for b in bins])
            else:
                bin_label = get_bin_label(value)
                return bin_label if bin_label is not None else value

        if isinstance(values, list):
            output = [transform_single(val) for val in values]
        else:
            output = transform_single(values)

#         print(f"[Output] transformed: {output}")
        return output

 
    
    def format_index_column_labels(self, i=None, j=None, isRow: bool=True, skip_agg=True) -> str:
        """
        Returns a natural-language-like string combining index and column labels.
        Applied on outlier

        Parameters:
            i (int): Row index
            j (int): Column index
            skip_agg (bool): Whether to skip the aggregation label in columns

        Returns:
            str: e.g., "region North, year 2020, and column sales"
        """
        parts = []
        pt = self.pt
        if not isRow: pt = pt.T

        index_names = pt.index.names
        col_names = pt.columns.names
        if not isinstance(index_names, list): index_names = [index_names]
        if not isinstance(col_names, list): col_names = [col_names]
        
        allNoneRow=all(name is None for name in index_names)
        allNoneCol=all(name is None for name in col_names)
        
        if i is not None and not allNoneRow:
            if i < 0 or i >= len(pt.index):
                parts.append(f"[Invalid index {i}]")
            else:                    
                index_values = pt.index[i]
                if not isinstance(index_values, list): index_values = [index_values]
                if not isRow and skip_agg:
                    if len(index_names) >=2: index_names = index_names[1:]
                    if len(index_values) >=2: index_values = index_values[1:]
                   
                if isinstance(index_values, tuple):
                    parts += [f"{n} {v}" for n, v in zip(index_names, index_values)]
                else:
                    parts +=  f"{index_names[0]} {index_values}" if isinstance(index_names, list) else f"{index_names} {index_values}"
        if j is not None and not allNoneCol:
            if j < 0 or j >= len(pt.columns):
                parts.append(f"[Invalid column {j}]")
            else:
                col_values = pt.columns[j]
                if not isinstance(col_values, list): col_values = [col_values]
                if isRow and skip_agg:
                    if len(col_names) >=2:col_names = col_names[1:]
                    if len(col_values) >=2:col_values = col_values[1:]

                if isinstance(col_values, tuple):
                    parts += [f"{n} {v}" for n, v in zip(col_names, col_values)]
                else:
                    parts +=  f"{col_names[0]} {col_values}" if isinstance(col_names, list) else f"{col_names} {col_values}"

        if not parts:
            return ""

        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        else:
            return f"{', '.join(parts[:-1])}, and {parts[-1]}"

    def format_column_label(self, j:int, skip_agg:bool = True) -> str:
        """
        Returns a natural-language-like string column labels.
        Applied on correlation, ratio.
        There is no and here just comma joined

        Args:
            j (int): index of column
            skip_agg (bool, optional): Dataframe name basically returns value attribute name too. Defaults to True to skip it.

        Returns:
            str: e.g., "region North, year 2020"
        """
        names = self.pt.columns.names
        values = self.pt.columns[j]

        if skip_agg:
            names = names[1:]
            values = values[1:]

        if not names:
            return str(values[0]) if values else ""

        if isinstance(values, tuple):
            return " , ".join(f"{n} {v}" for n, v in zip(names, values))
        else:
            return f"{names[0]} {values}"
    
    def extract_index_info(self, i: int, j: int, isRow: bool = True) -> Tuple[str, str]:
        """
        Extracts (index_attr, index_values) for use in classifier features.
        - index_attr: e.g., "Education"
        - index_values: e.g., "Basic|PhD"
        """
        pt = self.pt
        if not isRow:
            pt = pt.T

        names = pt.index.names
        values = [pt.index[i], pt.index[j]]
        
        if not isinstance(names, list):
            names = [names]
#         if not isinstance(values, tuple):
#             values = (values,)

        # Special case: if column pivot used
        if not isRow and len(names) >= 2:
            names = names[1:]
            values = values[1:]

        index_attr = names[0] if len(names) == 1 else "|".join(names)
        index_values = "|".join(str(v) for v in values) # join for the  decision tree.

        return index_attr, index_values

    from typing import Tuple

    def extract_cell_info(self, i: int, j: int, isRow: bool = True) -> Tuple[str, str, str, str]:
        """
        Extracts (index_attr, index_values, column_attr, column_values) for a cell in the pivot table,
        excluding aggregation function info from column_attr.

        Args:
            i (int): Row index in the pivot table.
            j (int): Column index in the pivot table.
            isRow (bool): Whether the pivot is in row-wise mode or transposed (default: True).

        Returns:
            Tuple[str, str, str, str]: 
                - index_attr: e.g., "Education|Gender"
                - index_values: e.g., "Basic|Male"
                - column_attr: e.g., "Marital" (aggfunc excluded)
                - column_values: e.g., "Single|mean"
        """
        pt = self.pt
        if not isRow:
            pt = pt.T

        index_names = pt.index.names
        index_values = pt.index[i]

        if not isinstance(index_names, list):
            index_names = [index_names]
        if not isinstance(index_values, tuple):
            index_values = (index_values,)

        index_attr = "|".join(str(name) for name in index_names if name is not None)
        index_value_str = "|".join(str(val) for val in index_values)

        column_names = pt.columns.names
        column_values = pt.columns[j]

        if not isinstance(column_names, list):
            column_names = [column_names]
        if not isinstance(column_values, tuple):
            column_values = (column_values,)

        # Special case: ignore first column level when transposed
        if not isRow and len(column_names) >= 2:
            column_names = column_names[1:]
            column_values = column_values[1:]

        if len(column_names) > 1:
            column_attr_names = column_names[:-1]
        else:
            column_attr_names = column_names

        column_attr = "|".join(str(name) for name in column_attr_names if name is not None)
        column_value_str = "|".join(str(val) for val in column_values)

        return index_attr, index_value_str, column_attr, column_value_str


    def format_index_label(self, i: int, isRow: bool=True) -> str:
        """
        Returns a natural-language-like string index labels.
        Applied on correlation, ratio.
        There is no and here just comma joined

        Args:
            i (int): index of the labels we want

        Returns:
            str: e.g., "region North, year 2020"
        """
        pt = self.pt
        if not isRow: pt = pt.T
        
        names = pt.index.names
        values = pt.index[i]
        
        if not isinstance(names, list): names = [names]
        if not isinstance(values, list): names = [values]
        
        if not isRow:
            if len(names) >=2: names = names[1:]
            if len(names) >=2: values = values[1:]
                    
        if isinstance(values, tuple):
            return " , ".join(f"{n} {v}" for n, v in zip(names, values))
        else:
            return f"{names[0]} {values}" if isinstance(names, list) else f"{names} {values}"

    def get_column_name(self, isRow=True) -> str:
        """
        Returns the column name(s), excluding the aggregation level.

        Returns:
            str: Comma-separated string of column level names.
        """
        pt = self.pt
        if not isRow:
            pt = pt.T

        names = pt.columns.names

        if not isRow and len(names) >= 2:
            names = names[1:]

        # Filter out None and convert names to string
        cleaned_names = [str(name) for name in names if name is not None]

        return ", ".join(cleaned_names)

    
    def get_value_name(self) -> str:
        """
        Returns the value name.

        Returns:
            str: just a single value attribute name.
        """
        return self.value
        
    def get_correlation_prompt_vars(self, i: int, j: int, isRow=True) -> dict:
        """
        Returns all necessary variables to fill a correlation prompt.

        Args:
            i (int): Index position of group A.
            j (int): Index position of group B.
            isRow (bool): boolean variable if the table is row-wise or not

        Returns:
            dict: {
                "groupA": ...,
                "groupB": ...,
                "column_attributes": ...,
                "aggfunc": ...,
                "value_attribute": ...,
            }
        """
        return {
            "groupA": self.format_index_label(i, isRow=isRow),
            "groupB": self.format_index_label(j, isRow=isRow),
            "column_attributes": self.get_column_name(isRow),
            "aggfunc": self.aggfunc,
            "value_attribute": self.get_value_name(),  
        }
    
    def get_correlation_clf_prompt_vars(self, i: int, j: int, isRow=True) -> dict:
        """
        Returns all necessary variables to fill a correlation prompt.

        Args:
            i (int): Index position of group A.
            j (int): Index position of group B.
            isRow (bool): boolean variable if the table is row-wise or not

        Returns:
            dict: {

            }
        """
        index_attr, index_vals = self.extract_index_info(i, j, isRow=isRow)
        index_vals = self.transform_values_in_categorization(index_attr, index_vals)
        
        return {
            "index_attr": index_attr,
            "index_values": index_vals,
            "column_attr": self.get_column_name(isRow),
            "value_attr": self.get_value_name(),  
            "aggfunc": self.aggfunc.lower(),
            "insight_type": 'corr',
        }
        
    def get_ratio_groups(self, i: int, j: int, isRow: bool) -> dict:
        """
        Returns formatted groupA, groupB, and column name for ratio task.

        Args:
            i (int): Index position of group A.
            j (int): Index position of group B.
            isRow (bool): boolean variable if the table is row-wise or not

        Returns:
            Tuple[str, str, str]: (groupA, groupB, column_attributes)
        """
        return {
            "groupA": self.format_index_label(i, isRow=isRow),
            "groupB": self.format_index_label(j, isRow=isRow),
            "column_attributes": self.get_column_name(isRow),
            "aggfunc": self.aggfunc,
            "value_attribute": self.get_value_name(),  
        }
    
    def get_ratio_clf_prompt_vars(self, i: int, j: int, isRow=True) -> dict:
        """
        Returns all necessary variables to fill a correlation prompt.

        Args:
            i (int): Index position of group A.
            j (int): Index position of group B.
            isRow (bool): boolean variable if the table is row-wise or not

        Returns:
            dict: {

            }
        """
        index_attr, index_vals = self.extract_index_info(i, j, isRow=isRow)
        index_vals = self.transform_values_in_categorization(index_attr, index_vals)
        
        return {
            "index_attr": index_attr,
            "index_values": index_vals,
            "column_attr": self.get_column_name(isRow),
            "value_attr": self.get_value_name(),  
            "aggfunc": self.aggfunc.lower(),
            "insight_type": 'rat',
        }
        
    def get_outlier_groups(self, i: int, j: int, isRow: bool) -> dict:
        """
        Returns formatted group string for outlier task.

        Args:
            i (int): Index position.

        Returns:
            str: e.g., "region North, year 2020"
        """
        return {
            "groups": self.format_index_column_labels(i, j, isRow=isRow),
            "aggfunc": self.aggfunc,
            "value_attribute": self.get_value_name(),  
        }
    
    def get_outlier_clf_prompt_vars(self, i: int, j: int, isRow: bool) -> dict:
        """
        Returns formatted group string for outlier task.

        Args:
            i (int): Index position.

        Returns:
            str: e.g., "region North, year 2020"
        """
        index_attr, index_vals, column_attr, column_values = self.extract_cell_info(i, j, isRow=isRow)
        index_vals = self.transform_values_in_categorization(index_attr, index_vals)
        column_values = self.transform_values_in_categorization(column_attr, column_values)
        
        return {
            "index_attr": index_attr,
            "index_value": index_vals,
            "column_attr": column_attr,
            "column_value": column_values,
            "value_attr": self.get_value_name(),  
            "aggfunc": self.aggfunc.lower(),
        }