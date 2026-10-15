from itertools import combinations
import pandas as pd

aggfunc_lists = ['count', 'mean', 'sum', 'min', 'max']

def generate_interesting_combinations(pivot_params):
    keys = list(pivot_params.keys())

    combinations_3 = []
    combinations_4 = []

    for index in keys:
        for value, functions in pivot_params.items():
            for func in functions:
                if index != value:  # index and value should not be the same
                    combinations_3.append({"index":index, "column":[], "value": value, "aggfunc":func})

                    for column in keys:
                        if column != index and column != value:  # index, column, and value should all be different
                            combinations_4.append({"index":index, "column":column, "value": value, "aggfunc":func})

    return combinations_3, combinations_4

def generate_canonical_combinations(df, df_columns):
    column_lists = sorted(df_columns)

    all_combinations_3 = [
        {"index": index, "column": [], "value": value, "aggfunc": func}
        for index in column_lists
        for value in column_lists
        if index != value
        for func in aggfunc_lists
        if not pd.api.types.is_object_dtype(df[value]) or func == "count"
    ]

    all_combinations_4 = [
        {"index": index, "column": column, "value": value, "aggfunc": func}
        for (index, column) in combinations(column_lists, 2) 
        for value in column_lists
        for func in aggfunc_lists
        if index != value and column != value
        if not pd.api.types.is_object_dtype(df[value]) or func == "count"
    ]

    all_combinations = all_combinations_3+all_combinations_4, 

    return (all_combinations_3, all_combinations_4)


def generate_canonical_interesting_combinations(df, df_columns, pivot_params):
    column_lists = sorted(df_columns)

    all_combinations_3 = [
        {"index": index, "column": [], "value": value, "aggfunc": func}
        for index in column_lists
        for value in column_lists
        if index != value
        for func in aggfunc_lists
        if not pd.api.types.is_object_dtype(df[value]) or func == "count"
    ]

    all_combinations_4 = [
        {"index": index, "column": column, "value": value, "aggfunc": func}
        for (index, column) in combinations(column_lists, 2) 
        for value in column_lists
        for func in aggfunc_lists
        if index != value and column != value
        if not pd.api.types.is_object_dtype(df[value]) or func == "count"
    ]

    interesting_combinations_3 = [
        comb for comb in all_combinations_3
        if (comb['index'] in pivot_params and 
           comb['value'] in pivot_params and 
           comb['aggfunc'].upper() in pivot_params[comb['value']]
           )
    ]

    interesting_combinations_4 = [
        comb for comb in all_combinations_4 
        if (comb['index'] in pivot_params and 
            comb['column'] in pivot_params and 
            comb['value'] in pivot_params and
            comb['aggfunc'].upper() in pivot_params[comb['value']])
    ]
    
    all_combinations = all_combinations_3+all_combinations_4, 
    interesting_combinations = interesting_combinations_3+interesting_combinations_4

    return all_combinations_3, all_combinations_4, interesting_combinations_3, interesting_combinations_4

def test_canonical_combinations(all_combs):
    (combinations_3, combinations_4) = all_combs
    
    # Check the length of combinations
    assert len(combinations_3) > 0, "No combinations of size 3 were generated."
    assert len(combinations_4) > 0, "No combinations of size 4 were generated."
    
    # Validate combinations of size 3
    for comb in combinations_3:
        assert 'index' in comb and 'value' in comb and 'aggfunc' in comb, "Invalid structure in combinations_3."
        assert isinstance(comb['column'], list), "Column must be a list in combinations_3."
        assert len(comb['column']) == 0, "Column must be empty in combinations_3."
    
        # Validate combinations of size 4
    seen_pairs = set()  # To track (index, column) pairs in alphabetical order
    for comb in combinations_4:
        assert 'index' in comb and 'column' in comb and 'value' in comb and 'aggfunc' in comb, "Invalid structure in combinations_4."
        assert isinstance(comb['column'], str), "Column must be a string in combinations_4."
        
        # Check that index and column are different
        assert comb['index'] != comb['value'], "Index and value must be different in combinations_4."
        assert comb['column'] != comb['value'], "Column and value must be different in combinations_4."
        assert comb['index'] != comb['column'], "Index and column must be different in combinations_4."

        # Check canonical representation for index and column
        index, column, value, agg = comb['index'], comb['column'], comb['value'], comb['aggfunc']
        pair = (index, column)
        reverse_pair = (column, index, value, agg)
        
        # Check the alphabetical order
        assert index < column, f"Non alphabetical combination found: {pair}"

        # Check that the pair (index, column) is in alphabetical order
        assert reverse_pair not in seen_pairs, f"Transposed combination found: {pair} out of {seen_pairs}"
        
        # Add the canonical form to the seen set
        seen_pairs.add(pair)
        
    print("All tests passed!")
    
def test_canonical_interesting_combinations(all_combs):
    combinations_3, combinations_4, valid_combinations_3, valid_combinations_4 = all_combs
    
    # Check the length of combinations
    assert len(combinations_3) > 0, "No combinations of size 3 were generated."
    assert len(combinations_4) > 0, "No combinations of size 4 were generated."
    
    # Validate combinations of size 3
    for comb in combinations_3:
        assert 'index' in comb and 'value' in comb and 'aggfunc' in comb, "Invalid structure in combinations_3."
        assert isinstance(comb['column'], list), "Column must be a list in combinations_3."
        assert len(comb['column']) == 0, "Column must be empty in combinations_3."
    
    # Validate combinations of size 4
    seen_pairs = set()  # To track (index, column) pairs in alphabetical order
    for comb in combinations_4:
        assert 'index' in comb and 'column' in comb and 'value' in comb and 'aggfunc' in comb, "Invalid structure in combinations_4."
        assert isinstance(comb['column'], str), "Column must be a string in combinations_4."
        
        # Check that index and column are different
        assert comb['index'] != comb['value'], "Index and value must be different in combinations_4."
        assert comb['column'] != comb['value'], "Column and value must be different in combinations_4."
        assert comb['index'] != comb['column'], "Index and column must be different in combinations_4."

        # Check canonical representation for index and column
        index, column, value, agg = comb['index'], comb['column'], comb['value'], comb['aggfunc']
        pair = (index, column)
        reverse_pair = (column, index, value, agg)
        
        # Check the alphabetical order
        assert index < column, f"Reverse combination found: {pair}"

        # Check that the pair (index, column) is in alphabetical order
        assert reverse_pair not in seen_pairs, f"Duplicate or reverse combination found: {pair} out of {seen_pairs}"
        
        # Add the canonical form to the seen set
        seen_pairs.add(pair)
    
    # Validate valid combinations
    assert len(valid_combinations_3) > 0, "Interesting combination 3 is empty."
    assert len(valid_combinations_4) > 0, "Interesting combination 4 is empty."
    
    for comb in valid_combinations_3:
        assert comb in combinations_3, "Valid combination not found in combinations_3."
    
    for comb in valid_combinations_4:
        assert comb in combinations_4, "Valid combination not found in combinations_4."
    
    print("All tests passed!")