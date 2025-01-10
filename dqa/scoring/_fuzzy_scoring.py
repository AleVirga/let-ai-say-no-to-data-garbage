
from pyspark.sql import functions as sf
from pyspark.sql.types import DoubleType
from thefuzz import fuzz


class FuzzyMatching:
    """
    A class for performing fuzzy matching operations on Spark DataFrames using thefuzz library.

    This class provides methods to calculate the fuzzy match score between two strings and generate
    conditions for fuzzy matching of column values in Spark DataFrames.

    Attributes:
    None

    Methods:
    - fuzz_match(str1, str2): Calculate the fuzzy match score between two strings.
    - generate_fuzzy_condition(internal_col, external_col, match_rt): Generate a condition for
      fuzzy matching of column values in Spark DataFrames.
    """

    def __init__(self):
        pass

    def fuzzy_match(self, str1, str2):
        """
        Calculate the fuzzy match score between two strings using thefuzz.WRatio.

        Parameters:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.

        Returns:
        float: The fuzzy match score. If either string is None or empty, returns 0.0.

        Example:
        >>> fuzz_match("New York", "New Yrk")
        96.0
        """
        # Check if either string is None or empty
        if not str1 or not str2:
            return 0.0
        return float(fuzz.WRatio(str1, str2))

    def generate_fuzzy_condition(self, condition_map, match_rt):
        """
        Generate a combined condition for fuzzy matching of column values in a dataframe.

        Parameters:
        condition_map (dict): A dictionary mapping columns in 'dataframe' to a list of columns
                            against which they should be fuzzily matched.
        match_rt (float): The match rate threshold for considering a match successful.

        Returns:
        Column: A condition given the dictionary mapping and the rate
        """
        fuzzy_match_udf = sf.udf(self.fuzzy_match, DoubleType())
        conditions = []
        for internal_col, external_cols in condition_map.items():
            for external_col in external_cols:
                # Calculate fuzzy matching score and compare it with the threshold
                condition = fuzzy_match_udf(internal_col, external_col) >= match_rt
                conditions.append(condition)

        # Combine all conditions using a logical OR to create a final condition
        final_condition = conditions[0]
        for condition in conditions[1:]:
            final_condition = final_condition | condition

        return final_condition
