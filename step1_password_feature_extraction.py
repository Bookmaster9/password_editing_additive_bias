"""
Output:
    dataset_enriched.csv - Original dataset with added feature columns
"""

import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher

try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print("Warning: python-Levenshtein not installed. Using fallback method for edit distance.")
    print("For better performance, install with: pip install python-Levenshtein")


# ============================================================================
# LEVEL 1: COARSE OPERATIONS
# ============================================================================

def calculate_edit_distance(original, modified):
    """
    Calculate Levenshtein edit distance between two strings.

    Returns:
        int: Minimum number of single-character edits needed
    """
    if HAS_LEVENSHTEIN:
        return Levenshtein.distance(original, modified)
    else:
        # Fallback: Use dynamic programming
        m, n = len(original), len(modified)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if original[i-1] == modified[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]


def calculate_similarity_ratio(original, modified):
    """
    Calculate similarity ratio (0-1) using SequenceMatcher.
    Higher values indicate more structural preservation.

    Returns:
        float: Similarity ratio between 0 (completely different) and 1 (identical)
    """
    return SequenceMatcher(None, original, modified).ratio()


def extract_level1_features(original, modified):
    """
    Extract coarse-level edit statistics.

    Returns:
        dict: Level 1 features including length changes, character counts, and structure preservation
    """
    features = {}

    # Basic length metrics
    features['L1_len_original'] = len(original)
    features['L1_len_modified'] = len(modified)
    features['L1_len_change'] = len(modified) - len(original)
    features['L1_len_change_abs'] = abs(features['L1_len_change'])

    # Edit distance and similarity
    features['L1_edit_distance'] = calculate_edit_distance(original, modified)
    features['L1_similarity_ratio'] = calculate_similarity_ratio(original, modified)

    # Character-level operations (approximation using alignment)
    matcher = SequenceMatcher(None, original, modified)
    operations = matcher.get_opcodes()

    chars_added = 0
    chars_deleted = 0
    chars_substituted = 0
    chars_unchanged = 0

    for op, i1, i2, j1, j2 in operations:
        if op == 'equal':
            chars_unchanged += (i2 - i1)
        elif op == 'delete':
            chars_deleted += (i2 - i1)
        elif op == 'insert':
            chars_added += (j2 - j1)
        elif op == 'replace':
            deleted = i2 - i1
            added = j2 - j1
            substituted = min(deleted, added)
            chars_substituted += substituted
            if deleted > added:
                chars_deleted += (deleted - added)
            elif added > deleted:
                chars_added += (added - deleted)

    features['L1_chars_added'] = chars_added
    features['L1_chars_deleted'] = chars_deleted
    features['L1_chars_substituted'] = chars_substituted
    features['L1_chars_unchanged'] = chars_unchanged

    # Dominant operation type
    total_edits = chars_added + chars_deleted + chars_substituted
    if total_edits == 0:
        features['L1_dominant_operation'] = 'none'
    else:
        if chars_added > chars_deleted and chars_added > chars_substituted:
            features['L1_dominant_operation'] = 'additive'
        elif chars_deleted > chars_added and chars_deleted > chars_substituted:
            features['L1_dominant_operation'] = 'subtractive'
        elif chars_substituted > chars_added and chars_substituted > chars_deleted:
            features['L1_dominant_operation'] = 'substitutive'
        else:
            features['L1_dominant_operation'] = 'mixed'

    return features


# ============================================================================
# LEVEL 2: SEMANTIC OPERATIONS
# ============================================================================

def detect_pattern_in_string(s):
    """
    Detect various patterns in a password string.

    Returns:
        dict: Boolean flags for different pattern types
    """
    patterns = {}

    # Handle edge case: empty or None string
    if not s or s == 'nan':
        return {
            'has_word': False,
            'has_special': False,
            'has_numbers': False,
            'has_repeat': False,
            'has_sequence': False,
            'has_keyboard_walk': False,
            'has_leetspeak': False,
            'num_pattern_types': 0
        }

    # Dictionary words (simple heuristic: 4+ consecutive letters)
    patterns['has_word'] = bool(re.search(r'[a-zA-Z]{4,}', s))

    # Special characters
    patterns['has_special'] = bool(re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]', s))

    # Numbers
    patterns['has_numbers'] = bool(re.search(r'\d', s))

    # Repeated characters (3+ same char)
    patterns['has_repeat'] = bool(re.search(r'(.)\1{2,}', s))

    # Sequences (abc, 123, etc.)
    patterns['has_sequence'] = bool(
        re.search(r'abc|bcd|cde|def|123|234|345|456|567|678|789', s.lower())
    )

    # Keyboard walks (qwerty, asdf, etc.)
    patterns['has_keyboard_walk'] = bool(
        re.search(r'qwert|asdf|zxcv|1234|qaz|wsx', s.lower())
    )

    # Leetspeak (common substitutions)
    has_leet_pattern = bool(re.search(r'[a@][3e][1il!][0o][5s]|[a@][3e]|[0o][5s]', s.lower()))
    has_mixed_alphanum = bool(re.search(r'\d', s)) and bool(re.search(r'[a-zA-Z]', s))
    has_leet_chars = any(c in s for c in ['0', '1', '3', '4', '5', '7', '8'])
    patterns['has_leetspeak'] = has_leet_pattern or (has_mixed_alphanum and has_leet_chars)

    # Count pattern types - ensure all values are boolean (True/False, not None)
    patterns['num_pattern_types'] = sum([
        1 if patterns['has_word'] else 0,
        1 if patterns['has_special'] else 0,
        1 if patterns['has_numbers'] else 0,
        1 if patterns['has_repeat'] else 0,
        1 if patterns['has_sequence'] else 0,
        1 if patterns['has_keyboard_walk'] else 0,
        1 if patterns['has_leetspeak'] else 0
    ])

    return patterns


def extract_level2_features(original, modified, zxcvbn_patterns_orig=None, zxcvbn_patterns_mod=None):
    """
    Extract semantic security-relevant operations.

    Args:
        original: Original password
        modified: Modified password
        zxcvbn_patterns_orig: Pattern types from zxcvbn for original (comma-separated string)
        zxcvbn_patterns_mod: Pattern types from zxcvbn for modified (comma-separated string)

    Returns:
        dict: Level 2 features focusing on security transformations
    """
    features = {}

    # Detect patterns in both passwords
    orig_patterns = detect_pattern_in_string(original)
    mod_patterns = detect_pattern_in_string(modified)

    # SUBTRACTIVE operations (breaking patterns)
    features['L2_broke_word'] = orig_patterns['has_word'] and not mod_patterns['has_word']
    features['L2_broke_repeat'] = orig_patterns['has_repeat'] and not mod_patterns['has_repeat']
    features['L2_broke_sequence'] = orig_patterns['has_sequence'] and not mod_patterns['has_sequence']
    features['L2_broke_keyboard_walk'] = orig_patterns['has_keyboard_walk'] and not mod_patterns['has_keyboard_walk']

    # ADDITIVE operations (adding complexity)
    features['L2_added_special'] = (not orig_patterns['has_special']) and mod_patterns['has_special']
    features['L2_added_numbers'] = (not orig_patterns['has_numbers']) and mod_patterns['has_numbers']
    features['L2_added_leetspeak'] = (not orig_patterns['has_leetspeak']) and mod_patterns['has_leetspeak']
    features['L2_added_word'] = (not orig_patterns['has_word']) and mod_patterns['has_word']

    # Count special chars and numbers
    orig_special_count = len(re.findall(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]', original))
    mod_special_count = len(re.findall(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]', modified))
    orig_number_count = len(re.findall(r'\d', original))
    mod_number_count = len(re.findall(r'\d', modified))

    features['L2_special_char_delta'] = mod_special_count - orig_special_count
    features['L2_number_delta'] = mod_number_count - orig_number_count

    # OTHER operations
    # Case changes
    orig_upper = sum(1 for c in original if c.isupper())
    orig_lower = sum(1 for c in original if c.islower())
    mod_upper = sum(1 for c in modified if c.isupper())
    mod_lower = sum(1 for c in modified if c.islower())

    features['L2_case_changes'] = abs(orig_upper - mod_upper) + abs(orig_lower - mod_lower)

    # Reordering detection (same character set, different order)
    orig_sorted = ''.join(sorted(original.lower()))
    mod_sorted = ''.join(sorted(modified.lower()))
    features['L2_likely_reordered'] = (orig_sorted == mod_sorted) and (original.lower() != modified.lower())

    # Count subtractive vs additive operations
    subtractive_ops = sum([
        features['L2_broke_word'],
        features['L2_broke_repeat'],
        features['L2_broke_sequence'],
        features['L2_broke_keyboard_walk']
    ])

    additive_ops = sum([
        features['L2_added_special'],
        features['L2_added_numbers'],
        features['L2_added_leetspeak'],
        features['L2_added_word']
    ])

    features['L2_subtractive_ops_count'] = subtractive_ops
    features['L2_additive_ops_count'] = additive_ops

    # Determine operation bias
    if subtractive_ops > additive_ops:
        features['L2_operation_bias'] = 'subtractive'
    elif additive_ops > subtractive_ops:
        features['L2_operation_bias'] = 'additive'
    elif additive_ops > 0 and subtractive_ops > 0:
        features['L2_operation_bias'] = 'mixed'
    else:
        features['L2_operation_bias'] = 'neutral'

    return features


# ============================================================================
# LEVEL 3: STRATEGY CLASSIFICATION
# ============================================================================

def classify_strategy(original, modified, level1_features, level2_features):
    """
    Classify the edit into prototypical strategy types using template matching.

    Strategy types:
    - append_suffix: Added characters at end
    - prepend_prefix: Added characters at start
    - insert_middle: Added characters in middle
    - leetspeak_substitution: Character substitutions with numbers/symbols
    - targeted_deletion: Removed specific characters/patterns
    - word_addition: Added new word(s)
    - scramble_reorder: Reordered existing characters
    - complete_rewrite: No significant overlap with original
    - minimal_change: Very small edits

    Returns:
        dict: Strategy classification features
    """
    features = {}

    # Get a list of strategy tags
    strategies = []

    # Check for complete match (no change)
    if original == modified:
        strategies.append('no_change')
        features['L3_primary_strategy'] = 'no_change'
        features['L3_strategy_count'] = 1
        features['L3_strategies'] = 'no_change'
        return features

    # Check for complete rewrite (very low similarity)
    if level1_features['L1_similarity_ratio'] < 0.3:
        strategies.append('complete_rewrite')

    # Check for append suffix
    if modified.startswith(original) and len(modified) > len(original):
        strategies.append('append_suffix')

    # Check for prepend prefix
    if modified.endswith(original) and len(modified) > len(original):
        strategies.append('prepend_prefix')

    # Check for insert middle (original is preserved but split)
    if len(modified) > len(original):
        # Try to find original as subsequence
        orig_idx = 0
        found_chars = 0
        for char in modified:
            if orig_idx < len(original) and char == original[orig_idx]:
                orig_idx += 1
                found_chars += 1

        if found_chars == len(original) and 'append_suffix' not in strategies and 'prepend_prefix' not in strategies:
            strategies.append('insert_middle')

    # Check for leetspeak substitution
    if level2_features['L2_added_leetspeak'] or (
        level1_features['L1_chars_substituted'] >= 2 and
        level1_features['L1_similarity_ratio'] > 0.6 and
        level2_features['L2_number_delta'] > 0
    ):
        strategies.append('leetspeak_substitution')

    # Check for targeted deletion
    if (level1_features['L1_dominant_operation'] == 'subtractive' and
        level1_features['L1_chars_deleted'] >= 2):
        strategies.append('targeted_deletion')

    # Check for word addition
    if level2_features['L2_added_word'] and level1_features['L1_len_change'] >= 3:
        strategies.append('word_addition')

    # Check for scramble/reorder
    if level2_features['L2_likely_reordered']:
        strategies.append('scramble_reorder')

    # Check for minimal change (1-3 character edits, high similarity)
    if (level1_features['L1_edit_distance'] <= 3 and
        level1_features['L1_similarity_ratio'] > 0.7 and
        len(strategies) == 0):
        strategies.append('minimal_change')

    # Check for case change strategy
    if level2_features['L2_case_changes'] >= 2 and level1_features['L1_edit_distance'] <= 5:
        strategies.append('case_modification')

    # Check for mixed strategy (multiple operations)
    if (level1_features['L1_chars_added'] > 0 and
        level1_features['L1_chars_deleted'] > 0 and
        level1_features['L1_chars_substituted'] > 0):
        strategies.append('hybrid_operations')

    # If no strategy identified, label as "other"
    if len(strategies) == 0:
        strategies.append('other')

    # Set features
    features['L3_primary_strategy'] = strategies[0]
    features['L3_strategy_count'] = len(strategies)
    features['L3_strategies'] = ','.join(strategies)

    # Individual strategy flags
    features['L3_is_append'] = 'append_suffix' in strategies
    features['L3_is_prepend'] = 'prepend_prefix' in strategies
    features['L3_is_insert'] = 'insert_middle' in strategies
    features['L3_is_leetspeak'] = 'leetspeak_substitution' in strategies
    features['L3_is_deletion'] = 'targeted_deletion' in strategies
    features['L3_is_word_addition'] = 'word_addition' in strategies
    features['L3_is_reorder'] = 'scramble_reorder' in strategies
    features['L3_is_rewrite'] = 'complete_rewrite' in strategies
    features['L3_is_minimal'] = 'minimal_change' in strategies
    features['L3_is_hybrid'] = 'hybrid_operations' in strategies

    return features


# ============================================================================
# SECURITY AND MEMORY COST METRICS
# ============================================================================

def calculate_security_metrics(row):
    """
    Calculate security deltas using zxcvbn data.
    Assumes original password zxcvbn values are computed separately.

    Returns:
        dict: Security-related metrics
    """
    features = {}

    # We have zxcvbn data for the modified password
    # For delta calculations, we'd need zxcvbn for original too
    # For now, we'll use the modified password's metrics as absolute values

    features['security_score'] = row.get('zxcvbn_score', np.nan)
    features['security_guesses_log10'] = row.get('zxcvbn_guesses_log10', np.nan)
    features['security_num_patterns'] = row.get('zxcvbn_num_patterns', np.nan)

    return features


def calculate_memory_cost(original, modified, level1_features):
    """
    Estimate memorability cost based on edit complexity.

    Simple heuristic:
    - More edits = harder to remember
    - Longer passwords = harder to remember
    - Lower similarity = harder to remember

    Returns:
        dict: Memory cost estimates
    """
    features = {}

    # Normalized edit distance (0-1 scale)
    max_len = max(len(original), len(modified))
    if max_len > 0:
        features['memory_edit_complexity'] = level1_features['L1_edit_distance'] / max_len
    else:
        features['memory_edit_complexity'] = 0

    # Inverse similarity (0-1, higher = harder)
    features['memory_dissimilarity'] = 1 - level1_features['L1_similarity_ratio']

    # Length penalty
    features['memory_length_penalty'] = min(len(modified) / 20.0, 1.0)  # Cap at 20 chars

    # Combined memory cost (weighted average)
    features['memory_cost_estimate'] = (
        0.4 * features['memory_edit_complexity'] +
        0.4 * features['memory_dissimilarity'] +
        0.2 * features['memory_length_penalty']
    )

    return features


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def extract_all_features(row):
    """
    Extract all hierarchical features for a single password edit.

    Args:
        row: pandas Series containing at minimum 'Original' and 'password' columns

    Returns:
        dict: All extracted features
    """
    original = str(row['Original'])
    modified = str(row['password'])

    all_features = {}

    # Level 1: Coarse operations
    level1 = extract_level1_features(original, modified)
    all_features.update(level1)

    # Level 2: Semantic operations
    level2 = extract_level2_features(original, modified)
    all_features.update(level2)

    # Level 3: Strategy classification
    level3 = classify_strategy(original, modified, level1, level2)
    all_features.update(level3)

    # Security metrics
    security = calculate_security_metrics(row)
    all_features.update(security)

    # Memory cost
    memory = calculate_memory_cost(original, modified, level1)
    all_features.update(memory)

    return all_features


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to process the dataset and output enriched CSV.
    """
    print("="*70)
    print("Password Feature Extraction Script")
    print("="*70)

    # Load dataset
    input_file = 'dataset_with_zxcvbn.xlsx'
    output_file = 'dataset_enriched.csv'

    print(f"\nLoading dataset from: {input_file}")
    df = pd.read_excel(input_file)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Extract features for each row
    print("\nExtracting features...")
    print("This may take a minute for 400 rows...\n")

    feature_rows = []
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"  Processing row {idx}/{len(df)}...")

        features = extract_all_features(row)
        feature_rows.append(features)

    # Convert to DataFrame and merge with original
    features_df = pd.DataFrame(feature_rows)
    enriched_df = pd.concat([df, features_df], axis=1)

    # Save to CSV
    print(f"\nSaving enriched dataset to: {output_file}")
    enriched_df.to_csv(output_file, index=False)

    print(f"\n{'='*70}")
    print("Feature Extraction Complete!")
    print(f"{'='*70}")
    print(f"\nOutput file: {output_file}")
    print(f"Total rows: {len(enriched_df)}")
    print(f"Total columns: {len(enriched_df.columns)} (original: {len(df.columns)}, new: {len(features_df.columns)})")

    # Print summary statistics
    print(f"\n{'='*70}")
    print("Feature Summary Statistics")
    print(f"{'='*70}")

    print("\n--- Level 1: Dominant Operations ---")
    print(enriched_df['L1_dominant_operation'].value_counts())

    print("\n--- Level 2: Operation Bias ---")
    print(enriched_df['L2_operation_bias'].value_counts())

    print("\n--- Level 3: Primary Strategies ---")
    print(enriched_df['L3_primary_strategy'].value_counts())

    print("\n--- Strategy Distribution by Task ---")
    for task in sorted(enriched_df['Task'].unique()):
        print(f"\nTask {task}:")
        task_data = enriched_df[enriched_df['Task'] == task]
        print(task_data['L3_primary_strategy'].value_counts().head())

    print("\n--- Level 2: Additive vs Subtractive Operations by Task ---")
    summary = enriched_df.groupby('Task').agg({
        'L2_additive_ops_count': 'mean',
        'L2_subtractive_ops_count': 'mean'
    }).round(2)
    print(summary)

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
