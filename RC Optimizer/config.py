# =====================================================
# FILE CONFIGURATION
# =====================================================

INPUT_FILE = "input/Agc 23 Call Price.xlsx"
OUTPUT_FILE = "output/Optimized_Result.xlsx"

# =====================================================
# COLUMN NAMES
# =====================================================

PART_COLUMN = "P/N"
CALL_COLUMN = "TotCall"
PRICE_COLUMN = "DN Price"

# =====================================================
# INITIAL CALL THRESHOLDS
#
# C0 = 0
# C1 = 1
# LA = 2-3
# A  = 4 - A_MAX
# B  = A_MAX+1 - B_MAX
# C  = B_MAX+1 - C_MAX
# D  = >C_MAX
# =====================================================

A_MAX = 8
B_MAX = 20
C_MAX = 40

# =====================================================
# INITIAL PRICE THRESHOLDS
#
# P1 : 0 - PRICE1_MAX
# P2 : PRICE1_MAX+1 - PRICE2_MAX
# P3 : PRICE2_MAX+1 - PRICE3_MAX
# P4 : >PRICE3_MAX
# =====================================================

PRICE1_MAX = 100
PRICE2_MAX = 300
PRICE3_MAX = 1000

# =====================================================
# CATEGORY ORDER
#
# IMPORTANT:
# Don't change the order.
# Pivot tables depend on this.
# =====================================================

CALL_ORDER = [
    "C0",
    "C1",
    "LA",
    "A",
    "B",
    "C",
    "D"
]

PRICE_ORDER = [
    "low",
    "medium",
    "high",
    "very high"
]

# =====================================================
# OPTIMIZATION
# =====================================================

OPTIMIZE = False
ITERATIONS = 5000

# =====================================================
# SCORE WEIGHTS
# =====================================================

WEIGHT_COUNT = 0.40
WEIGHT_CALLS = 0.40
WEIGHT_BALANCE = 0.10
WEIGHT_EMPTY = 0.10