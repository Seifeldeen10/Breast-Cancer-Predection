# Data Audit Report for `synthetic.csv`

**Rows:** 5, **Columns:** 4  

## Column Overview
- `id`: int64
- `age`: float64
- `sex`: str
- `outcome`: float64

## Missing Values
- `age`: 1 missing (20.0%)
- `outcome`: 1 missing (20.0%)

## Duplicate Rows
- 1 duplicate rows detected

## Categorical Columns Summary
- `sex`: 2 unique values (sample: F, M)

## Numerical Columns Summary
- `id`: count=5, mean=2.4, std=1.1402, min=1.0, 25%=2.0, 50%=2.0, 75%=3.0, max=4.0
- `age`: count=4, mean=45.75, std=5.6789, min=38.0, 25%=43.25, 50%=47.5, 75%=50.0, max=50.0
- `outcome`: count=4, mean=0.5, std=0.5774, min=0.0, 25%=0.0, 50%=0.5, 75%=1.0, max=1.0

## Candidate Target Distributions (Low‑cardinality columns)
- `id` distribution:
  - 2: 2
  - 1: 1
  - 3: 1
  - 4: 1
- `age` distribution:
  - 50.0: 2
  - 45.0: 1
  - nan: 1
  - 38.0: 1
- `sex` distribution:
  - F: 3
  - M: 2
- `outcome` distribution:
  - 1.0: 2
  - 0.0: 2
  - nan: 1

## Flags for Manual Review
_No immediate concerns detected._

## Provenance & License
- Source: unknown (to be filled after manual inspection)
- License: unknown


## Suitability Assessment
- Risk modeling: **requires further investigation**
- Stage modeling: **requires further investigation**
- Exploratory analysis: **possible**
