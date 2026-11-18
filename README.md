# DPPH IC50 analysis

A small Python tool that turns raw DPPH absorbance readings into **percent radical scavenging** and an **IC50 with its uncertainty**. DPPH is a standard assay for screening the antioxidant activity of plant extracts and pure compounds, and the IC50 (the concentration that scavenges 50 % of the radical) is the number usually reported.

![Example dose-response fit](docs/example_fit.png)

> **Example data are synthetic.** `data/example_data.csv` was simulated from a Hill curve with a known IC50 of 45 µg/mL plus random noise, so the tests can check the code against a known answer. It is not real laboratory data.

## Background

I wrote this while working on the antioxidant (radical-scavenging) part of my final-year Biochemistry research project at Bingham University, a study of a polyherbal tea. I originally calculated the IC50 values in GraphPad Prism and Excel. This repository re-implements that calculation in Python so it is scripted, repeatable and covered by tests, rather than tied to a spreadsheet or a point-and-click workflow.

## The calculation

For each extract concentration, with replicate absorbances at 517 nm:

1. Subtract the blank (optional) from every reading.
2. Average the control replicates (DPPH with no extract) to get `A_control`.
3. For each sample reading: `% scavenging = (A_control − A_sample) / A_control × 100`.
4. Average the replicates at each concentration (mean ± SD).
5. Fit a Hill (four-parameter logistic) curve

   ```
   response = bottom + (top − bottom) / (1 + (IC50 / x) ^ hill)
   ```

   with `bottom = 0` and `top = 100` fixed by default, which is the textbook definition of the DPPH IC50.
6. Report the IC50, its standard error (from the fit covariance, via the delta method on log10 IC50) and a 95 % confidence interval (t-distribution, `n − parameters` degrees of freedom).

## Quick start

```bash
pip install -r requirements.txt
python dpph_ic50.py data/example_data.csv --plot docs/example_fit.png
```

Output on the example data:

```
    conc (ug/mL)    mean %     SD %
              10     11.51     1.26
              25     31.92     0.51
              50     52.92     0.74
             100     73.88     0.19
             200     87.69     0.93

IC50 = 45.35 +/- 0.40 (SE) ug/mL
95% CI: 44.08 to 46.64 ug/mL
Hill slope = 1.32, R^2 = 0.9998, bottom = 0.0, top = 100.0 (n = 5 points)
```

The simulated truth was 45 µg/mL and the 95 % CI contains it.

Use it from Python:

```python
import dpph_ic50 as dp

readings = dp.load_readings("data/example_data.csv")
result = dp.analyze(readings, blank=0.0)
print(result.fit.ic50, result.fit.ic50_se, result.fit.ic50_ci95)
```

## Input format

A CSV with a header, then one row per concentration: the concentration first, then one column per replicate. **The row with concentration `0` is the DPPH control.** Lines starting with `#` and blank lines are ignored, and rows can be in any order.

```
concentration,rep1,rep2,rep3
0,0.794,0.802,0.785
10,0.712,0.703,0.692
25,0.538,0.545,0.538
...
```

Every concentration needs the same number of replicates.

## Options

| Flag | Meaning |
|---|---|
| `--blank X` | Subtract absorbance `X` (e.g. solvent without DPPH) from every reading. |
| `--fit-on replicates` | Fit every replicate as its own point instead of the replicate means (default `mean`). |
| `--free-top`, `--free-bottom` | Fit the plateaus instead of fixing them at 100 and 0. The reported `ic50` is then the curve's midpoint, which equals 50 % scavenging only if the plateaus are 0 and 100. |
| `--unit U` | Concentration unit used in labels (default `ug/mL`). |
| `--plot FILE.png` | Save the dose-response plot. |

## Edge cases

The tool refuses to extrapolate an IC50 the data cannot support:

- **No concentration reaches 50 %**: reports `IC50 > <highest concentration tested>` instead of a number.
- **Every concentration is already above 50 %**: reports `IC50 < <lowest concentration tested>`.
- **A blank larger than the control** (non-positive control absorbance) and **too few concentrations to fit** raise a clear error.
- A single replicate per concentration works, but there is no SD to report.

## Tests

```bash
pytest
```

The tests build synthetic absorbances from a curve with a known IC50, so the expected answer does not depend on the code under test. They check that:

- the percent-scavenging formula gives exact values (0, 50, 100 %);
- a known IC50 and Hill slope are recovered, noiselessly and with noisy replicates;
- the 95 % confidence interval covers the true IC50 about 95 % of the time over 100 simulated experiments;
- blank correction removes a constant offset (and ignoring it visibly biases the IC50);
- the two edge cases above are reported correctly;
- CSV loading, the example data and the command line work end to end.

## Checking against other software

**GraphPad Prism:** the matching setup is a variable-slope dose-response fit with bottom constrained to 0 and top to 100. Compare against `--fit-on replicates` if Prism treated each replicate as a separate point, and against the default (means) otherwise. Small differences are normal because of different optimisers and weighting.

**Excel:** Excel has no built-in Hill fit, so an IC50 taken from a trendline or from interpolating between the two points either side of 50 % will not match a curve fit exactly. Expect it to agree most closely when the data are close to a straight line around 50 %.

## Files

```
dpph_ic50.py            calculation, CSV loader, plot and command line
tests/test_dpph_ic50.py tests
data/example_data.csv   synthetic example readings
docs/example_fit.png    plot produced from the example data
requirements.txt        numpy, scipy, matplotlib, pytest
```
