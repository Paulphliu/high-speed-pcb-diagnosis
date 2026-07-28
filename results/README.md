# Reported Results

The reported results were obtained from:

- A separately generated 6,000-case training dataset
- An independent PN06 production-like validation dataset
- 30 validation lots
- 10 sampled boards per lot
- 20 test points per board

## Performance Summary

| Level | Task | Accuracy |
|---|---|---:|
| Measurement | Stage 1 overall severity | 99.72% |
| Measurement | Stage 1 impedance status | 99.65% |
| Measurement | Stage 1 loss status | 99.88% |
| Measurement | Stage 2 exact category | 73.42% |
| Measurement | Stage 2 signal group | 80.83% |
| Lot | Stage 3 primary diagnosis | 93.33% |
| Lot | Stage 3 engineering match | 100.00% |

The PN06 validation dataset was generated separately and was not used to fit the Stage 1 or Stage 2 models.

The datasets are not included in this repository because they can be regenerated using the provided source code.
