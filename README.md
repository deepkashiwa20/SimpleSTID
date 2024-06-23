# SimpleSTID

This is a STID implementation to train over a large sparse dataset.

## Status_20240612
### Done
- Basic "train" script
- Data Processing/Adapter
- Model Saving
- Result Plot
- LossFunction API

### Todo(Till 240623)
 


## Training Cmd

```
python3 traintest.py
```

## Result
| Step3 | MAE   | RMSE  | MAPE  |
| ----- | ----- | ----- | ----- |
| STID  | 2.566 | 4.906 | 6.67% |
| AGCRN | 2.567 | 4.838 | 6.51% |
| GWNET | 2.467 | 4.567 | 6.12% |


| Step6 | MAE   | RMSE  | MAPE  |
| ----- | ----- | ----- | ----- |
| STID  | 2.810 | 5.622 | 7.69% |
| AGCRN | 2.795 | 5.483 | 7.48% |
| GWNET | 2.740 | 5.305 | 7.13% |

| Step12 | MAE   | RMSE  | MAPE  |
| ------ | ----- | ----- | ----- |
| STID   | 3.213 | 6.489 | 9.23% |
| AGCRN | 3.116 | 6.381 | 8.84% |
| GWNET  | 3.045 | 6.154 | 8.39% |
