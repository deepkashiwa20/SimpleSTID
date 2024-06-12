# SimpleSTID

This is a STID implementation to train over a large sparse dataset.

## Status_20240612
### Done
- Basic "train" script
- Data Processing/Adapter

### Todo
- Model Saving 
- Result Plot
- LossFunction API


## Training Cmd

```
python3 traintest.py
```

## Result
| Step3 | MAE   | RMSE  | MAPE  |
| ----- | ----- | ----- | ----- |
| STID  | 2.566 | 4.906 | 6.67% |

| Step6 | MAE   | RMSE  | MAPE  |
| ----- | ----- | ----- | ----- |
| STID  | 2.810 | 5.622 | 7.69% |

| Step12 | MAE   | RMSE  | MAPE  |
| ------ | ----- | ----- | ----- |
| STID   | 3.213 | 6.489 | 9.23% |
