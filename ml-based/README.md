## ML-based pronunciation prediction

```
$ python preprocess.py
reading cmudict-0.7b.txt...
read 134430/134430, saved 117176 words

$ python train.py
...

$ mlflow ui
(To check your training results)

$ python deploy.py conf/seq2seq.yaml <path to the trained model produced by mlflow>

> python
パイトウン (P AY T OW N)
> ruby
ルウビイ (R UW B IY)
```