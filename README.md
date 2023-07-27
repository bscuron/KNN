# MNIST Digit Recognition

## Demo

![demo](./demo.gif)

## Quickstart

```sh
$ unzip data.zip                     # MNIST data
$ pip3 install -r ./requirements.txt # 3rd-party dependencies
$ python3 main.py
```

## Statistics

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
|            0 |      0.96 |   1.00 |     0.98 |     980 |
|            1 |      0.95 |   1.00 |     0.98 |    1135 |
|            2 |      0.98 |   0.96 |     0.97 |    1032 |
|            3 |      0.96 |   0.97 |     0.97 |    1010 |
|            4 |      0.97 |   0.97 |     0.97 |     982 |
|            5 |      0.96 |   0.97 |     0.96 |     892 |
|            6 |      0.98 |   0.98 |     0.98 |     958 |
|            7 |      0.95 |   0.96 |     0.96 |    1028 |
|            8 |      0.99 |   0.93 |     0.96 |     974 |
|            9 |      0.97 |   0.95 |     0.96 |    1009 |
|     accuracy |           |        |     0.97 |   10000 |
|    macro avg |      0.97 |   0.97 |     0.97 |   10000 |
| weighted avg |      0.97 |   0.97 |     0.97 |   10000 |
