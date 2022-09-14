## 1.environment setting
```conda
conda create -n number python=3.7
conda activate number
pip install -r ./requirements
```

## 2.train
```shell
python ./train.py
```

## 3.eval
```shell
python ./eval
```

modify the line in eval.py(line 18):

```python
weight_file = 'raw_numDet/weights/weight_17_0.008.npz'
```
to choose your weights

the weight is automatically saved during train in train.py(line96~97):
```python
weights_name = '.\weights\weight_' + str(epoch) + '_' + str(round(loss,3))
np.savez(weights_name, filter=conv.filters, weights=softmax.weights, biases=softmax.biases)
```
