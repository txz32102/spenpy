I want to convert the matlab code to python code

matlab code is below:

```m
deltaMat = repmat(Row(delta), NumKs, 1) ;
IdxPosMat = repmat(Row(IdxPositions), NumKs, 1) ;
```

final size of deltaMat is 128 128

torch code is below

```python
deltaMat = delta.repeat(1, NumKs)
IdxPosMat = IdxPositions.repeat(1, NumKs)
```

python repeat is different from matlab, but i want to keep the python code the same as matlab, can you help me change the python code?