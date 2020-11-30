# Description of the data

This data set comprises a total of 3000 images divided into 2000 images for
training and 1000 for testing.  The images are 460 pixel tall, 380 pixel wide,
and variously contain landscapes, animals, humans and other objects. 

### Consideration 1: Labels
The specificity of this dataset lies in that these previous categories are not mutually exclusive. It is possible to have at the same time a sunny landscape
with many animals and also a fellow human. The labels are therefore not a one-hot
encoding of unary classes but rather a binary representation of whether or not
the n^th object appears on a given picture.

### Consideration 2: Sparcity of the data set (cardinality)

### Dimension of data and label
* Label: 17 classes
* Image: 480, 320

### Example 1:

```
binary_key = ["landscape", "horse", "house", "baboon", "river", "sunny", "cloudy", "rainy"]
binary_encoding = [1, 1, 0, 0, 1, 0]
```
![photo1](https://raw.githubusercontent.com/abetaresh/ML-Project/main/data/testing/27-27707.jpg)

### Example 2:
```
binary_key = ["landscape", "horse", "house", "baboon", "river", "sunny", "cloudy", "rainy"]
binary_encoding = [1, 0, 1, 0, 0, 0, 1, 0, 0]
```
![photo2](https://raw.githubusercontent.com/abetaresh/ML-Project/main/data/testing/27-27708.jpg)
