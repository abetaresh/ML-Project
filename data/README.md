# Description of the data

This data set comprises a total of 3000 images divided into 2000 images for
testing and 1000 for testing.  The images are 460 pixel tall and 380 pixel wide
and variously contain landscapes, animals, humans and other objects. 

### Consideration 1: Labels
The specificity of this dataset lies in that these previous categories overlap
in such a way that it is possible to have at the same time a sunny landscape
with many animals and also humans. The labels are therefore not a one-hot
encoding of unary classes but rather a binary representation of weather or not
the nth object appears on a given picture.  What's more same features are in
effect mutually exclusive for instance "rainy" and "sunny", while others can be
added up "baboon", "river".

### Consideration 2: Sparcity of the data set


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
