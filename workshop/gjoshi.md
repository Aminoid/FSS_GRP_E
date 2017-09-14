## This is python code for workshop on Sept 12.

[1]: https://notebooks.azure.com/n/NiJVoSAq4CU/notebooks/E.ipynb

## Notebook
Notebook for this workshop can be found [here][1]

## TODO
* Increasing k significantly increased the number of wrongly predicted classes while reducing it, increased the sharpness of border of the clusters.
* We should choose a k that causes the maximum number of elements to exist in a single cluster while achieving maximum accuracy by correctly assigning those points to correct clusters. Cross-validation is one of the best way to calculate k.
* There are four different kernels - "linear", "polynomial", "rbf" and "sigmoid". There is also a provision for user to define his/her own kernel
* We should choose the kernel on the type of dataset. If data is linearly separable then linear kernel will cut it, otherwise we need to look into other kernels. Cross-validation using various kernels can give us an idea as to which kernel will perform the best.
