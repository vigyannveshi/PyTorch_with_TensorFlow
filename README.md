# PyTorch 🤝 TensorFlow
<p align = 'center'>
<img src = 'pytorch_with_tensorflow.png'/>
</p>

The repository explores the co-existence of Pytorch and Tensorflow, the similarities and the differences. After learning both the frameworks, TensorFlow being the recent learn, I seek to get the best of both. I had learnt Pytorch about 1.5 to 2 years (2023 - 2024) back as a begineer to Deep Learning through the course: [`A deep understanding of deep learning (with Python intro)`](https://www.udemy.com/course/deeplearning_x/?kw=deep+unders&src=sac). Brushing up the entire course again would have taken me two plus months again (although very worth it to understand and get a full feel of Deep Learning!). I found [`Practical Deep Learning using Pytorch - CampusX`](https://www.youtube.com/watch?v=QZsguRbcOBM&list=PLKnIA16_Rmvboy8bmDCjwNHgTaYH2puK7) as an alternative. It is a sincere effort, and it is a good choice for syntax revision + in detail understanding of Pytorch as a framework. Some specific topics like `Autograd` caught my attention towards this playlist.

On a sincere note, when I had explored PyTorch for the first time, I had missed on a lot of subtle implementation details eg: inplace operations {zero_()}, detailed understanding of autograd, and it makes a lot of difference in understanding. What I loved about Pytorch was its `class` implementation of models, custom training loops etc. When I recently learnt TensorFlow (with Keras), I somehow found it easier w.r.t understanding the tensor implementation, and overall flow. What I found difficult in TensorFlow is ability to customize using Python, serializing, saving and loading the model, customized-training although I tried doing it while learning TensorFlow. 

**Notes:**
* In TensorFlow, we don't need to manually place the model on the GPU. TensorFlow automatically places the model on the GPU if one is available and visible.
* In TensorFlow, we don't need to manually move the data to GPU. As long as the data is a `tf.Tensor` or `tf.data.Dataset`, TensorFlow will automatically handle the device placement.
* Even though you write your custom training function, you don't need to manually handle GPU placement for:
  * model
  * data
  * or computations

### **Contents:**

| **Concept** | **Notebook** |
|---------|----------|
| 01. Introduction  |[01](01_overview.ipynb)|
| 02. Basics of tensors  |[01](01_overview.ipynb)|
| 03. Creating tensors  |[01](01_overview.ipynb)|
| 04. Tensor shapes  |[01](01_overview.ipynb)|
| 05. Tensor dtypes  |[01](01_overview.ipynb)|
| 06. Mathematical operations  |[01](01_overview.ipynb)|
| 07. Copying a tensor  |[01](01_overview.ipynb)|
| 08. Tensor operations on GPU  |[01](01_overview.ipynb)|
| 09. Reshaping tensors |[01](01_overview.ipynb)|
| 10. Autograd in PyTorch |[02](02_automatic_differentiation.ipynb)|
| 10.01 Computing gradient for scalars |[02](02_automatic_differentiation.ipynb)|
| 10.02 Computing gradient for vectors |[02](02_automatic_differentiation.ipynb)|
| 10.03 Clearing gradients |[02](02_automatic_differentiation.ipynb)|
| 10.04 Disabling gradient tracking |[02](02_automatic_differentiation.ipynb)|
| 11. GradientTape in TensorFlow |[02](02_automatic_differentiation.ipynb)|
| 11.01 Computing gradient for scalars |[02](02_automatic_differentiation.ipynb)|
| 11.02 Computing gradient for vectors |[02](02_automatic_differentiation.ipynb)|
| 12. PyTorch implementation of basic training pipeline |[03](03_training_pipeline_pytorch.ipynb)|
| 13. TensorFlow implementation of basic training pipeline |[04](04_training_pipeline_tensorflow.ipynb)|
| 14. PyTorch's NN module|[05](05_pytorch_nn_module.ipynb)|
| 14.01 Basic neural network creation|[05](05_pytorch_nn_module.ipynb)|
| 14.02 Creating neural network with hidden layers|[05](05_pytorch_nn_module.ipynb)|
| 15. TensorFlow's Keras and Functional modules |[06](06_tensorflow_keras_functional.ipynb)|
| 16. Dataset and DataLoaders in PyTorch |[07](07_dataset_and_dataloader_pytorch.ipynb)|
| 17. `tf.data` API in TensorFlow |[08](08_tf_data_tensorflow.ipynb)|
| 18. Building an ANN with end to end workflow in PyTorch |[09](09_end_to_end_ann_pytorch.ipynb)|
| 19. Making training GPU compatible in PyTorch |[10](10_gpu_training_pytorch.ipynb)|
| 20. Hyperparameter tuning using Optuna |[11](11_hyperparameter_tuning_using_optuna.ipynb)|
| 20.01. Optuna Visualizations |[11](11_hyperparameter_tuning_using_optuna.ipynb)|
| 20.02. Define by run |[11](11_hyperparameter_tuning_using_optuna.ipynb)|
| 21. Hyperparameter tuning in PyTorch using Optuna|[12](12_hyperparameter_tuning_pytorch.ipynb)|
| 22. CNN in PyTorch | [13](13_cnn_pytorch.ipynb)|
| 23. Transfer Learning in PyTorch | [14](14_transfer_learning_pytorch.ipynb)|