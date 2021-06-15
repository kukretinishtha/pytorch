1. What are the two primitives to work with data?
    Dataloader - wraps an iterable around dataset
    Dataset - stores samples and their corresppnding labels

2. What are domain specific libraries inclused in Torch?
    TorchText
    TorchVision
    TorchAudio

3. Which argument is used to modify samples?
    transform

4. Which argument is  used to modify labels?
    target_tansform

6. What are Tensors?
    Tensors are a specialized data structure that are very similar to arrays and matrices. 
    Tensors are similar to Numpy's ndarrays

8. What are the benefits of Tensors?
    Tensors can run on GPU or other hardware accelrators. 
    Tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data.
    Tensors are also optimized for automatic differentiation.

9. How can we initialize tensors?
    1. Directly from data
    2. From numpy array
    3. From another tensor
    4. With random or constant values

10. What are the attributes of a Tensor?
    Tensor attributes describe their shape, data type and the device on which it is stored.
    tensor.shape
    tensor.dtype
    tensor.device

11. What are the operations on Tensors?
    Standard numpy-like indexing and slicing
    Joining tensors
    Arithmetic operations
    Single-element tensors
    In-place operations

12. What is the working of Computational Graphs?
    In a forward pass, autograd does two things simultaneously:
    run the requested operation to compute a resulting tensor
    maintain the operation’s gradient function in the DAG.

    The backward pass kicks off when .backward() is called on the DAG root. autograd then:
    computes the gradients from each .grad_fn,
    accumulates them in the respective tensor’s .grad attribute
    using the chain rule, propagates all the way to the leaf tensors.


13. What are hyperparameter?
    Hyperparameters are adjustable parameters that let you control the model optimization process. Different hyperparameter values can impact model training and convergence rates (read more about hyperparameter tuning)

    Define the following hyperparameters for training:

    Number of Epochs - the number times to iterate over the dataset
    Batch Size - the number of data samples propagated through the network before the parameters are updated
    Learning Rate - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

15. What is loss function?
    Loss function measures the degree of dissimilarity of obtained result to the target value, and it is the loss function that we want to minimize during training. To calculate the loss we make a prediction using the inputs of our given data sample and compare it against the true data label value.

16. What is optimizer?
    Optimization is the process of adjusting model parameters to reduce model error in each training step. 

17. Define optimization algorithm?
    Optimization algorithms define how this process is performed. All optimization logic is encapsulated in the optimizer object. Here, we use the SGD optimizer; additionally, there are many different optimizers available in PyTorch such as ADAM and RMSProp, that work better for different kinds of models and data.

18. Explain Optimization steps
    Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
    Backpropagate the prediction loss with a call to loss.backwards(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
    Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.

19. What is stat_dict()?
    PyTorch models store the learned parameters in an internal state dictionary, called state_dict.

20. What is load_state_dict()?
    To load model weights, you need to create an instance of the same model first, and then load the parameters using load_state_dict() method.

21. What can you do with ONNX model?
    ONNX  model can be used to run inference on different platforms and in different programming languages.

22. What is Data Parallelism?
    DataParallel splits your data automatically and sends job orders to multiple models on several GPUs. After each model finishes their job, DataParallel collects and merges the results before returning it to you.