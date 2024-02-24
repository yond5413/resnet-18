Hello to execute lab2.py you must have the following libraries installed:
    torch and torchvision

To execute the program run python lab2.py with default parameters
    To specify parameters along side given prompt from earlier you will need to pass
    arguments alongside it.
    note -- is a delimiter, here is an example of an execution specifying parameters 
    [python lab2.py --lr 0.1 --device cuda]
    --lr -> learning rate (default 0.1) needs to be a float less than 1
    --device -> device (default cpu) string datatype must be either cpu or cuda 
    or execution will fail
    --num_workers (default 2) must be an integer and provides main process workers 
    to help expedite I/O operations
    --data_path (default ./data) (str) specfies the irectory dataset will be downloaded
    --opt (default sgd) (str) provides option to choose optimizer among SGD, Nesterov, ADAM, Adagrad, adadelta
    --c7 (default False) (bool) runs resnet-18 without batchnormalization when set to True