# Lab 2: Neural Network Implementation

This project implements a neural network using PyTorch and TorchVision. It includes training and evaluation of a ResNet-18 model on the CIFAR-10 dataset. The project allows customization of hyperparameters such as learning rate, optimizer, and computing device.

---

## **Requirements**
- Python 3.x
- PyTorch
- TorchVision

Install the required libraries using:
```bash
pip install torch torchvision
```

Usage
Run with Default Parameters
bash
Copy
python lab2.py
Run with Custom Parameters
bash
Copy
python lab2.py --lr 0.01 --device cuda --opt adam --c7
Parameters
Parameter	Description	Default Value	Options/Constraints
--lr	Learning rate	0.1	Must be a float < 1
--device	Computing device	cpu	cpu, cuda
--num_workers	Number of I/O workers	2	Must be an integer
--data_path	Dataset download directory	./data	Path to directory
--opt	Optimizer selection	sgd	sgd, nesterov, adam, adagrad, adadelta
--c7	Disable batch normalization in ResNet-18	False	Set to True to disable batch normalization
Project Structure
Copy
lab2/
├── lab2.py               # Main script for training and evaluation
├── plotting.py           # Script for generating figures
├── README.md             # Project documentation
└── data/                 # Directory for storing the dataset (default)
Additional Files
plotting.py: Generates figures for the report, such as training loss and test accuracy plots.

Example Commands
Train the model with a learning rate of 0.01 on a GPU using the Adam optimizer:

bash
Copy
python lab2.py --lr 0.01 --device cuda --opt adam
Train the model without batch normalization:

bash
Copy
python lab2.py --c7
Output
The script will print the training loss and test accuracy for each epoch. Example output:

Copy
Epoch 1, Train Loss: 1.2345, Test Accuracy: 0.5678
Epoch 2, Train Loss: 1.1234, Test Accuracy: 0.6789
...
Notes
Ensure the dataset is downloaded to the correct directory (./data by default).

Experiment with different hyperparameters to observe their impact on performance.

If you encounter GPU memory issues, reduce the batch size or switch to the CPU.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Copy

### **How to Use**:
1. Copy the entire Markdown code above.
2. Open a text editor (e.g., Notepad, VS Code, Sublime Text).
3. Paste the copied content into the editor.
4. Save the file as `README.md`.

This will preserve the formatting exactly as I provided it. Let me know if you need further assistance! 😊
New chat

