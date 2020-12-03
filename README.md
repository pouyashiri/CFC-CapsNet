# CFC-CapsNet
This is the code for Convolutional Fully-Connected Capsule Network (CFC-CapsNet) paper submitted to DASIP 2021 workshop.
CFC-CapsNet introduces the CFC layer and integrates it on top of the feature extractor of CapsNet to translate the low-level features into capsules.<br />
The base code for this project is the following: <br />
https://github.com/gram-ai/capsule-networks


# Installation
Step 1. Install PyTorch and Torchvision:<br />
`conda install -c pytorch pytorch torchvision`<br />
Step 2. Install Torchnet: <br />
`pip install torchnet`

# Usage
The "Main.py" file trains the network and prints the results to the files in the specified folder (input args). <br />
Parameters:<br />
`--dset`: Choice of dataset (options: MNIST, F-MNIST, SVHN and CIFAR-10)<br />
`--nc`: Number of classes in the chosen dataset<br />
`--w` : The width/height of input images<br />
`--bsize`: Batch size<br />
`--ne`: Number of epochs to train the model<br />
`--niter`: Number of iterations for DR algorithm<br />
`--fck`: Fully-Connected Kernel size (K parameter of the CFC layer)<br />
`--fdim`: The output dimensionality (D parameter of the CFC layer)<br />
`--ich`: number of channels in the input image<br />
`--dec_type`: The type of decoder used (options: FC, DECONV)<br />
`--res_folder`: The output folder to print the results into<br />
`--aug`: Whether or not use a little augmentation to the dataset (options: 0,1)<br />
`--nc_recon`: Performing the reconstruction in a single channel or all channels (options: 1,3)<br />
`--hard`: Perform hard-training at the end or not (hard-training: training while tightening the bounds of the margin loss, options: 0,1)
`--checkpoint`: The file address of the checkpoint file (used for hard training)