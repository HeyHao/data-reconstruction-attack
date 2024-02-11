# data-reconstruction-attack
Conduct data reconstruction attack on partially encrypted updates using MaskCrypt. The attack implementation is borrowed from https://github.com/JonasGeiping/invertinggradients.

## How to run
Step 1. Install Pytorch
```
pip3 install torch torchvision
```

Step 2. Run the attack script with CIFAR10
```
 python inversion_mask.py --model_arch ResNet18 --dataset CIFAR10 --mask_policy [gradient | random] --mask_ratio [0.0~1.0]
```
where ```--mask_policy gradient``` means the encryption mask will be selected by MaskCrypt's gradient-guided method, and ```--mask_policy random``` uses random mask. ```--mask_ratio``` can be chosen between 0.0 and 1.0.

For MNIST, we need to change the model architecture and dataset.
```
python inversion_mask.py --model_arch LeNet5 --dataset MNIST --mask_policy gradient --mask_ratio 0.01
```
The ground truth image will be stored by name ```ground_truth_$dataset.png```, and the reconstructed image will be stored by name ```$dataset_$mask_policy_$mask_ratio.png```