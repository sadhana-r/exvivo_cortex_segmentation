import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

# Set up GPU if available    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReconstructionLoss(nn.Module):
    def __init__(self,weight = None):
        super(ReconstructionLoss, self).__init__()
        
        self.weight = weight

    def forward(self, input, target, seg):
        
        idx_include = (seg >= 1) & (seg < 4)
#        idx_include = (seg > 2) | (target > 0)
        target_mask = target[idx_include]
        input_mask = input[idx_include]
        #mean square loss
        loss = torch.sum((input_mask - target_mask)**2)/len(input_mask)
        
        return loss
    
    
class GeneralizedDiceLoss(nn.Module):

    def __init__(self, epsilon=1e-5, weight=None, num_classes=2, ignore_index = None): 
        super(GeneralizedDiceLoss, self).__init__()
        
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        
        input = self.normalization(input)

        def get_one_hot_image(target):
            one_hot = []
            for c in range(0, self.num_classes):
                #print("splitting label", c)
                t_label = (target==c)
                t_label_numpy = t_label.cpu().data.numpy()
                one_hot.append(t_label_numpy)
            return np.asarray(one_hot)

        target_one_hot = [get_one_hot_image(target[x,:,:,:]) for x in range(0,len(target))]
        target_one_hot = np.asarray(target_one_hot)
        target_one_hot = torch.from_numpy(target_one_hot).float().to(device)

        assert input.size() == target_one_hot.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target_one_hot = flatten(target_one_hot)

        #target = target.float()
        target_sum = target_one_hot.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False)

        intersect = (input * target_one_hot).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target_one_hot).sum(-1) * class_weights).sum()
        loss = 1. - 2. * intersect / denominator.clamp(min=self.epsilon)

        return loss
    
class CELoss_deepsupervision(nn.Module):
    
    def __init__(self, weights):
        super(CELoss_deepsupervision, self).__init__()
        
        self.weights = weights
        self.CELoss = torch.nn.CrossEntropyLoss(self.weights)

    def forward(self, input, deep_inputs, target):
        
        alpha = [0.25,0.5,0.75,1]
        total_loss = 0
        for i in range(0,len(deep_inputs)-1):
            loss = self.CELoss(deep_inputs[i], target)
            total_loss += alpha[i]*loss
        # last layer
        total_loss += alpha[3]*self.CELoss(input,target)
            ## Just add weight decay for regularization?

        return total_loss

class DSCLoss_deepsupervision(nn.Module):
    
    def __init__(self, weights, alpha, num_classes):
        super(DSCLoss_deepsupervision, self).__init__()
        
        self.weights = weights
        self.alpha = alpha
        self.GeneralizedDSCLoss = GeneralizedDiceLoss(num_classes=num_classes, weight = weights)
        
    def forward(self, input, deep_inputs, target):
        
        alpha = self.alpha
        total_loss = 0
        for i in range(0,len(deep_inputs)-1):
            loss = self.GeneralizedDSCLoss(deep_inputs[i], target)
            total_loss += alpha[i]*loss
        # last layer
        total_loss += alpha[-1]*self.GeneralizedDSCLoss(input,target)
            ## Just add weight decay for regularization?

        return total_loss