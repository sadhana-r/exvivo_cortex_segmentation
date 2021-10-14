def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Colorize:

    def __init__(self, n=4):
        
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def computeGeneralizedDSC_patch(probability, seg):
    
     seg = seg.cpu().numpy()
     probability = probability.cpu().numpy()
     preds = np.argmax(probability, 1)
     
     gt = seg[seg > 0]
     myseg = preds[seg > 0]
     
     gdsc = sum(gt == myseg)/ len(gt)
     
     return gdsc

def computeGeneralizedDSC(gt, seg):
    
     gt_seg = gt[gt > 0]
     myseg = seg[gt > 0]
     
     gdsc = 100*(sum(gt_seg == myseg)/ len(gt_seg))
     
     return gdsc
 
def generate_prediction(output):    
    """
    Generates predictions based on the output of the network
    """    
    #convert output to probabilities
    probability = F.softmax(output, dim = 1)
    _, preds_tensor = torch.max(probability, 1)
    
    return preds_tensor, probability
    
def plot_images_to_tfboard(img, seg, output, step, is_training = True, num_image_to_show = 1):
    
    color_transform = Colorize()
    preds, probability = generate_prediction(output)
    i = 0
    if is_training:
#        for i in range(num_image_to_show):
            writer.add_image('Training/Intensity images/'+str(i), img[i,:,:,:,24], global_step = step)
            writer.add_image('Training/Ground Truth seg/'+ str(i), color_transform(seg[i,None,:,:,24]), global_step = step)
            writer.add_image('Training/Predicted seg/'+ str(i), color_transform(preds[i,None,:,:,24]), global_step = step)
    else:
#        for i in range(num_image_to_show):
            writer.add_image('Validation/Intensity images/'+str(i), img[i,:,:,:,24], global_step = step)
            writer.add_image('Validation/Ground Truth seg/'+ str(i), color_transform(seg[i,None,:,:,24]), global_step = step)
            writer.add_image('Validation/Predicted seg/'+ str(i), color_transform(preds[i,None,:,:,24]), global_step = step)

def center_crop(layer, target_size):
    # only four elements since channels is 1
    _, layer_height, layer_width, layer_depth = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    diff_z = (layer_width - target_size[2]) // 2
    return layer[
        :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1]),  diff_z : (diff_z + target_size[2])
    ]
