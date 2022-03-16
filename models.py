import torch
import torchvision.models as models
import torchvision.transforms as tf

class ResNet(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(ResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.start = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.slice1 = resnet.layer1
        self.slice2 = resnet.layer2
        self.slice3 = resnet.layer3
        self.slice4 = resnet.layer4
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.start(X)
        h = self.slice1(h)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]
    
class RMACLayer(torch.nn.Module):
    def __init__(self):
        super(RMACLayer, self).__init__()

    def forward(self, encs, regions=None):
        """
        Forward pass
        :param encs: list of encodings
        :param regions: list of dict (each region-index corresponds to enc with that index) with format:
            {
                "x": int,
                "y": int,
                "w": int,
                "h": int
            }
        :return: max value per activation map in the whole spatial domain or in given region
        """

        max_outputs = []
        for i, act_map in enumerate(encs):
            if regions is not None:
                x = regions[i]["x"]
                y = regions[i]["y"]
                w = regions[i]["w"]
                h = regions[i]["h"]
                act_map = act_map[:, :, y:y+h, x:x+w] # TODO h/w in correct order?

            spatial_flattened_act_map = torch.reshape(act_map, (act_map.shape[0], act_map.shape[1], -1))
            # spatial_flattened_act_map = act_map.view(act_map.shape[0], act_map.shape[1], -1)

            spatial_max_act_map = spatial_flattened_act_map.max(dim=2)
            max_outputs.append(spatial_max_act_map.values)

        return max_outputs

class BoundingBoxEncoder(torch.nn.Module):
    def __init__(self, requires_grad=True, resize_shape=None, use_rmac_layer=False):
        super(BoundingBoxEncoder, self).__init__()
        self.model = ResNet(requires_grad=requires_grad)

        if resize_shape is not None:
            self.crop_transform = tf.Compose([
                tf.Lambda(lambda x: x.cpu() if torch.cuda.is_available() else x),
                tf.ToPILImage(),
                tf.Resize(resize_shape),
                tf.ToTensor(),
                tf.Lambda(lambda x: x.cuda() if torch.cuda.is_available() else x)
            ])
        else:
            self.crop_transform = None
        
        self.use_rmac_layer = use_rmac_layer
        self.rmac = RMACLayer()

    def forward(self, batch):
        # crop each image by its bounding-box
        # TODO: can I vectorize it?
        # Problem: if no crop_transform then images have not same size and cannot be cat into one tensor
        images = ()
        for i in range(batch["image"].shape[0]):
            # load bbox from batch
            x = batch["bbox"]["x"][i]
            y = batch["bbox"]["y"][i]
            w = batch["bbox"]["w"][i]
            h = batch["bbox"]["h"][i]

            # crop image along each channel
            img = batch["image"][i, :, y:y+h, x:x+w]

            # apply transform if it exists
            if self.crop_transform is not None:
                img = self.crop_transform(batch["image"][i])

            # add batch dimension for cat later
            img = img.unsqueeze(0)

            # add to tuple of images for cat later
            images += (img,)

        if self.crop_transform is not None:
            # cat all images into cropped tensor because all have same size here!
            images = torch.cat(images, dim=0)
            
            # encode
            encodings = self.model(images)

            # apply rmac if it is selected
            if self.use_rmac_layer:
                encodings = self.rmac(encodings)

            return encodings

        else:
            # loop through each image (non-vectorized!) because they have different sizes due to different bboxes
            encodings = []
            for image in images:
                # encode
                enc = self.model(image)

                # apply rmac if it is selected
                if self.use_rmac_layer:
                    enc = self.rmac(enc)

                encodings.append(enc)

            if self.use_rmac_layer:
                # convert list of encodings where each element is a list of output layer encodings into
                # a list of output layer encodings where each element is one tensor with batch_size = number of images (encodings)
                encodings = [torch.cat(tuple([encodings[i][layer] for i in range(len(encodings))]), dim=0) for layer in range(len(encodings[0]))]
            else:
                # we want to return the encodings as a tensor for each layer for the next steps to work more smoothly.
                # if this is a problem later, we need to fix the dataloader method "output_to_triplets" to also take in a list of all image encodings instead of tensors
                raise ValueError("use_rmac_layer must be true when resize_shape is None.")

            return encodings
        
class REObj_Conv_Block(torch.nn.Module):
    def __init__(self):
        super(REObj_Conv_Block, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.max1 = torch.nn.MaxPool2d(kernel_size=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1)
        self.max2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv3 = torch.nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1)
        self.max3 = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.max3(x)

        return x

class REObjEncoder(torch.nn.Module):
    def __init__(self, requires_grad=True):
        super(REObjEncoder, self).__init__()
        self.model = ResNet(requires_grad=requires_grad)
        self.conv_block = REObj_Conv_Block()

    def forward(self, batch):
        images = batch["image"]

        if images.shape[2] != 224 or images.shape[3] != 224:
            raise ValueError("Re-OBJ expects input in shape 224x224, but input shape was", images.shape)

        encodings = self.model(images)[-1] # only use last layer of encoding as this is what the re-obj baseline does
        embeddings = self.conv_block(encodings)

        return [embeddings] # return list of size 1 to have same API as the other encoders that return list of mulitple encodings