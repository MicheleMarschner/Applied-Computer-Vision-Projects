import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbedderMaxPool(nn.Module):
    """
    Convolutional encoder that down-samples via MaxPool2d and outputs a flat feature vector.

    This is used as a shared building block for the early and intermediate
    fusion architectures.
    """
    def __init__(self, in_ch, feature_dim=128):
        """
        Args:
            in_ch (int): Number of input channels.
            feature_dim (int): Number of output channels in the last conv layer.
        """
        kernel_size = 3
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(64, feature_dim, kernel_size, padding=1)
        self.pool = nn.MaxPool2d(2)

        # For 64x64 input and 3 pooling steps we end up at 8x8 spatial size.
        self.flatten_dim = feature_dim * 8 * 8


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Flattened feature tensor of shape (B, flatten_dim).
        """
        x = self.pool(F.relu(self.conv1(x)))    # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv2(x)))    # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        return x
    

class EmbedderStrided(nn.Module):
    """
    Embedder where all spatial downsampling is done via stride-2 convolutions.
    """
    def __init__(self, in_ch, feature_dim=128):
        """
        Args:
            in_ch (int): Number of input channels.
            feature_dim (int): Number of output channels in the last conv layer.
        """
        kernel_size = 3
        stride_size = 2
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size, stride=stride_size, padding=1)           # 64 -> 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size, stride=stride_size, padding=1)              # 32 -> 16
        self.conv3 = nn.Conv2d(64, feature_dim, kernel_size, stride=stride_size, padding=1)     # 16 -> 8

        # For 64x64 input and 3 conv with stride 2 we end up at 8x8 spatial size.
        self.flatten_dim = feature_dim * 8 * 8


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Flattened feature tensor of shape (B, flatten_dim).
        """
        x = F.relu(self.conv1(x))           
        x = F.relu(self.conv2(x))          
        x = F.relu(self.conv3(x))          
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        return x


class FullyConnectedHead(nn.Module):
    """
    Fully-connected classification head mapping features to class logits.
    """
    def __init__(self, input_dim, output_dim=2):
        """
        Args:
            input_dim (int): Dimensionality of the flattened feature vector.
            output_dim (int): Number of output classes.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, output_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (B, input_dim).

        Returns:
            torch.Tensor: Class logits of shape (B, output_dim).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class EarlyFusionModel(nn.Module):
    """
    Early fusion model that concatenates RGB and LiDAR channels at the input.

    The 8-channel tensor (4 RGB-like + 4 XYZA) is passed through a shared
    CNN embedder and a fully-connected classification head.
    """
    def __init__(self, in_ch=8, output_dim=2, embedder_cls=EmbedderMaxPool):
        """
        Args:
            in_ch (int): Number of input channels after concatenation.
            output_dim (int): Number of output classes.
        """
        super().__init__()

        # Shared embedder for all channels
        self.embedder = embedder_cls(in_ch)

        # Fully-connected head on top of the shared embedding
        self.fullyConnected = FullyConnectedHead(
            input_dim=self.embedder.flatten_dim,
            output_dim=output_dim
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_ch, 64, 64).

        Returns:
            torch.Tensor: Class logits of shape (B, output_dim).
        """
        features = self.embedder(x)     # → (B, 12800)
        preds = self.fullyConnected(features)  # → (B, output_dim)
        return preds  


class ConcatIntermediateNet(nn.Module):
    """
    Intermediate fusion model using feature concatenation.

    Two separate EmbedderMaxPool encoders are applied to RGB and XYZA
    inputs, their flattened features are concatenated, and a shared
    FullyConnectedHead maps the joint representation to class logits.
    """
    def __init__(self, rgb_ch, xyz_ch, output_dim, feature_dim=128, embedder_cls=EmbedderMaxPool):
        super().__init__()

        # Independent Encoders
        # RGB learns textures/colors
        self.rgb_encoder = embedder_cls(in_ch=rgb_ch, feature_dim=feature_dim)    
        # LiDAR learns geometry/depth
        self.xyz_encoder = embedder_cls(in_ch=xyz_ch, feature_dim=feature_dim)   

        # Calculate combined dimension
        # (200 * 8 * 8) + (200 * 8 * 8)
        combined_dim = self.rgb_encoder.flatten_dim + self.xyz_encoder.flatten_dim

        # Shared FullyConnected Head
        self.head = FullyConnectedHead(input_dim=combined_dim, output_dim=output_dim)

    def forward(self, x_rgb, x_xyz):
        # Extract features independently
        x_rgb = self.rgb_encoder(x_rgb)                                 # (B, D)
        x_xyz = self.xyz_encoder(x_xyz)                                 # (B, D)

        # Fuse (Concatenate) at the feature level
        x_fused = torch.cat((x_rgb, x_xyz), dim=1)                      # (B, 2*D)

        # Predict
        output = self.head(x_fused)

        return output


class AddIntermediateNet(nn.Module):
    """
    Intermediate fusion model using element-wise addition.

    Two separate encoders process each modality independently.
    The resulting feature vectors must have the same size; they are
    added element-wise and passed to a shared FullyConnectedHead.
    """
    def __init__(self, rgb_ch, xyz_ch, output_dim, feature_dim=128, embedder_cls=EmbedderMaxPool):
        super().__init__()

        # Independent Encoders
        # RGB learns textures/colors
        self.rgb_encoder = embedder_cls(in_ch=rgb_ch, feature_dim=feature_dim)    
        # LiDAR learns geometry/depth
        self.xyz_encoder = embedder_cls(in_ch=xyz_ch, feature_dim=feature_dim)    

        # For addition, shapes must match
        fused_dim = self.rgb_encoder.flatten_dim                        # same size after addition

        # Shared FullyConnected Head
        self.head = FullyConnectedHead(input_dim=fused_dim, output_dim=output_dim)

    def forward(self, x_rgb, x_xyz):
        # Extract features independently
        x_rgb = self.rgb_encoder(x_rgb)                                 # (B, D)
        x_xyz = self.xyz_encoder(x_xyz)                                 # (B, D)

        # Additive fusion in feature space
        x_fused = x_rgb + x_xyz                                         # (B, D)

        # Predict
        output = self.head(x_fused)                                     # (B, output_dim)

        return output   


class MatmulIntermediateNet(nn.Module):
    """
    Intermediate fusion model using matrix multiplication.

    The two modality-specific embeddings are reshaped into matrices
    and combined via a bilinear interaction (matrix product) before
    the fully-connected head.
    """
    def __init__(self, rgb_ch, xyz_ch, output_dim, feature_dim, embedder_cls=EmbedderMaxPool):
        super().__init__()

        # Independent Encoders
        # RGB learns textures/colors
        self.rgb_encoder = embedder_cls(in_ch=rgb_ch, feature_dim=feature_dim)    
        # LiDAR learns geometry/depth
        self.xyz_encoder = embedder_cls(in_ch=xyz_ch, feature_dim=feature_dim)    

        # For multiplication, shapes must match
        #embedding_dim = self.rgb_encoder.flatten_dim
        #fused_dim = embedding_dim * embedding_dim                       # D * D after matmul
        self.feature_dim = feature_dim
        self.spatial_dim = 8
        fused_dim = self.feature_dim * self.spatial_dim * self.spatial_dim


        # Shared FullyConnected Head
        self.head = FullyConnectedHead(input_dim=fused_dim, output_dim=output_dim)

    def forward(self, x_rgb, x_xyz):
        B = x_rgb.size(0)

        # Extract features independently
        x_rgb = self.rgb_encoder(x_rgb)                                 # (B, D)
        x_xyz = self.xyz_encoder(x_xyz)                                 # (B, D)

        C = self.feature_dim
        H = self.spatial_dim
        x_rgb = x_rgb.view(B, C, H, H)   # (B, C, 8, 8)
        x_xyz = x_xyz.view(B, C, H, H)   # (B, C, 8, 8)

        # Per-channel spatial matrix multiplication 
        x_fused = torch.matmul(x_rgb, x_xyz)                            # (B, C, 8, 8)
        x_fused = x_fused.flatten(1)                                # (B, C*8*8)

        # Predict
        output = self.head(x_fused)                                     # (B, output_dim)

        return output


class HadamardIntermediateNet(nn.Module):
    """
    Intermediate fusion model using the Hadamard (element-wise) product.

    After independent encoding, the feature vectors are multiplied
    element-wise to capture multiplicative interactions between
    modalities, then fed to the classification head.
    """
    def __init__(self, rgb_ch, xyz_ch, output_dim, feature_dim, embedder_cls=EmbedderMaxPool):
        super().__init__()

        # Independent Encoders
        # RGB learns textures/colors
        self.rgb_encoder = embedder_cls(in_ch=rgb_ch, feature_dim=feature_dim)    
        # LiDAR learns geometry/depth
        self.xyz_encoder = embedder_cls(in_ch=xyz_ch, feature_dim=feature_dim)    

        # For elementwise multiplication, shapes must match
        fused_dim = self.rgb_encoder.flatten_dim                        # same size after addition

        # Shared FullyConnected Head
        self.head = FullyConnectedHead(input_dim=fused_dim, output_dim=output_dim)

    def forward(self, x_rgb, x_xyz):
        # Extract features independently
        x_rgb = self.rgb_encoder(x_rgb)                                 # (B, D)
        x_xyz = self.xyz_encoder(x_xyz)                                 # (B, D)

        # Multiplicative / gating-like fusion
        x_fused = x_rgb * x_xyz                                         # shape: (B, D)

        # Predict
        output = self.head(x_fused)                                     # (B, output_dim)

        return output


class LateNet(nn.Module):
    """
    Late fusion model combining unimodal logits.

    Two independent classifiers are trained for RGB and LiDAR.
    Their logits are then averaged (or combined) at the decision level
    to obtain the final prediction.
    """
    def __init__(self, rgb_ch, xyz_ch, output_dim, embedder_cls=EmbedderMaxPool):
        super().__init__()
        self.rgb = embedder_cls(rgb_ch)
        self.xyz = embedder_cls(xyz_ch)

        # each embedder outputs flatten_dim (e.g. 12800)
        fusion_dim = self.rgb.flatten_dim * 2  # rgb + xyz

        # single FullyConnected head in which data is fused
        self.fullyConnected = FullyConnectedHead(
            input_dim=fusion_dim,
            output_dim=output_dim,
        )

    def forward(self, x_rgb, x_xyz):
        # Extract features independently
        x_rgb = self.rgb(x_rgb)     # (B, 12800)
        x_xyz = self.xyz(x_xyz)     # (B, 12800)

        # this concatenates the features from the two branches
        x_fused = torch.cat((x_rgb, x_xyz), dim=1)    # (B, 25600)

        # Predict
        preds = self.fullyConnected(x_fused)           # (B, output_dim)
        return preds
    

class CILPBackbone(nn.Module):
    def __init__(self, in_ch, embedder_cls=EmbedderMaxPool, feature_dim=128, emb_size=200):
        super().__init__()

        # Encoder
        self.encoder = embedder_cls(in_ch=in_ch, feature_dim=feature_dim)

        # Projection to CILP embedding space
        self.dense_emb = nn.Sequential(
            nn.Linear(self.encoder.flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, emb_size)
        )

    def forward(self, x):
        feats = self.encoder(x)          # (B, flatten_dim)
        emb = self.dense_emb(feats)      # (B, emb_size)
        return F.normalize(emb, dim=-1)  # unit-norm embeddings


class ContrastivePretraining(nn.Module):
    def __init__(self, img_embedder: nn.Module, lidar_embedder: nn.Module):
        super().__init__()
        self.img_embedder = img_embedder
        self.lidar_embedder = lidar_embedder
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, rgb_imgs, lidar_depths):
        # 1. Encode
        img_emb = self.img_embedder(rgb_imgs)          # (B, D_cilp)
        lidar_emb = self.lidar_embedder(lidar_depths)  # (B, D_cilp)

        # 2. Pairwise cosine similarities, without hardcoding batch_size
        #    shape: (B, 1, D_cilp) and (1, B, D_cilp) → broadcast → (B, B)
        similarity = self.cos(
            img_emb.unsqueeze(1),      # (B, 1, D_cilp)
            lidar_emb.unsqueeze(0),    # (1, B, D_cilp)
        )                              # (B, B)
        # similarity = (similarity + 1) / 2         ## TODO: überlegen ob rein oder nicht

        logits_per_img = similarity          # img → lidar      # (B,B)
        logits_per_lidar = similarity.T      # lidar → img      # (B,B)

        return logits_per_img, logits_per_lidar
    

class Projector(nn.Module):
    """
    Maps RGB embeddings -> LiDAR embedding space.
    Used after CILP encoders are frozen.
    """
    def __init__(self, img_dim, lidar_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, lidar_dim),
        )

    def forward(self, x):
        return self.net(x)
    

class Classifier(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        kernel_size = 3
        n_classes = 1

        self.embedder = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 64 -> 32
            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 32 -> 16
            nn.Conv2d(100, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 16 -> 8
            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 8 -> 4
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
        )

    def get_embs(self, imgs):
        return self.embedder(imgs)     # shape: (B, D_cilp*4*4) = (B, 3200)

    def forward(self, raw_data=None, data_embs=None):
        assert (raw_data is not None or data_embs is not None), "No images or embeddings given."
        if raw_data is not None:
            data_embs = self.get_embs(raw_data)
        return self.classifier(data_embs)           # (B, 1)


class RGB2LiDARClassifier(nn.Module):
    def __init__(self, CILP, projector, lidar_cnn):
        super().__init__()
        self.img_embedder = CILP.img_embedder   # frozen
        self.projector = projector              
        self.shape_classifier = lidar_cnn             # pretrained LiDAR classifier

    def forward(self, imgs):
        with torch.no_grad():
            img_encodings = self.img_embedder(imgs)           # (B, emb_dim)
            proj_lidar_embs = self.projector(img_encodings)   # (B, emb_dim)
        # lidar_cnn should have a forward mode that accepts embeddings
        return self.shape_classifier(data_embs=proj_lidar_embs)

