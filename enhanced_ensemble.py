import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class BaseModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.3)
        
        self.in_channels = 16
        self.layer1 = self._make_layer(16, 32, 2)
        self.dropout2 = nn.Dropout(0.4)
        
        self.layer2 = self._make_layer(32, 64, 2)
        self.dropout3 = nn.Dropout(0.4)
        
        self.layer3 = self._make_layer(64, 128, 2)
        self.dropout4 = nn.Dropout(0.5)
        
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(EnhancedBlock(self.in_channels, out_channels, stride=2))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(EnhancedBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.layer1(x)
        x = self.dropout2(x)
        
        x = self.layer2(x)
        x = self.dropout3(x)
        
        x = self.layer3(x)
        x = self.dropout4(x)
        
        att = self.attention(x)
        x = x * att
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.fc(x)

class EnhancedEnsemble(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model1 = BaseModel(num_classes)
        self.model2 = BaseModel(num_classes)
        self.model3 = BaseModel(num_classes)
        
        # Specialized attention for different body parts
        self.part_attention = nn.ModuleDict({
            'SHOULDER': nn.Sequential(
                nn.Linear(num_classes * 3, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Softmax(dim=1)
            ),
            'ELBOW': nn.Sequential(
                nn.Linear(num_classes * 3, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Softmax(dim=1)
            ),
        })
        
        # Initialize weights
        self.default_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x, body_part):
        # Get predictions from each model
        pred1 = self.model1(x)
        pred2 = self.model2(x)
        pred3 = self.model3(x)
        
        # Stack predictions
        preds = torch.stack([pred1, pred2, pred3], dim=1)  # [batch_size, 3, num_classes]
        
        # Get body part specific weights if available
        if body_part in self.part_attention:
            # Concatenate predictions for attention
            combined = torch.cat([pred1, pred2, pred3], dim=1)
            weights = self.part_attention[body_part](combined)
            weighted_preds = preds * weights.unsqueeze(-1)
        else:
            # Use default weights for other body parts
            weighted_preds = preds * F.softmax(self.default_weights, dim=0).unsqueeze(0).unsqueeze(-1)
        
        # Sum the weighted predictions
        return torch.sum(weighted_preds, dim=1)