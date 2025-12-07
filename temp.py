import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Linear(channels, channels)
        self.key_conv = nn.Linear(channels, channels)
        self.value_conv = nn.Linear(channels, channels)
        self.softmax = nn.Softmax(dim=-1)  # Apply softmax on the attention scores

    def forward(self, x):
        # x is of shape [batch_size, channels, height, width] = [1024, 5, 20, 20]
        batch_size, channels, height, width = x.size()
        num_patches = height * width  # Flatten spatial dimensions
        # Flatten the height and width dimensions into one (num_patches)
        x = x.view(batch_size, channels, num_patches).permute(0, 2, 1)  # Shape: [batch_size, num_patches, channels]

        # Linear projections for Q, K, V
        Q = self.query_conv(x)  # Shape: [batch_size, num_patches, channels]
        K = self.key_conv(x)    # Shape: [batch_size, num_patches, channels]
        V = self.value_conv(x)  # Shape: [batch_size, num_patches, channels]

        # Scaled dot-product attention
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(channels, dtype=torch.float32))
        attention_probs = self.softmax(attention_scores)  # Shape: [batch_size, num_patches, num_patches]

        # Apply attention weights to the values
        attention_out = torch.bmm(attention_probs, V)  # Shape: [batch_size, num_patches, channels]

        # Reshape attention output back to [batch_size, channels, height, width]
        attention_out = attention_out.permute(0, 2, 1).view(batch_size, channels, height, width)

        return attention_out

x = torch.randn(1024, 5, 20, 20)  # Input feature map of shape [batch_size, channels, height, width]
self_attention_layer = SelfAttention(channels=5, num_heads=1)
output = self_attention_layer(x)
print(output.shape)  # Output shape should be [1024, 5, 20, 20]
