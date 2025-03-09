import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
"""
I choose to use pytorch due to performance issue on tensorflow and more documentation on DDPM model.

(in addition my intership in Aalto University consist of improving diffusion model for driving simulation video)
"""
T = 5
betas = torch.linspace(0.05, 0.1, T) 
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, 0)

"""
using sinusoidal embedding for the time index (like in the transformer model) to represent the time.

The sinusoidal embedding is a more way to represent the time index in a continuous space because because it's cyclic and not croissant like the linear time index.
"""



def sinusoidal_embedding(timesteps, dim):
    """Generates sinusoidal embeddings for a given timestep."""
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    # Create the range tensor on the same device as timesteps
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb_scale)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # pad if dim is odd
        emb = torch.cat([emb, torch.zeros(timesteps.size(0), 1, device=timesteps.device)], dim=1)
    return emb
"""
The noise is computed by adding a gaussian noise to the image with a factor alpha getted from the alphas_cumprod list.
"""
def add_noise(x, t, alphas_cumprod):
    a = alphas_cumprod[t].view(-1, 1, 1, 1)
    noise = torch.randn_like(x)
    return torch.sqrt(a) * x + torch.sqrt(1 - a) * noise, noise



"""
unet using sinusoidal embedding for time index.

"""
class SinusTimestamplUNet(nn.Module):
    def __init__(self, T, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, 28*28)
        )
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU())
        self.conv4 = nn.Conv2d(16, 1, 1)
        self.time_emb_dim = time_emb_dim

    def forward(self, x, t):
        # Create sinusoidal embeddings and process through an MLP
        t_emb = sinusoidal_embedding(t, self.time_emb_dim).to(x.device)
        t_emb = self.time_mlp(t_emb).view(-1, 1, 28, 28)
        x = x + t_emb  # Conditioning with time embedding
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        c2 = self.conv2(p1)
        up = self.up(c2)
        x = self.conv3(torch.cat([up, c1], 1))
        return self.conv4(x)



"""
Minimal UNet model for image denoising with time embeding.

The model is compose of 1 emebeding for mapping the time index, 1 convolutional layer, 1 max pooling layer, 2 convolutional layers with a 1 upsampling layer and 1 convolutional layer.



"""
class MinimalUNet(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.time_embed = nn.Embedding(T, 28 * 28)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Sequential(nn.Conv2d(24, 8, 3, padding=1), nn.ReLU())
        self.conv4 = nn.Conv2d(8, 1, 1)

    def forward(self, x, t):
        # Embed timestep and add as conditioning
        t_emb = self.time_embed(t).view(-1, 1, 28, 28)
        #the timestep and the image are concatened
        x = x + t_emb
        #dimension is reduced (encoder)
        c1 = self.conv1(x) 
        p1 = self.pool(c1)
        #dimension is upped (decoder)
        c2 = self.conv2(p1)
        up = self.up(c2)
        #last layer is concatined with the first convolution layer, so we not loose information due to the pooling
        x = self.conv3(torch.cat([up, c1], 1))
        return self.conv4(x)
    
#load MNST dataset
def load_data(bs=128):
    transform = transforms.ToTensor()
    train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test  = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test, batch_size=bs, shuffle=False)
    return train_loader, test_loader


"""
Training :

train model with noisy image using a random timestep to train the model on different noise level.

the model predict using the timestep value and the noisy image.

the loss function (MSE) is computed between the noisy image and the noise:
* We train the model to separate the noise from the image
* I have better decoding result when I predict the noiseless image directly instead of predicting the noise and then remove it, but i cannot compute the inference like intended

bs correspond to the batch size

alphas_cumprod is a list of alpha with a timestep as indices
"""
def train(model, loader, epochs, device, alphas_cumprod):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in loader:
            x = x.to(device)
            bs = x.size(0) 
            t = torch.randint(0, T, (bs,), device=device)
            noisy, noise = add_noise(x, t, alphas_cumprod)
            opt.zero_grad()
            pred = model(noisy, t)
            loss= loss_fn(pred, noise)
           # loss = loss_fn(pred, add_noise(x, t-1, alphas_cumprod)[0])
            loss.backward()
            opt.step()
            total_loss += loss.item() * bs
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader.dataset):.4f}")
        #save the model
    torch.save(model.state_dict(), f"model_epoch_{epochs}.pth")
        #return the loss
    return total_loss/len(loader.dataset)


"""
using a random noise image as a input.

at each timestep iteration we remove progressively the noise hallucinated by the model to match his training distribution.

We do it in a reverse order to remove the noise from the image.
"""
def inference(model, device, alphas_cumprod, T, shape=(1, 1, 28, 28)):
 
    x = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        t_tensor = torch.full((x.size(0),), t, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(x, t_tensor)
        a = alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
        # remove the predicted noise from the image.
        x = (x - torch.sqrt(1 - a) * pred) / torch.sqrt(a)
     #   x = pred
    return x


import argparse

# python diffusion.py --epochs 200 --batch_size 128 --model minimal
# python diffusion.py --epochs 200 --batch_size 128 --model sinus --time_emb_dim 128
def main():
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--model", type=str, default="minimal", choices=["minimal", "sinus"],
                        help="Model type to use: 'minimal' or 'sinus'")
    parser.add_argument("--time_emb_dim", type=int, default=128,
                        help="Time embedding dimension (only used for sinus model)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data with the batch size
    train_loader, test_loader = load_data(bs=args.batch_size)
    
    # Choose model based on the argument
    if args.model == "minimal":
        model = MinimalUNet(T).to(device)
    elif args.model == "sinus":
        model = SinusTimestamplUNet(T, time_emb_dim=args.time_emb_dim).to(device)
    
    alphas_cumprod_dev = alphas_cumprod.to(device)
    
    # Train the model with specified epochs and capture the final loss
    last_loss = train(model, train_loader, epochs=args.epochs, device=device, alphas_cumprod=alphas_cumprod_dev)
    
    # Visualize iterative denoising on test sample
    model.eval()
    test_x, _ = next(iter(test_loader))
    test_x = test_x.to(device)
    
    # Get the noisy version at the maximum noise level
    t_initial = T - 1
    noisy, _ = add_noise(test_x, t_initial, alphas_cumprod_dev)
    
    # Iteratively denoise starting from the noisy image
    intermediate_images = [noisy]
    x = noisy.clone()
    for t in reversed(range(T)):
        t_tensor = torch.full((x.size(0),), t, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(x, t_tensor)
        a = alphas_cumprod_dev[t].view(-1, 1, 1, 1)
        x = (x - torch.sqrt(1 - a) * pred) / torch.sqrt(a)
        intermediate_images.append(x)
    
    # Plot results for a few test samples
    num_samples = 5
    num_cols = 1 + len(intermediate_images)  # original + each intermediate step
    plt.figure(figsize=(2 * num_cols, 2 * num_samples))
    titles = ["Original", f"Noisy (t={t_initial})", *[f"Step {t}" for t in reversed(range(T))]]
    
    for i in range(num_samples):
        plt.subplot(num_samples, num_cols, i * num_cols + 1)
        plt.imshow(test_x[i].cpu().squeeze(), cmap='gray')
        plt.title(titles[0])
        plt.axis('off')
        for j in range(len(intermediate_images)):
            plt.subplot(num_samples, num_cols, i * num_cols + 2 + j)
            plt.imshow(intermediate_images[j][i].cpu().squeeze(), cmap='gray')
            plt.title(titles[j+1])
            plt.axis('off')
    plt.suptitle(f"Iterative Denoising - Last Loss: {last_loss:.4f}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Generate a new image from random noise
    generated = inference(model, device, alphas_cumprod_dev, T, shape=(1, 1, 28, 28))
    generated_img = generated.cpu().squeeze().detach().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(generated_img, cmap='gray')
    plt.title("Generated Image")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()