import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Define the Liquid Neural Network
class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([self._create_layer(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def _create_layer(self, input_size, hidden_size):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

# Define the ODE Solver
class ODESolver(nn.Module):
    def __init__(self, model, dt):
        super(ODESolver, self).__init__()
        self.model = model
        self.dt = dt

    def forward(self, x):
        with torch.enable_grad():
            for layer in self.model.layers:
                x = layer(x)
        return self.model.output_layer(x)

    def loss(self, x, y):
        outputs = self.forward(x)
        return nn.functional.cross_entropy(outputs, y)

# Training function
def train(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the images
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss(inputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, loss: {total_loss/len(dataloader)}')

# Evaluation function
def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the images
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy







# Visualization function
def visualize_liquid_diagram(model, input_data, num_steps=100):
    model.eval()
    with torch.no_grad():
        hidden_states = []
        x = input_data
        for _ in range(num_steps):
            for layer in model.model.layers:  # Access layers from the model inside ODESolver
                x = layer(x)
            hidden_states.append(x.numpy())

        hidden_states = np.array(hidden_states)

        if hidden_states.size == 0:
            print("No hidden states to visualize.")
            return

        plt.figure(figsize=(10, 6))
        for i in range(min(5, hidden_states.shape[2])):  # Plot first 5 dimensions
            plt.plot(range(num_steps), hidden_states[:, 0, i], label=f'Dimension {i+1}')

        plt.title('Liquid Neural Network: Hidden State Trajectory')
        plt.xlabel('Time Steps')
        plt.ylabel('Hidden State Value')
        plt.legend()
        plt.show()



# Main execution
if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    try:
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Try downloading the dataset manually and placing it in the './data' directory.")
        exit()

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Set hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28
    hidden_size = 64
    num_layers = 3
    output_size = 10  # 10 digit classes
    learning_rate = 0.001
    epochs = 10

    # Create and train model
    model = LiquidNeuralNetwork(input_size, hidden_size, num_layers, output_size)
    ode_solver = ODESolver(model, dt=0.1)
    optimizer = optim.Adam(ode_solver.parameters(), lr=learning_rate)

    print("Starting training...")
    train(ode_solver, train_loader, optimizer, epochs)

    # Evaluate accuracy
    print("Evaluating accuracy...")
    accuracy = evaluate_accuracy(ode_solver, test_loader)

    # Visualize liquid diagram
    print("Visualizing hidden state trajectory...")
    sample_input = next(iter(test_loader))[0][0].view(1, -1)  # Take the first test sample
    visualize_liquid_diagram(ode_solver, sample_input)
