import torch
import typer
from torch import nn
from data import corrupt_mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
app = typer.Typer()

@app.command()
def train(lr: float = 1e-3, batch_size: int = 25, epochs: int = 5)-> None:
    """Train a model on MNIST"""
    print("Training....")
    print(lr)

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set,batch_size = batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for e in range(epochs):
        model.train()
        for i , (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE) #make sure the images are in GPU if possible
            optimizer.zero_grad() # zero the gradients for each batch
            y_pred = model(img) #infer batch
            y_target = target #set target
            loss = loss_fn(y_pred,y_target) #calculate the loss (difference between pred and target)
            loss.backward() # backpropagate the loss to update weights
            optimizer.step() #step according to gradient from backprop

            statistics["train_loss"].append(loss.item()) #use item to extract the float value from the tensor
            
            # compare the y_pred class (y_pred.shape [batch_size, numclass] i.e. [25,10]) to target class and convert to float to calculate the mean over the batch and then extract the value from the tensor with .item()
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item() 
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0: # checks if i is divisible by 100 and prints status every 100 iterations
                print(f"Epoch {e}, iter {i}, loss: {loss.item()}")


    print("Training complete")
    torch.save(model.state_dict(), "model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")

@app.command()
def evaluate(model_checkpoint: str):
    """evaluate model"""
    print("Evaluating model")
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _ , test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size= 32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)

        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    app()