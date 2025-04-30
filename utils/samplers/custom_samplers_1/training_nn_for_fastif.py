import torch
from torch import optim, nn
from tqdm import tqdm


def train_nn(net, train_loader, x_eval, y_eval, num_epochs, learning_rate, loss_function=None):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_hist = []
    eval_hist = []
    if loss_function is None:
        loss_function = nn.MSELoss()

    with tqdm(total=len(train_loader) * num_epochs, position=0, leave=True) as pbar:

        for epoch in range(1, num_epochs + 1):
            running_loss = 0.0

            net.train()
            for batch_num, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = loss_function(outputs, targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                pbar.set_description("Epoch: %d, Batch: %2d, Loss: %.2f" % (epoch, batch_num, running_loss))
                pbar.update()
            loss_hist.append(running_loss / len(train_loader))
            net.eval()
            with torch.no_grad():
                eval_hist.append(loss_function(net(x_eval), y_eval).item())
        pbar.close()

    return loss_hist, eval_hist
