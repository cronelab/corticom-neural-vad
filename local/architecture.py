import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from torch.optim.lr_scheduler import StepLR
from typing import Optional, Tuple
from torchinfo import summary
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score
from pathlib import Path
import matplotlib.pyplot as plt
from local.utils import compute_trial_based_error, StoreBestModel
from Levenshtein import distance


States = Tuple[torch.Tensor, torch.Tensor]
LstmState = Tuple[torch.Tensor, torch.Tensor]


# region Model architectures
class UnidirectionalVoiceActivityDetector(nn.Module):
    """
    The nVAD model from the delayed speech synthesis paper
    """
    def __init__(self, nb_layer: int = 2, nb_hidden_units: int = 512, nb_electrodes: int = 128, dropout: float = 0.0):
        super(UnidirectionalVoiceActivityDetector, self).__init__()
        self.nb_hidden_units = nb_hidden_units
        self.nb_layer = nb_layer
        self.lstm = nn.LSTM(input_size=nb_electrodes, hidden_size=self.nb_hidden_units, num_layers=nb_layer,
                            dropout=dropout, batch_first=True)
        self.classifier = nn.Linear(in_features=self.nb_hidden_units, out_features=2)

    def initial_states(self, batch_size: int, device: str = "cpu", req_grad: bool = False) -> LstmState:
        """
        Initialize LSTM states for the corresponding layers in the model
        """
        params = dict(requires_grad=req_grad, device=device, dtype=torch.float32)
        return (torch.zeros(self.nb_layer, batch_size, self.nb_hidden_units, **params),
                torch.zeros(self.nb_layer, batch_size, self.nb_hidden_units, **params))

    @staticmethod
    def visualize_vad_predictions(pred: np.ndarray, orig: np.ndarray, speech_probs: np.ndarray, filename: Path):
        """
        Plot the original and the predicted curves of the VAD. Title indicates how many frames have been correctly
        classified.
        """
        fig, ax = plt.subplots(1, 1, num=1, clear=True)
        ax.plot(orig, c="black", linestyle="--")
        ax.plot(pred, c="orange")
        ax.plot(speech_probs, c="blue")
        ax.axhline(0.5, c="gray", alpha=0.5)
        ax.set_xlim(0, len(speech_probs))
        ax.set_xlabel("Time [seconds]")
        ax.set_ylabel("Probability")
        ax.set_xticks([0, 100])
        ax.set_xticklabels([0, 1])
        ax.set_title(f"Trial accuracy: {list(pred == orig).count(True) / len(pred) * 100:.2f}")
        plt.savefig(filename.as_posix(), dpi=72)

    def forward(self, x: torch.Tensor, state: Optional[LstmState] = None) -> Tuple[torch.Tensor, LstmState]:
        """
        Forward pass of the model
        """
        if state is None:
            state = self.initial_states(batch_size=x.size(0), device=next(self.parameters()).device)

        x, new_state = self.lstm(x, state)
        out = self.classifier(x)
        return out, new_state

    def optimize(self, n_epochs: int, tr_dataloader: DataLoader, va_dataloader: DataLoader, cfunc: nn.Module,
                 optim: torch.optim.Optimizer, device: str, seq_len: int, best_model: Path) -> None:
        """
        Train the UnidirectionalVoiceActivityDetector model
        """
        model_updater = StoreBestModel(filename=best_model.as_posix())
        update_steps_counter = 0
        scheduler = StepLR(optim, step_size=2, gamma=0.5)

        for epoch in range(n_epochs):
            # Keep running loss during epoch computation
            train_loss = []

            # Iterate over all trials
            self.train()
            pbar = tqdm.tqdm(tr_dataloader, total=len(tr_dataloader))
            for x_train, y_train in tr_dataloader:
                # Initialize state
                state = self.initial_states(batch_size=x_train.size(0), device=device, req_grad=True)

                # Use truncated backpropagation with k1 == k2
                sequences = zip(x_train.split(seq_len, dim=1), y_train.double().split(seq_len, dim=1))
                for x_train_seq, y_train_seq in sequences:
                    # Zero-out gradients
                    for param in self.parameters():
                        param.grad = None

                    # Process each sample in the current sequence
                    x_train_seq = x_train_seq.to(device=device).float()
                    y_train_seq = y_train_seq.to(device=device).float()

                    output, state = self(x_train_seq, state)  # Forward propagation

                    # Make backward propagation
                    loss = cfunc(output.reshape((-1, 2)), y_train_seq.reshape(-1).long())
                    loss.backward()
                    optim.step()
                    update_steps_counter += 1

                    # Detach state from computational graph
                    state = (state[0].detach(), state[1].detach())

                    train_loss.append(loss.item())

                pbar.set_description(f'Epoch {epoch + 1:>04}: Train loss: {sum(train_loss) / len(train_loss):.04f} '
                                     f'-- Validation loss:...')
                pbar.update()

            # Adjust learning rate
            scheduler.step()

            # Compute loss and accuracy on validation data
            self.eval()
            valid_loss = []

            valid_output = []
            valid_target = []
            valid_trials = []
            for val_index, (x_val, y_val) in enumerate(va_dataloader):
                init_state = self.initial_states(batch_size=x_val.size(0), device=device)

                # Process each sample in the current trial
                x_val = x_val.to(device=device).float()
                y_val = y_val.to(device=device).float()

                output, _ = self(x_val, init_state)  # Forward propagation

                # Compute loss
                loss = cfunc(output.reshape((-1, 2)), y_val.reshape(-1).long())
                valid_loss.append(loss.item())

                # Visualize the output of the neural VAD in comparison with the ground truth
                pred = F.softmax(output, dim=2).argmax(dim=2).squeeze().detach().cpu().numpy()
                prob = F.softmax(output, dim=2).squeeze().detach().cpu().numpy()[:, 1]

                filename = best_model.parent / "dev" / f"epoch={epoch + 1:03d}" / f"trial_id={val_index+1:03d}.png"
                os.makedirs(filename.parent.as_posix(), exist_ok=True)
                self.visualize_vad_predictions(pred=pred, orig=y_val.cpu().numpy().flatten(), speech_probs=prob,
                                               filename=filename)

                # Store results to list in order to compute the median alignment error
                valid_output.extend(pred.tolist())
                valid_target.extend(y_val.cpu().numpy().flatten().tolist())
                valid_trials.extend(np.ones_like(pred) * val_index)

                err = compute_trial_based_error(nvad=np.array(valid_output), vad=np.array(valid_target),
                                                trial_ids=np.array(valid_trials))

                pbar.set_description(
                    f'Epoch {epoch + 1:>04}: Train loss: {np.mean(train_loss):.04f} -- '
                    f'Validation loss: {np.mean(valid_loss):.04f} (Avg cost per trial: {np.mean(err):.02f} sec, '
                    f'Update steps: {update_steps_counter})')
                pbar.update()
            pbar.close()

            # Store new model weights if accuracy score has improved
            model_updater.update(model=self, score=np.mean(valid_loss),
                                 info={"update_steps": update_steps_counter, "epoch": epoch + 1})

    def predict(self, te_dataloader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return sequences of the predicted and original voice activity labels
        """
        self.eval()

        test_output = []
        test_target = []
        for idx, (x_test, y_test) in enumerate(te_dataloader):
            memory = self.initial_states(batch_size=x_test.size(0), device=device)
            x_test = x_test.to(device=device).float()
            y_test = y_test.to(device=device).float()

            # Forward propagation
            output, _ = self(x_test.float(), memory)

            test_output.append(F.softmax(output, dim=2).argmax(dim=2).detach().cpu().numpy().squeeze())
            test_target.append(y_test.detach().cpu().numpy().squeeze())

        return np.concatenate(test_output), np.concatenate(test_target)
# endregion


# region Baseline LeNet model (similar to Soroush et al., Speech Activity Detection from Stereotactic EEG, 2021)
class LeNetBasedVAD(nn.Module):
    """
    A network architecture based on LeNet and convolutional GRU cells for sequence modelling.
    """
    def __init__(self, n_classes: int = 2, in_channels: int = 1):
        super(LeNetBasedVAD, self).__init__()

        # Convolutional layers
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=1, padding='valid')
        self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding='valid')

        # Pooling layers
        self.s1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.s2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64, out_features=128), nn.Tanh(),
            nn.Linear(in_features=128, out_features=64), nn.Tanh(),
            nn.Linear(in_features=64, out_features=n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.c1(x))
        x = self.s1(x)
        x = torch.tanh(self.c2(x))
        x = self.classifier(x.squeeze())
        return x

    def optimize(self, n_epochs: int, tr_dataloader: DataLoader, va_dataloader: DataLoader, cfunc: nn.Module,
                 optim: torch.optim.Optimizer, best_model: Path, device: str = "cpu") -> None:
        """
        The training procedure to optimize the network weights based on the training and reports generalization on
        held-out validation data
        """
        best_model = StoreBestModel(filename=best_model.as_posix())
        update_steps = 0
        scheduler = StepLR(optim, step_size=2, gamma=0.5)

        # Iterate over each epoch
        for epoch in range(n_epochs):
            # Keep running losses during epoch computation
            train_loss = []
            valid_loss = []

            self.train()
            # Iterate over all trials
            pbar = tqdm.tqdm(tr_dataloader, total=len(tr_dataloader))
            for x_train, y_train in tr_dataloader:
                x_train = x_train.to(device).float()
                y_train = y_train.to(device).float()

                # Zero-out gradients
                for param in self.parameters():
                    param.grad = None

                # Put channel in the first position after batch size
                with torch.no_grad():
                    x_train = torch.permute(x_train, (0, 3, 1, 2))

                # Forward propagation
                output = self(x_train)

                # Backward propagation
                loss = cfunc(output, y_train.long())
                loss.backward()
                optim.step()
                train_loss.append(loss.item())
                update_steps += 1

                # Update progress bar
                pbar.set_description(
                    f'Epoch {epoch + 1:>04}: Train loss: {np.mean(train_loss):.04f} -- Validation loss:...')
                pbar.update()

            scheduler.step()
            self.eval()
            valid_output = []
            valid_target = []
            valid_trials = []
            for k, (x_val, y_val, y_ids) in enumerate(va_dataloader):
                x_val = x_val.to(device).float()
                y_val = y_val.to(device).float()

                # Forward propagation
                with torch.no_grad():
                    x_val = torch.permute(x_val, (0, 3, 1, 2))
                output = self(x_val)

                # Compute loss
                loss = cfunc(output, y_val.long())
                valid_loss.append(loss.item())

                # Compute running accuracy
                pred = F.softmax(output, dim=1).argmax(dim=1).detach().cpu().numpy()
                orig = y_val.detach().cpu().numpy()

                # Store results to list in order to compute the median alignment error
                valid_output.extend(pred.tolist())
                valid_target.extend(orig.tolist())
                valid_trials.extend(y_ids.tolist())

                err = compute_trial_based_error(nvad=np.array(valid_output), vad=np.array(valid_target),
                                                trial_ids=np.array(valid_trials))

                pbar.set_description(
                    f'Epoch {epoch + 1:>04}: Train loss: {np.mean(train_loss):.04f} -- Validation loss: '
                    f'{np.mean(valid_loss):.04f} (Avg cost per trial: {np.mean(err):.02f} sec, '
                    f'Update steps: {update_steps})')
                pbar.update()
            pbar.close()

            best_model.update(self, np.mean(valid_loss), info=dict(update_steps=update_steps))

    def predict(self, te_dataloader: DataLoader, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
        """
        Return sequences of the predicted and original voice activity labels
        """
        pred = []
        orig = []
        for x_test, y_test, _ in te_dataloader:
            x_test = x_test.to(device).float()
            y_test = y_test.to(device).float()

            with torch.no_grad():
                x_test = torch.permute(x_test, (0, 3, 1, 2))

            # Forward propagation
            output = self(x_test)
            if output.ndim == 1:
                output = torch.reshape(output, (1, 2))

            pred.append(F.softmax(output, dim=1).argmax(dim=1).detach().cpu().numpy())
            orig.append(y_test.detach().cpu().numpy())

        return np.concatenate(pred), np.concatenate(orig)
# endregion


if __name__ == '__main__':
    print("Unidirectional Voice Activity Detector")
    x = torch.rand(8, 400, 64, requires_grad=True, device="cpu")

    net = UnidirectionalVoiceActivityDetector(nb_electrodes=64, nb_layer=2, nb_hidden_units=128)
    states = net.initial_states(batch_size=8, device="cpu")
    summary(net, input_data=(x, states))
