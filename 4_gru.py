import pandas as pd
import backprophet_models as bpm
import backprophet_utils as bpu


TENSORBOARD = False  # Use tensorboard for logging, but you need to start it manually in the background
RENDER_PLOTS = False
LOOK_BACK_PERIOD = 60  # Number of days to look back for building "sliding window", cf. bpu.get_train_test_set()
MODEL_NAME = "gru"


def main():
    device = bpu.set_torch_device()
    # end_date = pd.Timestamp.today() - pd.DateOffset(days=1)  # Only use if you want to run the model again for e.g. yesterday
    end_date = pd.Timestamp.today()
    end_date = pd.to_datetime(end_date).date()
    backprophet_data = bpu.BackprophetData(end_date, MODEL_NAME, LOOK_BACK_PERIOD)

    # Hyperparameters
    learning_rate = 0.001
    training_epochs = 300
    batch_size = 128
    hidden_size = 256
    num_layers = 3

    model = bpm.GRUModel(backprophet_data.n_features, hidden_size, num_layers, 1).to(device)
    bpu.train_eval_model(model, backprophet_data, device, learning_rate, training_epochs, batch_size, RENDER_PLOTS, TENSORBOARD)


if __name__ == "__main__":
    main()
