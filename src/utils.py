import os
import pickle


def save_processed_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(model, open(path, 'wb'))


def load_model(path):
    return pickle.load(open(path, 'rb'))