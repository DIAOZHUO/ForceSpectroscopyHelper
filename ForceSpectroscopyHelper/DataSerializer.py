import pickle, os
import numpy as np


class DataSerializer:

    def __init__(self, path, custom_ext=".pkl"):
        self.path = path
        self._ext = custom_ext
        self.header = None
        self.data_dict = {}

    @property
    def header_key(self):
        return "data_main_header"

    def set_header(self, header):
        self.data_dict[self.header_key] = header

    def save(self):
        if self.header_key not in self.data_dict:
            raise ValueError("Save file need a header, use set_header(type(dict)) function")
        filename, file_extension = os.path.splitext(self.path)
        with open(filename + self._ext, "wb") as f:
            pickle.dump(self.data_dict, f, pickle.HIGHEST_PROTOCOL)
            print("save to", filename + self._ext)

    def load(self):
        filename, file_extension = os.path.splitext(self.path)
        if file_extension != self._ext:
            filename = self.path
        with open(filename + self._ext, "rb") as f:
            self.data_dict = pickle.load(f)
            self.header = self.data_dict[self.header_key]

    def add_data(self, key, data, overwrite=False, save=False):
        if key in self.data_dict:
            if overwrite:
                self.data_dict.pop(key)
                self.data_dict[key] = data
        else:
            self.data_dict[key] = data

        if save:
            self.save()

    def remove_data(self, key, save=False):
        if key in self.data_dict:
            self.data_dict.pop(key)
        if save:
            self.save()

    @staticmethod
    def to_matrix_buffer(ndarray):
        return ndarray.tobytes()

    @staticmethod
    def from_matrix_buffer(buffer):
        return np.frombuffer(buffer)
