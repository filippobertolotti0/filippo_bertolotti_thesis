from numpy import full
import pickle

class Dictionary:
    def __init__(self, init_value, n_actions, filename=None):
        self.table = {}
        self.init_value = init_value
        self.n_actions = n_actions

        if filename:
            try:
                with open(filename, 'rb') as file:
                    self.table = pickle.load(file)
            except FileNotFoundError:
                print("cache file not found. creating a new cache")
            except Exception as e:
                print(f"an error occurred while loading the cache: {e}")

    def get(self, observation):
        try:
            cell = self.table[observation]
        except KeyError:
            self.table[observation] = full(self.n_actions, self.init_value, dtype=float)
            cell = self.table[observation]

        return cell
    
    def save_table(self, filename):
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self.table, file)
        except Exception as e:
            print(f"An error occurred while saving the cache: {e}")