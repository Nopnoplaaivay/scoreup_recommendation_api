import pickle
import os

class Score:
    def __init__(self, file_path="state.pkl"):
        self.file_path = file_path
        
    def save_state(self, score_history, best_score):
        with open(self.file_path, 'wb') as f:
            pickle.dump({'score_history': score_history, 'best_score': best_score}, f)

    def load_state(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                state = pickle.load(f)
                return state['score_history'], state['best_score']
        return [], 0
    
    def delete_state(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            return True
        return False