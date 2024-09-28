import random

from model.mongodb import Database

class Actions:
    def __init__(self, db) -> None:

        self.actions = db.action_space
        self.n = len(self.actions)

class observation_space:
    def __init__(self, n) -> None:
        self.shape = (n,)

class Environment:
    '''Khởi tạo môi trường học tập với chương hiện tại'''
    def __init__(self, db, cur_chapter) -> None:
        self.db = db
        self.db.update_action_space(cur_chapter=cur_chapter)
        self.observation_space = observation_space(8)
        self.action_space = Actions(db)
        self.actions = [self.get_action(action) for action in self.db.action_space]
        self.cur_chapter = cur_chapter

    '''1. Hàm phần thưởng'''
    def reward_func(self, state, next_state):

        '''Prepare state'''
        diff_t1 = state[0]
        bookmarked_t1 = state[2]
 
        '''Prepare next state'''
        diff_t2 = next_state[0] 
        bookmarked_t2 = next_state[2]

        '''Score'''
        score = next_state[1]  # Score at time t (example

        '''Handle knowledge concept'''
        kncp_t1 = state[3:]
        kncp_t2 = next_state[3:]
        kncp_t1_str = ''.join([str(bit) for bit in kncp_t1])
        kncp_t2_str = ''.join([str(bit) for bit in kncp_t2])

        '''Compute reward'''
        R1 = -2 if (score == 0 and kncp_t1_str != kncp_t2_str) else 1
        R2 = -(diff_t2 - diff_t1) ** 2
        R3 = 3 if bookmarked_t2 == 1 else 0

        return 0.4 * R1 + 0.4 * R2 + 0.2 * R3

    '''2. Trích xuất state từ log'''
    def extract_state(self, log):
        exercise_id = log["exercise_id"]
        difficulty = log["difficulty"]
        score = log["score"]
        bookmarked = log["bookmarked"] if "bookmarked" in log else 0

        knowledge_concept_id = log["knowledge_concept"]
        knowledge_concept_bin = self.db.kncp.find_one(
            {"_id": knowledge_concept_id}
        )["binary_code"]
        knowledge_concept_bin_array = [int(bit) for bit in knowledge_concept_bin]
        state = [difficulty, score, bookmarked] + knowledge_concept_bin_array

        return exercise_id, state
    
    '''3. Trích xuất state từ body request của API'''
    def convert_state(self, raw_state):
        difficulty = raw_state[0]
        score = raw_state[1]
        bookmarked = raw_state[2]
        knowledge_concept = raw_state[3]

        knowledge_concept_bin = self.db.kncp.find_one(
            {"_id": knowledge_concept}
        )["binary_code"]
        knowledge_concept_bin_array = [int(bit) for bit in knowledge_concept_bin]
        state = [difficulty, score, bookmarked] + knowledge_concept_bin_array

        return state
    
    '''4. Lấy action từ exercise_id'''
    def get_action(self, exercise_id):
        action = self.db.questions.find_one({"_id": exercise_id})["encoded_exercise_id"]
        return action
    
    '''5. Lấy exercise_id từ action'''
    def get_exercise_by_action(self, action):
        exercise_id = self.db.questions.find_one({"encoded_exercise_id": action})["_id"]
        return exercise_id

# env = Environment(Database(), "chuong-2")
# print(env.action_space.actions)
# print(env.action_space.n)
