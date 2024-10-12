import os
import numpy as np

from pymongo import MongoClient
from dotenv import load_dotenv
from utils.print_module import Print

load_dotenv()


class Database:
    """Khỏi tạo course_id Nhập môn Công nghệ thông tin"""

    def __init__(self, course_id="c3a788eb31f1471f9734157e9516f9b6"):
        # self.client = MongoClient(os.getenv("LOCAL_MONGO_URL"))
        self.client = MongoClient(os.getenv("MONGO_URL"))
        self.course_id = course_id
        self.db = self.client["codelab1"]
        self.logs = self.db["logs-questions"]
        self.users = self.db["users"]
        self.questions = self.db["questions"]
        self.kncp = self.db["knowledge_concepts"]

        # Initialize action space
        self.action_space = []
        self.update_action_space()

    """1. Kiểm tra kết nối"""

    def check_connection(self):
        try:
            databases = self.client.list_database_names()
            print("Connection successful!")
            print("Databases:", databases)
        except Exception as e:
            print("Connection failed:", e)

    """2. Cập nhật không gian hành động dựa trên chương hiện tại"""

    def update_action_space(self, cur_chapter="chuong-1"):
        chapters_num = cur_chapter.split("-")[-1]
        chapters = [f"chuong-{i}" for i in range(1, int(chapters_num) + 1)]
        course_questions = self.questions.find(
            {"notionDatabaseId": self.course_id, "chapter": {"$in": chapters}}
        )
        action_space = [elem["_id"] for elem in course_questions]
        self.action_space = action_space

    """3. Mã hóa các ID bài tập"""

    def encode_exercise_ids(self):

        chapters = [f"chuong-{i}" for i in range(1, 5)]

        exercise_by_chapter = {}
        for chapter in chapters:
            exercise_by_chapter[chapter] = list(
                self.questions.find({"chapter": chapter})
            )

        new_exercise_ids = 0

        for chapter in chapters:
            for exercise in exercise_by_chapter[chapter]:
                exercise["encoded_exercise_id"] = new_exercise_ids
                new_exercise_ids += 1

                self.questions.update_one(
                    {"_id": exercise["_id"]},
                    {"$set": {"encoded_exercise_id": exercise["encoded_exercise_id"]}},
                )

        Print.success("Exercise IDs encoded successfully!")

    """4. Mã hóa các khái niệm kiến thức"""

    def encode_knowledge_concepts(self):
        kncp_list = [elem["_id"] for elem in self.kncp.find()]

        num_bits = len(bin(len(kncp_list))) - 2
        concept_mapping = {}

        for idx, concept in enumerate(kncp_list):
            binary_code = format(idx, "0{}b".format(num_bits))
            concept_mapping[concept] = binary_code
            print(concept, idx, binary_code)

        # Save to mongodb
        for concept, code in concept_mapping.items():
            self.kncp.update_one({"_id": concept}, {"$set": {"binary_code": code}})
        Print.success("Knowledge concepts encoded successfully!")

    """5. Cập nhật các phạm trù kiến thức của logs"""

    def update_log_knowledge_concepts(self):
        logs = self.logs.find()
        kncp_list = [elem["_id"] for elem in self.kncp.find()]
        for log in logs:
            # If knowledge concept of log is not in kncp_list, remove log
            if log["knowledge_concept"] not in kncp_list:
                self.logs.delete_one({"_id": log["_id"]})
                # print(f"Log {log['_id']} removed!")
        Print.success("Knowledge concepts of logs updated successfully!")

    """6. Cập nhật độ khó cho bài tập"""

    def update_difficulty(self):
        questions = self.questions.find({"notionDatabaseId": self.course_id})
        questions = [ques for ques in questions]

        # Update difficulty for questions
        for ques in questions:
            ques_id = ques["_id"]
            logs = self.logs.find({"exercise_id": ques_id})
            log_by_question = [log for log in logs]
            accuracies = []
            exercise_difficulty = 0

            if len(log_by_question) > 0:
                for log in log_by_question:
                    total_answers = len(log["user_ans"])
                    correct_answers = sum(
                        [
                            1
                            for i in range(total_answers)
                            if log["correct_ans"][i] == log["user_ans"][i]
                        ]
                    )
                    accuracy = correct_answers / total_answers
                    accuracies.append(accuracy)
                accuracies = np.array(accuracies)

                high_threshold = np.percentile(accuracies, 73)
                low_threshold = np.percentile(accuracies, 27)

                high_scoring_group = accuracies[accuracies >= high_threshold]
                low_scoring_group = accuracies[accuracies <= low_threshold]

                exercise_difficulty = (
                    1 - (np.mean(high_scoring_group) + np.mean(low_scoring_group)) / 2
                )
            # print(f"Exercise Difficulty: {exercise_difficulty:.2f}")

            self.questions.update_one(
                {"_id": ques_id}, {"$set": {"difficulty": exercise_difficulty}}
            )
        Print.success("Difficulties for exercises updated successfully!")
        # Update difficulty for log
        logs = self.logs.find()
        for log in logs:
            exercise_id = log["exercise_id"]
            question = self.questions.find_one({"_id": exercise_id})
            difficulty = question["difficulty"]
            self.logs.update_one(
                {"_id": log["_id"]}, {"$set": {"difficulty": difficulty}}
            )
        Print.success("Difficulties for logs updated successfully!")

    """7. Kiểm tra logs không tồn tại trong không gian hành động"""

    def reset_logs(self):
        exercise_ids = self.action_space
        logs_exer_ids = [log["exercise_id"] for log in self.logs.find()]
        for log_exer_id in logs_exer_ids:
            if log_exer_id not in exercise_ids:
                Print.warning(f"Exercise {log_exer_id} not in action space!")
                self.logs.delete_one({"exercise_id": log_exer_id})
        Print.success("Logs reset successfully!")

    """8. Update chapter cho logs"""

    def update_chapter(self):
        logs = self.logs.find()
        for log in logs:
            exercise_id = log["exercise_id"]
            question = self.questions.find_one({"_id": exercise_id})
            chapter = question["chapter"]
            self.logs.update_one({"_id": log["_id"]}, {"$set": {"chapter": chapter}})
        Print.success("Chapters for logs updated successfully!")

    """9. Update knowledge concepts cho logs"""
    def get_exercise_message(self, exercise_id, user_id):
        # Check if user has answered the question before
        try:
            log = self.logs.find_one({"exercise_id": exercise_id, "user_id": user_id})
            if log:
                # Check if log has bookmarked
                if "bookmarked" in log:
                    bookmarked = log["bookmarked"]
                    if bookmarked:
                        return {"message": "bookmarked"}
                # Check if user has answered the question correctly
                elif "score" in log:
                    score = log["correct_ans"]
                    if score == 0:
                        return {"message": "incorrect"}
                    else:
                        return {"message": "correct"} 
            else:
                # Get difficulty of the question
                question = self.questions.find_one({"_id": exercise_id})
                difficulty = question["difficulty"] if question["difficulty"] else 0
                if difficulty >= 0.5:
                    return {"message": "difficult"}
                else:
                    return {"message": "easy"}
        except Exception as e:
            print(e)
            return {"message": None}
        

        