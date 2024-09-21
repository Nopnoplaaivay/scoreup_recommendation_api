import os
import pandas as pd
import numpy as np

from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from print_module import Print

load_dotenv()

class Database:
    def __init__(self, course_id="c3a788eb31f1471f9734157e9516f9b6"):
        # self.client = MongoClient(os.getenv("LOCAL_MONGO_URL"))
        self.client = MongoClient(os.getenv("MONGO_URL"))
        self.course_id = course_id
        self.db = self.client["codelab1"]
        self.logs = self.db["logs-questions"]
        self.users = self.db["users"]
        self.questions = self.db["questions"]
        self.kncp = self.db["knowledge_concepts"]
        # self.episodes = self.db["episodes"]
        self.action_space = self.call_action_space(self.course_id)

    def check_connection(self):
        try:
            databases = self.client.list_database_names()
            print("Connection successful!")
            print("Databases:", databases)
        except Exception as e:
            print("Connection failed:", e)

    def call_action_space(self, course_id):
        course_questions = self.questions.find({"notionDatabaseId": course_id})
        action_space = [elem['_id'] for elem in course_questions]
        return action_space
    
    def encode_exercise_ids(self):
        exercise_ids = self.action_space
        # Create a mapping dictionary and save to mongodb
        exercise_mapping = {exercise_id: idx for idx, exercise_id in enumerate(exercise_ids)}
        for exercise_id, code in exercise_mapping.items():
            self.questions.update_one(
                {"_id": exercise_id}, {"$set": {"encoded_exercise_id": code}}
            )
        Print.success("Exercise IDs encoded successfully!")

    def encode_knowledge_concepts(self):
        kncp_list = [elem["_id"] for elem in self.kncp.find()]

        num_bits = len(bin(len(kncp_list))) - 2
        concept_mapping = {}

        for idx, concept in enumerate(kncp_list):
            binary_code = format(idx, '0{}b'.format(num_bits))
            concept_mapping[concept] = binary_code
            print(concept, idx, binary_code)

        # Save to mongodb
        for concept, code in concept_mapping.items():
            self.kncp.update_one(
                {"_id": concept}, {"$set": {"binary_code": code}}
            )
        Print.success("Knowledge concepts encoded successfully!")

    def update_log_knowledge_concepts(self):
        logs = self.logs.find()
        kncp_list = [elem["_id"] for elem in self.kncp.find()]
        for log in logs:
            # If knowledge concept of log is not in kncp_list, remove log
            if log["knowledge_concept"] not in kncp_list:
                self.logs.delete_one({"_id": log["_id"]})
                # print(f"Log {log['_id']} removed!")
        Print.success("Knowledge concepts of logs updated successfully!")

    def update_difficulty(self, course_id):
        questions = self.questions.find({"notionDatabaseId": course_id})
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

    def reset_logs(self):
        exercise_ids = self.action_space
        logs_exer_ids = [log["exercise_id"] for log in self.logs.find()]
        for log_exer_id in logs_exer_ids:
            if log_exer_id not in exercise_ids:
                Print.warning(f"Exercise {log_exer_id} not in action space!")
                self.logs.delete_one({"exercise_id": log_exer_id})
        Print.success("Logs reset successfully!")

    def latest_user_log(self, user_id):
        user_logs = self.logs.find({"user_id": user_id}).sort("timestamp", -1)
        return user_logs[0]


# db = Database()
# db.check_connection()
# print(db.latest_user_log("669d16e11db84069209550bd"))
# db.encode_knowledge_concepts()
# db.update_log_knowledge_concepts()
# db.encode_exercise_ids()
# db.update_difficulty(db.course_id)
# db.reset_logs()