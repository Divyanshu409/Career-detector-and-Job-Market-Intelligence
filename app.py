from flask import Flask, render_template, request, jsonify

from modules.data_processing import load_and_preprocess
from modules.model import SalaryPredictor, analyse_skill_gap
from modules.recommender import (
    similar_roles,
    generate_learning_path
)

app = Flask(__name__)

# ==========================
# Load Dataset
# ==========================

print("Loading jobs dataset...")

data = load_and_preprocess("data/jobs.csv")

df = data["df"]

print("Jobs Loaded:", len(df))

# ==========================
# Train Salary Model
# ==========================

salary_model = SalaryPredictor()

result = salary_model.train(df)

print("Model Training Complete")
print(result)

# ==========================
# Home Page
# ==========================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================
# Salary Prediction API
# ==========================

@app.route("/predict_salary", methods=["POST"])
def predict_salary():

    data = request.json

    role = data.get("role", "data scientist")
    experience = data.get("experience", "mid")
    skills = data.get("skills", [])

    result = salary_model.predict(
        role,
        experience,
        skills
    )

    return jsonify(result)

# ==========================
# Skill Gap API
# ==========================

@app.route("/skill_gap", methods=["POST"])
def skill_gap():

    data = request.json

    career_goal = data.get("career_goal")
    skills = data.get("skills", [])

    result = analyse_skill_gap(
        career_goal,
        skills
    )

    return jsonify(result)

# ==========================
# Similar Roles API
# ==========================

@app.route("/similar_roles", methods=["POST"])
def get_similar_roles():

    data = request.json

    role = data.get("career_goal")

    result = similar_roles(role)

    return jsonify(result)

# ==========================
# Roadmap API
# ==========================

@app.route("/roadmap", methods=["POST"])
def roadmap():

    data = request.json

    role = data.get("career_goal")
    skills = data.get("skills", [])

    result = generate_learning_path(
        role,
        skills
    )

    return jsonify(result)

# ==========================
# Start App
# ==========================

if __name__ == "__main__":
    app.run(debug=True)