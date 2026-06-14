from modules.data_processing import (
    load_and_preprocess,
    skill_frequency_df,
    trending_skills_df
)

from modules.model import (
    SalaryPredictor,
    analyse_skill_gap
)

from modules.recommender import (
    similar_roles,
    generate_learning_path
)

from modules.visualization import (
    top_skills_bar,
    skill_category_donut
)

def main():

    # Path to your CSV file
    csv_file = "data/jobs.csv"

    print("Loading and preprocessing data...")
    result = load_and_preprocess(csv_file)

    df = result["df"]

    print("\nDataset Statistics:")
    print(result["stats"])

    # ----------------------------------
    # Top Skills Analysis
    # ----------------------------------
    freq_df = skill_frequency_df(df)

    print("\nTop Skills:")
    print(freq_df.head(10))

    # Generate charts
    chart1 = top_skills_bar(freq_df)
    chart2 = skill_category_donut(freq_df)

    print("\nCharts generated successfully.")

    # ----------------------------------
    # Salary Prediction
    # ----------------------------------
    predictor = SalaryPredictor()

    print("\nTraining salary prediction models...")
    training_results = predictor.train(df)

    print(training_results)

    prediction = predictor.predict(
        role="data scientist",
        experience="mid",
        skills=["python", "sql", "machine learning", "pandas"]
    )

    print("\nSalary Prediction:")
    print(prediction)

    # ----------------------------------
    # Skill Gap Analysis
    # ----------------------------------
    gap = analyse_skill_gap(
        "data scientist",
        ["python", "sql", "pandas"]
    )

    print("\nSkill Gap Analysis:")
    print(gap)

    # ----------------------------------
    # Similar Roles
    # ----------------------------------
    roles = similar_roles("data scientist")

    print("\nSimilar Roles:")
    print(roles)

    # ----------------------------------
    # Learning Path
    # ----------------------------------
    roadmap = generate_learning_path(
        "data scientist",
        ["python", "sql", "pandas"]
    )

    print("\nLearning Roadmap:")
    print(roadmap)

    print("\nProject executed successfully.")


if __name__ == "__main__":
    main()