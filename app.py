from flask import Flask, render_template, request
import joblib
import pandas as pd 

app = Flask(__name__)

model = joblib.load("customer_retention_model.pkl")


def get_recommendations(tenure, monthly_charges,
                        contract, internet_service,
                        tech_support):

    recommendations = []
    reasons = []

    if contract == "Month-to-month":
        reasons.append("Customer is on a month-to-month contract.")
        recommendations.append(
            "Offer a longer-term contract with a suitable loyalty benefit."
        )

    if tenure < 12:
        reasons.append("Customer has a short tenure.")
        recommendations.append(
            "Provide an early-stage loyalty offer to encourage continued usage."
        )

    if monthly_charges > 80:
        reasons.append("Customer has relatively high monthly charges.")
        recommendations.append(
            "Consider a personalized pricing or service-value offer."
        )

    if tech_support == "No":
        reasons.append("Customer does not have technical support.")
        recommendations.append(
            "Offer technical support assistance or a suitable support plan."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Maintain regular engagement and provide loyalty benefits."
        )

    return reasons, recommendations


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        tenure = int(request.form["tenure"])
        monthly_charges = float(request.form["monthly_charges"])
        total_charges = float(request.form["total_charges"])

        contract = request.form["contract"]
        internet_service = request.form["internet_service"]
        tech_support = request.form["tech_support"]
        payment_method = request.form["payment_method"]

        customer = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service,
    "TechSupport": tech_support,
    "PaymentMethod": payment_method
}])

        probability = model.predict_proba(customer)[0][1]

        risk_percentage = round(probability * 100, 1)

        if risk_percentage >= 70:
            risk = "HIGH"
        elif risk_percentage >= 40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        reasons, recommendations = get_recommendations(
            tenure,
            monthly_charges,
            contract,
            internet_service,
            tech_support
        )

        result = {
            "risk": risk,
            "risk_percentage": risk_percentage,
            "reasons": reasons,
            "recommendations": recommendations
        }

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)