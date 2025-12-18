from pydantic import BaseModel, Field, validator
from typing import Optional, Literal


class PredictionRequest(BaseModel):
    """
    Schema for a single credit application based on common 
    Credit Risk Dataset features and RFM proxies.
    """

    # --- Demographic Features ---
    person_age: int = Field(..., gt=18, lt=100,
                            description="Age of the applicant (18-100)")
    person_income: float = Field(..., gt=0, description="Annual Income in USD")
    person_home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"] = Field(
        ..., description="Home ownership status"
    )
    person_emp_length: float = Field(..., ge=0,
                                     description="Years of employment")

    # --- Loan Details ---
    loan_intent: Literal["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"] = Field(
        ..., description="The purpose of the loan"
    )
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(
        ..., description="Internal credit grade assigned to the loan"
    )
    loan_amnt: float = Field(..., gt=0, description="Loan amount requested")
    loan_int_rate: float = Field(..., gt=0,
                                 description="Interest rate on the loan")
    loan_percent_income: float = Field(
        ..., ge=0, le=1.0, description="Loan amount as a fraction of total income (0.0 to 1.0)"
    )

    # --- Credit History ---
    cb_person_default_on_file: Literal["Y", "N"] = Field(
        ..., description="Whether the person has defaulted before"
    )
    cb_person_cred_hist_length: int = Field(..., ge=0,
                                            description="Credit history length in years")

    # --- RFM / Proxy Features (Calculated from Task 4) ---
    frequency: int = Field(0, ge=0, description="Number of past transactions")
    recency: int = Field(0, ge=0, description="Days since last transaction")
    monetary: float = Field(0.0, ge=0, description="Total transaction volume")

    # --- Custom Validator Example ---
    @validator("person_age")
    def age_must_be_reasonable(cls, v, values):
        # Business logic: Employment length cannot exceed age
        if "person_emp_length" in values and v < values["person_emp_length"] + 14:
            raise ValueError(
                "Age must be at least 14 years greater than employment length")
        return v

    class Config:
        """Example payload for FastAPI /docs"""
        json_schema_extra = {
            "example": {
                "person_age": 30,
                "person_income": 65000.0,
                "person_home_ownership": "MORTGAGE",
                "person_emp_length": 10.0,
                "loan_intent": "VENTURE",
                "loan_grade": "A",
                "loan_amnt": 15000.0,
                "loan_int_rate": 7.5,
                "loan_percent_income": 0.23,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 8,
                "frequency": 12,
                "recency": 5,
                "monetary": 4500.0
            }
        }


class PredictionResponse(BaseModel):
    """
    Standardized output for the API including score and decision.
    """
    risk_probability: float = Field(...,
                                    description="The probability of default (0.0 to 1.0)")
    risk_score: int = Field(...,
                            description="Credit score (e.g., 300 to 850 scale)")
    prediction: int = Field(..., description="0 for Approved, 1 for Denied")
    status: str = Field(..., description="Human-readable decision label")
