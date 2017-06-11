import pandas


class DataTransformer:
    CATEGORICAL_VARIABLES = {"Country": ["EE", "ES", "FI", "SK"],
                             "CreditScoreEeMini": [0.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
                             "CreditScoreEsEquifaxRisk": ["A", "AA", "AAA", "B", "C", "D"],
                             "CreditScoreEsMicroL": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"],
                             "CreditScoreFiAsiakasTietoRiskGrade": ["RL0", "RL1", "RL2", "RL3", "RL4", "RL5"],
                             "Education": [1.0, 2.0, 3.0, 4.0, 5.0],
                             "EmploymentDurationCurrentEmployer": ["MoreThan5Years", "TrialPeriod", "UpTo1Year",
                                                                   "UpTo2Years", "UpTo3Years", "UpTo4Years",
                                                                   "UpTo5Years"],
                             "EmploymentStatus": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                             "Gender": [0.0, 1.0, 2.0],
                             "HomeOwnershipType": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                             "LanguageCode": [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 15, 21, 22],
                             "MaritalStatus": [1.0, 2.0, 3.0, 4.0, 5.0],
                             "MonthlyPaymentDay": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                   20, 21, 22, 23, 24, 25, 26, 27, 28],
                             "NewCreditCustomer": [False, True],
                             "OccupationArea": [-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                                13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
                             "Rating": ["A", "AA", "B", "C", "D", "E", "F", "HR"],
                             "UseOfLoan": [0, 1, 2, 3, 4, 5, 6, 7, 8, 101, 102, 104, 106, 107, 108, 110],
                             "VerificationType": [1.0, 2.0, 3.0, 4.0],
                             "NrOfDependants": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10Plus"],
                             "WorkExperience": ["10To15Years", "15To25Years", "2To5Years", "5To10Years",
                                                "LessThan2Years", "MoreThan25Years"]}

    NUMERIC_VARIABLES = ["Age", "AppliedAmount", "DebtToIncome", "ExpectedLoss", "LiabilitiesTotal", "FreeCash",
                         "IncomeFromChildSupport", "IncomeFromFamilyAllowance", "IncomeFromLeavePay",
                         "IncomeFromPension", "IncomeFromPrincipalEmployer", "IncomeFromSocialWelfare", "IncomeOther",
                         "IncomeTotal", "Interest", "LoanDuration", "LossGivenDefault", "MonthlyPayment",
                         "ProbabilityOfDefault"]

    PREDICTOR_VARIABLES = sorted(list(CATEGORICAL_VARIABLES.keys())) + NUMERIC_VARIABLES

    @classmethod
    def assign_categories(cls, column):
        return column.astype("category", categories=cls.CATEGORICAL_VARIABLES[column.name])

    @classmethod
    def transform(cls, data):
        data = data[cls.PREDICTOR_VARIABLES]
        data[cls.NUMERIC_VARIABLES] = data[cls.NUMERIC_VARIABLES].astype("float64")
        ordered_categorical_keys = sorted(list(cls.CATEGORICAL_VARIABLES.keys()))
        data[ordered_categorical_keys] = data[ordered_categorical_keys].apply(cls.assign_categories)
        return pandas.get_dummies(data)
