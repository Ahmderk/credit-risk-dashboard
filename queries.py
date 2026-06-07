import sqlite3
import pandas as pd

conn = sqlite3.connect("credit_risk.db")

##Overall default rate

q1 = """
    SELECT
        COUNT(*)    AS total_loans,
        SUM(is_default)     AS total_defaults,
        ROUND(AVG(is_default)*100, 2)  AS default_rate_pct
    FROM loans;
"""
print("OVERALL DEFAULT RATE")
print(pd.read_sql(q1,conn).to_string(index=False))

##DEFAULT RATE BY LOAN GRADE
q2 = """
    SELECT
        grade,
        COUNT(*)        AS total_loans,
        ROUND(AVG(is_default)*100, 2)   AS default_rate_pct,
        ROUND(AVG(int_rate), 2)     AS avg_interest_rate
    FROM loans
    GROUP BY grade
    ORDER BY grade;
"""

print("DEFAULT RATE BY LOAN GRADE")
print(pd.read_sql(q2, conn).to_string(index=False))


##DEFAULT RATE BY LOAN PURPOSE
q3 ="""
    SELECT
        purpose,
        COUNT(*)    AS total_loans,
        ROUND(AVG(is_default)*100, 2)   AS defaulte_rate_pct
    FROM loans
    GROUP BY purpose
    ORDER BY ROUND(AVG(is_default)*100, 2) DESC
    LIMIT 10;
"""
print("RISKIEST LOAN BY PURPOSE")
print(pd.read_sql(q3, conn).to_string(index=False))

##AVERAGE LOAN AMOUNT BY HOME OWNERSHIP
q4 = """
    SELECT
        home_ownership,
        COUNT(*)    AS total_loans,
        ROUND(AVG(loan_amnt), 0) AS avg_loan_amount,
        ROUND(AVG(is_default)*100, 2) AS default_rate_pct

    FROM loans
    GROUP BY home_ownership
    ORDER BY total_loans desc;
"""

print("LOANS BY HOME OWNER")
print(pd.read_sql(q4, conn).to_string(index=False))

