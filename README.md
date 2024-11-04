Define Problems:

* A big Online Supermarket - 'Economy Hub' (EH) faced an increasing %cancellation in every Double-Day Campaigns. This leaded to increasing cost of operations, delivery & impacted negatively on P&L (Profit & Loss) of the company
* EH after doing some interviews, realized 2 main reasons that could lead to this situation:

(1) Users are Resellers, who leveraged the vouchers in special campaigns to buy mobile phones & resell to other customers

(2) Some stores intentionally increased GMV (revenue) by fake orders to achieve target bonus, then cancelled

* Dataset: we have GMV performance of product to store level in D8 (Double-Day campaign 8.8), 1 to 30.9, D10, D11, 6-11.12 and D12 of Year 2020


* Requirement:

(1) Identify which %Cancellation rate is highly alarmed!

(2) Build ML models to predict stores that likely to have High Cancellation Rate

(3) Evaluate different model performance and translate into insights, what features that detect the most likely high %cancellation rate and recommendations

* Files in Github:
1. E-commerce Data Preprocessing.ipynb: data processing for all D8, D9, D10, D11 and D12.csv files and concat into ecom_df.csv
2. E-commerce Modeling Cancellation Prediction.ipynb: including EDA, Data Transformation and Feature Engineering, Train model, Evaluation and Results interpretation with Recommendation
