Define Problems:

* A big Online E-commerce Marketplace - 'E-com Hub' (EH) faced an increasing %cancellation in every Double-Day Campaigns. This leaded to increasing cost of operations, delivery & impacted negatively on P&L (Profit & Loss) of the company
* EH after doing some interviews, realized 2 main reasons that could lead to this situation:

(1) Users are Resellers, who leveraged the vouchers in special campaigns to buy mobile phones & resell to other customers

(2) Some stores intentionally increased GMV (revenue) by fake orders to achieve target bonus, then cancelled

* Dataset: we have GMV performance of store level in D8 (Double-Day campaign 8.8), 1 to 30.9, D10, D11, D12 and 6-11.12 of Year 2020

---

* Requirement:

Part 1: Prediction
(1) Describe EH business performances and characteristics. How are GMV Marketplace (the listing channel), GMV Affiliate (GMV from KOL/KOC Affiliate channel) and GMV Live (GMV from Livestream channel)

(2) Identify which %Cancellation rate is highly alarmed!

(3) Build ML models to predict stores that likely to have High Cancellation Rate

(4) Evaluate different model performance. Main Objective: Identify as "MANY POTENTIAL HIGH cancellation store" as possible (High Recall). Translate into insights, what features that detect the most likely high %cancellation rate

Part 2: Cluster Analysis
(5) Please Clustering our data, draw-out some insights and recommendations for each Cluster

* Files in Github:
1. E-commerce Data Preprocessing.ipynb: data processing for all D8, D9, D10, D11 and D12.csv files and concat into ecom_df.csv
2. E-commerce Modeling Cancellation Prediction.ipynb: including EDA, Data Transformation and Feature Engineering, Train model, Evaluation and Results interpretation with Recommendation

---

* SUMMARY ANALYSIS
1. EDA:


GMV TREND

![image](https://github.com/user-attachments/assets/bcf86d78-ddf4-4052-a325-caf02d720215)

![image](https://github.com/user-attachments/assets/1d86a78f-2e58-4f10-87e7-b833f2de5d56)

Overall, from here we see that very strange movement:
- Fashion & Electronics contributes the most GMV in EH
- We start very low in 8.8, 2,7M and and x2 in 9.9 up to 4.8M USD - 50% contributed from Electronics!!!
- Then we see 10.10, we decreased again -2x to 2.5M USD
- 11.11, 6.12 and 12.12 we see a big push of D-Day, respectively 18.4M, 15.6M & 22.9M - Awesome!!!
- 7.12 is just a normal day after campaigns, but still 3M!

From the research with the team, 8.8 was not the peak sales of E-commerce yet, and 10.10 we faced a major system collapse! Thats why we see so low GMV from D8 and D10

GMV BY CHANNEL

![image](https://github.com/user-attachments/assets/f75a9907-3a60-455b-a838-527d4e12e2a3)

By Channel, we have some insights below:
- Affiliate contribute 35% GMV, while Marketplace contribute 24%. So in this business, we were quite rely heavily on Affiliate (have to pay much to KOL/KOC for affiliate revenue - which contribute 1/3 of our GMV)
- Marketplace channel is still weak. This one is the way that we grow GMV by ourselves but only contribute 1/4 GMV
- As Livestreaming is increasing recently, with Instagram Live, TikTok Shop, etc. Of course, GMV_Live contribute 47% of total GMV. Surprising to me is: Electronics 20% GMV come from livestream, even higher than Fashion (12%) - could be a breakthrough for our business

GMV CANCELLATION RATE - After 1 Day & After 30 Days

![image](https://github.com/user-attachments/assets/df037386-26cc-4015-ab37-b2caab995eae)

Taking Avg. seems not so trustful, because seems Cancellation rate in Campaigns is lower than normal days:
- Normal GMV_Cancel_1D_pct could be 23%, GMV_Cancel_30D_pct is 31%
- In Campaigns, GMV_Cancel_1D_pct is 20%, GMV_Cancel_30D_pct is 27%

Trend of Cancellation in Campaigns Day:
- The trend of GMV_Cancel_1D_pct in Campaigns days is relatively 20%, but increase in D11 & D12, reaching 22% in D12
- D10 cancellation after 30D is suddenly low at 23% - for the technical issues happenning so it is better than usual. I believe that it should be 26-27% in campaigns. It is very high - 32% in 7.12 (what happenned?)

--> First assumptions: GMV_Cancel_30D_pct: lets take **30% as Warning Threshold**

---
2. Answer Questions: How much %Cancellation is considered as High:

![image](https://github.com/user-attachments/assets/de7ecc8d-bfc6-4cb3-ac14-6d9ff1cd514f)

![image](https://github.com/user-attachments/assets/bc8b7fa4-6815-4c36-b269-9afe6d0c86cc)

- After 1 Day, 65% stores have cancellation ratebelow 30%. There always special group with cancellation rate 80%+
- After 30 Days, the cancellation rate increased, Notably 20%% stores have 80%+. Around 47% stores have cancellation rate from 30-80%

---> To make an aggressive control, I propose we set **Stores with Cancellation rate 30%+** after 30 Days should be Warning!

---
3. Data Transformation
4. Baseline Model
5. Upsampling
6. Model
I tried Random Forest, LightGBM, CatBoost, XGBoost with different tuning techniques, from choosing appropriate Precision/Recall Threshold, SMOTE, RandomSearchCV

Pls see in the E-commerce Modeling Cancellation Prediction.ipynb file

---
7. Model Conclusion & Intepretation

![image](https://github.com/user-attachments/assets/c6a4a946-799e-4ef1-9d6c-2976780906e9)

![image](https://github.com/user-attachments/assets/8507834c-cea8-42ee-9e2a-8c25e2a687a9)

- Seems all the SMOTE or Tuning don't help much to improve our f1-score for Label_1 (High_cancellation_30% = Yes). Because the data is already highly imbalanced, not ez to change
- However, in the problem set, we want to prioritize Recall more than Precision - Detect Highly_Cancelled_30% Stores as MANY as possible.
- With XGB and using Precision/Recall Threshold = 0.195, we have better recall_Label_1 = 71% (while others only 47%), but still keep f1-score = 55% and Accuracy = 73% (Acceptable rate)

Feature Importance: some important features that impact to High Cancellation: first_cat, Log_GMV, log_avg_item_price and second_cat

![image](https://github.com/user-attachments/assets/15b51d07-ecca-4372-b876-4019f0c9e280)

SHAP Explanation: 

![image](https://github.com/user-attachments/assets/a9f747f7-9df9-44af-8375-93ea71dbe9ac)

The summary plot shows the feature importance of each feature in the model. 

- The results show that %GMV_Cancel_1D,” “Industry”, 'first_cat' and 'Log_GMV' play major roles in determining the results.
- The more red dots in the positive SHAP value, the higher %cancellation rate in our prediction
- For eg: lets see Log_GMV, we see that most red dots in negative side of SHAP value, so Higher GMV --> Less probability of %Cancellation
- Interesting observation: higher avg_item_price, more likely to have high Cancellation rate? (cause more red dots in the positive side of SHAP value)

---
8. Clustering

I realize that I can define into 4 Clusters using k_means clustering:

![image](https://github.com/user-attachments/assets/5af81bcc-1c6a-4584-9f06-9c89edbadd68)

---
9. Cluster Analysis & Final Recommendation

![image](https://github.com/user-attachments/assets/f4154a4c-aab8-4f06-884b-41e328851392)

---
Cluster 0 - **Electronics Growth**

Characteristics:
- High overall GMV with the largest share in **Phones & Electronics**.
- Moderate cancel rate after 30 days (26.2%), with **Jewelry & Accessories** as the category with the highest cancellation.
- Over one-third of stores have account management support.

Recommendations:
- Increase targeted promotions for **Jewelry & Accessories** to build customer trust in this category.
- Educate stores on setting realistic expectations for electronic accessories' availability and delivery times.
- Enhance **post-sale support** to reduce the need for cancellations.

---
Cluster 1 - **Emerging Fashion**

Characteristics:
- Primarily a fashion cluster with **Beauty & Personal Care** as the highest GMV category.
- Moderate cancel rate after 30 days (14.3%) and relatively lower cancel rates compared to other clusters.
- Low GMV per store indicates smaller, potentially newer or boutique sellers.

Recommendations:
- Improve inventory and supply chain support for fashion sellers to further decrease cancel rates.
- Develop an **onboarding program** tailored to help emerging stores optimize their listings and inventory levels.
- Offer **promotional support** for Beauty & Personal Care and **Collectibles** (highest cancel categories) to improve buyer confidence.

---
Cluster 2 - **High-Cancel Electronics**

Characteristics:
- Highest cancellation rate across all clusters (75.9%), especially in **Phones & Electronics**.
- Very low percentage of stores with account management support (19.7%).
- Low GMV per store suggests these may be smaller or less-established sellers.

Recommendations:
- Implement a **high-touch account management program** for this cluster to improve seller practices and reduce cancellations.
- Focus on **inventory management training** for sellers in Phones & Electronics.
- Consider stricter entry requirements or **screening processes** for electronics sellers to ensure higher quality and reliability.

---
Cluster 3 - **Niche Fashion**

Characteristics:
- Smallest GMV and lowest cancel rate (3.8%) of all clusters, focused in **Beauty & Personal Care**.
- Relatively low GMV per store, indicating niche or boutique offerings.
- **Collectibles** category shows the highest cancel rate but is less impactful overall.

Recommendations:
- Provide **niche marketing support** to grow GMV for specialty stores.
- Encourage stores in **Collectibles** to better manage customer expectations regarding rare or unique items.
- Promote customer reviews and ratings to build credibility for niche stores and maintain low cancellation rates.
   
