### Electronic Health Data in the Context of Patient Length of Stay Prediction Using Generative Adversarial Nets for Synthetic Data Creation

---

### Introduction and Background (Pages 1–2)

* **The Importance of LOS**: A patient's Length of Stay (LOS) is a critical metric for hospital management, helping decision-makers allocate resources, measure efficiency, and reduce costs.


* **The Data Challenge**: Machine Learning (ML) algorithms require large amounts of data, but collecting and labeling Electronic Health Records (EHR) is costly, time-consuming, and strictly restricted by privacy regulations.


* **Proposed Solution**: The study investigates using Generative Adversarial Networks (GANs)—specifically Conditional Tabular GAN (CTGAN) and Copula GAN—to create synthetic tabular EHR data as a proxy for real-world data in multi-class LOS prediction.


* **Generative Models Explored**:
* **Variational Auto Encoder (VAE)**: Reconstructs input data using an encoder and decoder while minimizing the Kullback-Leibler (KL) divergence.


* **GANs**: Consist of a generator (creating fake data) and a discriminator (identifying fake data) competing in a min-max game.


* **CTGAN**: Specifically designed for tabular data, it handles non-Gaussian distributions, mixed data types, and class imbalance via conditional generators.


* **Copula GAN**: Utilizes Gaussian copulas to transform populations into Gaussian-like distributions.





### Related Work and Data Origin (Pages 2–3)

* **Previous Studies**: Past research, such as work by Chin-Cheong et al. and Fernandes, explored generating EHR with GANs but relied on massive datasets (up to 4 million rows), which resulted in extended training times and bypassed the issue of data scarcity.


* **Addressing Data Scarcity**: Unlike previous work, this paper utilizes smaller datasets to explicitly address data scarcity in the healthcare domain while ensuring the data is free of charge and admission requirements.


* **Dataset Specifications**: The researchers utilized the "NY Hospital Inpatient Discharges in 2015" dataset, selecting randomized data from three specific university hospitals: Nassau University Medical Center (NUMC), State University of New York (SUNY), and New York University (NYU).


* **Feature Selection**: To prevent information leakage, the datasets were narrowed down to 13 relevant features known at the time of admission, including high-cardinality diagnosis and procedure codes.



### Methodology and Evaluation Metrics (Pages 4–5)

* **Target Variable Binning**: The numerical LOS variable was divided into three distinct classes: short-stays (2 days or less), medium-stays (between 2 and 7 days), and long-stays (exceeding 7 days).


* **Quality Metrics**: The Synthetic Data Vault was used to evaluate synthetic data quality through multiple metrics:
* **Goodness of Fit**: Kolmogorov-Smirnov tests for continuous samples and Chi-squared tests for discrete features.


* **Likelihood and Detection Metrics**: Used to estimate the likelihood of artificial data fitting the learned distribution and distinguishing fake data from real data.


* **Privacy Preserving Capabilities**: Evaluated by counting the number of duplicate rows overlapping between the synthetic and original datasets.


* **ML Efficacy**: Measured using the macro F1-score across classifiers fitted to predict patient LOS.




* **Implementation**: Original datasets were partitioned into 5,000-row training sets and hold-out test sets, and the Table VAE (TVAE) model was included to benchmark GAN performance against a non-GAN architecture.



### Results and Machine Learning Efficacy (Pages 5–6)

* **Hyperparameter Search**: CTGAN demonstrated superior performance in aggregated quantitative results. TVAE and Copula GAN models were heavily penalized for high similarity to original data, with TVAE failing to replicate core diagnosis and procedure columns adequately in visual evaluations.


* **Classification Performance**: The classifiers (Random Forest, SVM, K-Nearest Neighbor) trained on synthetic data failed to match the performance of models trained on real data across all datasets. Copula GAN exhibited the worst performance.


* **The Duplicate Issue**: While TVAE appeared to perform well initially, it was rejected due to its extremely high number of privacy-violating duplicates.


* **Minority Class Struggles**: The moderate performance of GAN-trained classifiers was primarily traced to poor prediction capabilities for the minority "long-stay" target class, as the models disproportionately favored the larger medium and short-stay classes.



### Discussion and Future Outlook (Pages 7–8)

* **GANs for Over-Sampling**: When testing GANs as an over-sampling technique alongside conventional methods like SMOTE, no synthetic over-sampling technique was able to surpass the performance of classifiers trained purely on real data frames.


* **Training Challenges**: The training process for CTGAN is highly time-consuming, with a 30-hyperparameter search taking over a full week, making Copula GAN a potential alternative when timely creation is required despite performance losses.


* **Handling Imbalance**: Current GAN models struggle significantly with highly imbalanced categorical columns and high-cardinality features, often failing to reproduce minority categories (such as rare patient diagnoses).


* **Conclusion**: While CTGAN is the superior model for creating synthetic EHR for LOS prediction due to its conditional sampling and privacy preservation, state-of-the-art GAN architectures must be improved to handle clinical data imbalances efficiently. Future research should focus on accelerating the training process and testing ML efficacy in LOS regression tasks.



---