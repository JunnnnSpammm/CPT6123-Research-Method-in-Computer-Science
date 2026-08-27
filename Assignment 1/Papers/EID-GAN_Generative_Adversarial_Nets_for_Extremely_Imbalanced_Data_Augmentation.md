### EID-GAN Generative Adversarial Nets for Extremely Imbalanced Data Augmentation

---

The paper **"EID-GAN: Generative Adversarial Nets for Extremely Imbalanced Data Augmentation"** (IEEE Transactions on Industrial Informatics, Vol. 19, No. 3, March 2023) by Wei Li et al. addresses the challenge of augmenting industrial image datasets containing extremely small outlier defects.

---

**Page 3208 (Page 1)**

* **Problem Definition**: Industrial defect detection suffers from extremely imbalanced data where non-defective (positive) samples dominate, and defective (negative) samples contain tiny outliers covering roughly $0.05\%$ of the image area (e.g., Fabric and Piston datasets). Existing augmentation models fail because background features dominate generator training.
* **Proposed Solution**: Introduces EID-GAN, featuring a cropped norm penalty function to enforce learning of tiny outlier details, and a 2-detector evaluation framework using $k$-fold cross-validation.

**Page 3209 (Page 2)**

* **Dataset Characteristics**: Visualizes defect ratios across datasets: Fabric ($\sim 1/2000$), Piston ($\sim 1/2000$), and DAGM ($\sim 1/23$).
* **Core Mechanism**: During training, a region matching the size of a target outlier is cropped from the generated image $G_{crop}(z)$ and subtracted from a real outlier $ano_i$ to calculate a penalty, which updates the generator via backpropagation alongside standard GAN loss.
* **Evaluation Motivation**: Traditional evaluation fails because positive and negative image distributions overlap heavily. The authors propose evaluating synthetic samples by measuring whether adding them to a training set improves downstream defect detector performance (Error-Escape, False-Alarm, F1-Score).
* **Related Work (Interpolation)**: Discusses SMOTE and Prototype Generation.

**Page 3210 (Page 3)**

* **Related Work Limitations**: SMOTE creates boundary misclassifications and duplicate samples. Prototype generation improperly clusters positive and negative samples together.
* **GAN Baselines**: Evaluates vanilla GAN, DCGAN, AnoGAN, and BAGAN. These focus on global distribution transformation rather than localized pixel details, causing generated samples to lack tiny outliers entirely. BAGAN's autoencoder structure also introduces blurriness.

**Page 3211 (Page 4)**

* **Architectural Details**: Displays EID-GAN architecture (Fig. 3) and pixel-level masks (Fig. 4).
* **Formulations**: Defines the penalty function $f_{outlier} = \lambda \times \vert{}ano_i - G_{crop}(z)\vert{}$. Contrast with pix2pix (which uses an auxiliary full-image $L1$ loss with U-Net) and standard $L1/L2$ norms (which penalize the whole image $\vert{}x - G(z)\vert{}$ and erase tiny features).
* **Theoretical Framework**: Establishes the minimax objective function incorporating $f_{outlier}(x)$.

**Page 3212 (Page 5)**

* **Mathematical Theorems & Proofs**:
* **Theorem 1**: Discriminator optimal point remains $D^*(x) = \frac{p_r(x)}{p_r(x) + p_G(x)}$.
* **Theorem 2**: When distributions $p_r$ and $p_G$ do not overlap, $C(G) \approx f_{outlier} > 0$, providing non-zero gradients that prevent training stall.
* **Theorem 3**: Global minimum of virtual training criterion $C(G)$ lies in $[\delta - 2\log 2, d - 2\log 2]$ when $p_r(x) \approx p_G(x)$.


* **Training & Evaluation Workflow**: Provides pseudocode (Algorithm 1) and details the step-by-step 2-detector evaluation methodology.

**Page 3213 (Page 6)**

* **Evaluation & Sampling**: Details the evaluation pipeline (Fig. 5) using Faster R-CNN as the outlier detector. Demonstrates image sampling at local nadir points of the $f_{outlier}$ curve across epochs (Fig. 6).
* **Experimental Setup**: Experiments on Fabric (100 samples, $1024\times1024$), Piston (70 samples, $480\times480$), and DAGM Category 1 (150 samples, $512\times512$) against baselines (SMOTE, DCGAN, BAGAN, SN-GAN, WGAN-GP). Sets $\lambda = 0.01$.

**Page 3214 (Page 7)**

* **Qualitative & Hyperparameter Analysis**: Visual comparisons (Fig. 7) show EID-GAN is the only model that synthesizes distinct outliers. Standard full-image $L1/L2$ norms produce blurry results without defects (Fig. 8). Hyperparameter sweep confirms $\lambda = 0.01$ is optimal (Fig. 9).
* **Metric Inadequacies**: Shows CNN classification accuracy (0.55), K-Means clustering (Fig. 10a), and FID scores fail to accurately reflect generated outlier quality.

**Page 3215 (Page 8)**

* **Quantitative Benchmark Performance**:

| Dataset | Method / Setup | Error-Escape | False-Alarm | F1-Score |
| --- | --- | --- | --- | --- |
| **Fabric** (Table I) | Without Synthetic | 8.33% | 6.25% | 94.0% |
|  | EID-GAN Augmented | **1.70%** | **0%** | **98.9%** |
| **Piston** (Table II) | Without Synthetic | 5.00% | 2.00% | 94.3% |
|  | EID-GAN Augmented | **1.67%** | **0.83%** | **98.0%** |
| **DAGM** (Table III) | Without Synthetic | 0.95% | 0.30% | 97.2% |
|  | EID-GAN Augmented | **0.40%** | **0%** | **99.7%** |

**Page 3216 (Page 9)**

* **Alternative Augmentation Baselines**: Visualizes and analyzes traditional techniques (Fig. 14): cropping destroys outliers; flipping creates exact copy-pastes; Gaussian noise creates irrelevant artifacts; Mixup dilutes defects; VAE yields blurry outputs.
* **Discussion**: EID-GAN synthesizes outliers effectively even when using positive samples to generate negative samples. Compares application context to scene generation models like CSG and ContraGAN.

**Page 3217 (Page 10)**

* **Conclusion**: Confirms EID-GAN generates high-quality outlier instances for both extremely imbalanced and general imbalanced data, improving industrial defect detection.
* **Documentation & Biographies**: Contains references 1–30 and academic biographies for authors Wei Li and Jinlin Chen.

**Page 3218 (Page 11)**

* **Biographies (Cont.)**: Academic background and research profiles for co-authors Jiannong Cao, Chao Ma, Jia Wang, Xiaohui Cui, and Ping Chen.