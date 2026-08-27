### Generative Adversarial Nets for Cost-Sensitive Face Recognition

---

### Introduction and Problem Statement

Traditional face recognition technologies frequently rely on standard frontal face databases, but real-world scenarios typically capture profile images at various angles. Furthermore, conventional systems try to find an accurate classifier by minimizing the error rate, which implicitly and incorrectly assumes that all misclassification costs are equal. To address these issues, the authors propose combining a two-channel Generative Adversarial Network (GAN) for image synthesis with cost-sensitive learning for the recognition phase.

### The Two-Channel GAN Approach

To generate photorealistic, identity-preserving frontal images from profile inputs, the researchers developed a two-channel generator framework.

* The global channel extracts generalized features to focus on the overall contour of the face.


* The local channel specifically handles detailed facial regions, namely the left eye, right eye, nose, and mouth.


* The system is trained using pairs of corresponding frontal and profile images, minimizing the objective function $\tilde{\sigma}=\frac{1}{N}argmin_{\sigma_{G}}\sum_{n=1}^{N}\{L(G_{\sigma_{G}}(I_{n}^{P}),I_{n}^{F})\}$.


* The comprehensive loss function combines pixel-wise loss, symmetry loss, adversarial loss, perceptual loss, and total variation regularization to ensure high-resolution outputs.



### Cost-Sensitive Learning Mechanism

Instead of treating all errors identically, the system classifies misclassification costs into three distinct categories: false acceptance ($\lambda_{PN}$), false rejection ($\lambda_{NP}$), and false identification ($\lambda_{PP}$).

* These values construct a cost matrix where correct recognition costs are negligible (zero).


* The expected loss function is designed to minimize the expected cost $E_{x,y}(\lambda_{y\zeta(x)})$ rather than strictly prioritizing raw accuracy.


* A k-nearest neighbor-based probability estimation method, mckNN, is utilized to calculate posterior probabilities for optimal prediction.



### Experimental Results and Conclusion

The proposed method was tested using head-up images with normal expressions from the CMU MultiPIE dataset.

* Synthesized frontal images achieved a recognition accuracy of 93.23% at a $\pm15^{\circ}$ angle, outperforming the original profile images which scored 84.23%.


* The cost-sensitive mckNN classifier reduced the total operational cost to 515, compared to 672 for the traditional cost-blind kNN classifier.