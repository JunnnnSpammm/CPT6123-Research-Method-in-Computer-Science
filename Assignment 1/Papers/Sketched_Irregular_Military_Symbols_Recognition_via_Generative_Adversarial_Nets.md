### Sketched Irregular Military Symbols Recognition via Generative Adversarial Nets

---

## Page 1: Introduction and Problem Statement

* **The Challenge:** Traditional military situation mapping relies on human-computer interaction (WIMP interfaces) which is time-consuming and inadequate for modern information warfare.


* **Sketched Symbols:** While sketch-based mapping systems offer a faster alternative, irregular military symbols are difficult to accurately recognize using standard database correlation and computer graphics.


* **The Proposed Solution:** The authors introduce a novel approach using Generative Adversarial Networks (GANs) to convert hand-drawn irregular military symbols into standardized formats.


* **Key Contributions:** The paper proposes a specific architecture named RSM-GAN (ResNet-based encoder, Style-based generator, and Markovianity-based discriminator) and introduces a newly constructed dataset called MMS-Sketch.



---

## Page 2: Related Work and Preliminaries

* **Advantages of GANs:** Unlike traditional Convolutional Neural Networks (CNNs) that often produce blurred averages when dealing with probabilistic one-to-many mappings, GANs excel at generating entirely new, clear data.


* **GAN Variations:** The paper reviews Conditional GANs (cGAN), which use additional information to conditionally generate images, CycleGAN, which translates images between domains without paired training data, and StyleGAN, which uses style modulation for high-level feature control.


* **Generative Models:** The core task is to learn the joint probability distribution $p(X,Y)$ and the conditional probability $p(Y\vert{}X)$ to generate new data from random vectors.


* **Discriminative Models:** These models aim to acquire a decision function $f(x)$ or conditional probability $p(Y\vert{}X)$ to distinguish between genuine data and data produced by the generator.



---

## Page 3: Markovian Processes and RSM-GAN Architecture

* **Markovian Processes:** A stochastic process is Markovian if its future state relies solely on its current state, making it conditionally independent of past historical paths.


* **RSM-GAN Encoder:** The encoding network uses a ResNet model containing 17 convolutional layers and one fully-connected layer to embed input image vectors into a feature space.


* **RSM-GAN Generator:** A Style-based network consisting of a Mapping network (to generate a hidden style variable $w$) and a Synthesis network (which combines $w$ with random noise $z$ to control details).


* **RSM-GAN Discriminator:** Utilizes a Markovian structure that divides the image into fixed-size blocks, independently evaluates the authenticity of each block, and averages the results, making the model agnostic to input image size.



---

## Page 4: Model Training and MMS-Sketch Dataset Creation

* **Optimization:** Training involves a two-player minimax game where the discriminator maximizes its ability to identify real data while the generator minimizes the objective function $V(D,G)$.


* **Loss Functions:** The objective function combines a reconstruction loss ($\mathcal{L}_{L_{1}}(G)$) measuring the distance between pseudo-standard and real standard images, alongside an adversarial loss.


* **Algorithm 1 (Training):** The discriminator parameters are updated by ascending its stochastic gradient using both real and generated inputs, after which the generator is updated by descending its stochastic gradient to maximize the discriminator's score.


* **Dataset Creation:** The MMS-Sketch dataset was built using 226 open-source NATO Navy Common Military Marking images.


* **Data Augmentation:** Images were normalized to 64x64 pixels, converted to grayscale, and rotated by 90°, 180°, and 270° to yield 678 standard images. Corresponding hand-drawn versions were created using a tablet PC and stylus.



---

## Page 5: Experimental Setup and Analysis

* **Hardware and Software:** Experiments were conducted in Python 3.7 on a server with an NVIDIA GeForce GTX 3070 GPU.


* **Hyperparameters:** Key configurations included a batch size of 8, a learning rate of 0.0001, and a maximum of 100,000 steps.


* **Post-Processing:** Although the model converged, generated images initially contained noise.


* **Refinement Techniques:** The authors applied grayscale processing, truncation point filtering for noise reduction, and a Laplace convolution kernel paired with minimum filtering to sharpen the final output images.



---

## Page 6: Comparative Results and Conclusion

* **Evaluation Metric:** The Fréchet Inception Distance (FID) was used to quantify the similarity between the generated images and the evaluation set.


* **Comparative Performance:** RSM-GAN outperformed standard GAN and CycleGAN models, particularly in limited data scenarios (e.g., achieving an FID of 251.10 compared to CycleGAN's 289.14 at a dataset size of 50).


* **Conclusion 1 - Efficiency:** The generative approach is more flexible and faster than traditional database matching, skipping the two-step feature extraction and pattern matching process.


* **Conclusion 2 - Scalability:** The method effectively handles complex battlefield maps by converting symbols before they interfere with one another, allowing for easier secondary identification.


* **Conclusion 3 - Fault Tolerance:** RSM-GAN excels at standardizing extremely irregular hand-drawn images, overcoming the limitations of fuzzy matching and restricted feature extraction.



---