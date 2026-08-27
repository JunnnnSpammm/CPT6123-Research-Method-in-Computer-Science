### The Comparison Between Conditional Generative Adversarial Nets and Deep Convolutional Generative Adversarial Network and its GUI-related application

---

### Introduction to Generative Models

* The paper examines Generative Adversarial Nets (GAN) and two variants: Conditional Generative Adversarial Nets (CGAN) and Deep Convolutional Generative Adversarial Networks (DCGAN).


* While GANs are useful in fields like healthcare and automotive arts, they struggle with discrete data, training instability, and pattern collapse.


* CGAN addresses controllability by adding conditional constraints, whereas DCGAN improves feature extraction by replacing multi-layer perceptrons with convolutional neural networks.



### Architectural Differences

* A standard GAN relies on a zero-sum game between a generator creating fake images and a discriminator attempting to classify them as real or fake.


* The core objective function of a standard GAN is represented as $\min_{G}\max_{D}V(D,G)=\mathbb{E}_{x\sim P_{data}(x)}[\log D(x)] + \mathbb{E}_{z\sim P_{z}(z)}[\log(1-D(G(z)))]$.


* CGANs alter this by feeding conditional information, such as class labels or one-hot vectors, into both the generator and the discriminator to specify the generated output.


* DCGANs implement structural improvements by using convolution and deconvolution instead of pooling layers, applying batch normalization, removing fully connected layers, and utilizing Tanh and LeakyReLU activation functions.



### Experimental Results on Datasets

* The models were evaluated using the MNIST dataset and an Animation dataset comprised of 10 million scraped images of anime faces.


* During MNIST training, CGAN utilized linear layers, but its loss curves showed significant fluctuations and divergence in the first 25,000 iterations.


* DCGAN utilized Conv2d layers for spatial downsampling and demonstrated much smoother loss curves during training on both datasets.


* Ultimately, DCGAN proved to be highly efficient, successfully generating realistic animation images in just 200 training iterations.



### GUI Integration and Conclusion

* To make the system accessible, the authors developed a Graphical User Interface (GUI) using Python's Tkinter module.


* The interface utilizes an Entry widget for capturing user details like name and birthday, and a Listbox widget for selecting face photos with different tags.


* The study concludes that due to its stable loss curves and high-quality outputs, DCGAN outperforms CGAN in simple image generation tasks.