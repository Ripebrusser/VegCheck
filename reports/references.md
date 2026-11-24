# References

Only sources that were actually consulted are listed. Dataset facts (image count,
licence, categories) were verified against the primary sources below rather than
taken from a secondary description.

## Dataset and its publication

1. Suryawanshi, Y., Patil, K., & Chumchu, P. (2022). *VegNet: Dataset of vegetable
   quality images for machine learning applications.* **Data in Brief**, 45, 108657.
   https://doi.org/10.1016/j.dib.2022.108657
   — Open access. States 6,850 images across four vegetables and five quality
   categories, collected at Vishwakarma University, Pune, India.

2. Suryawanshi, Y., Patil, K., & Chumchu, P. (2022). *VegNet: Vegetable Dataset
   with quality (Unripe, Ripe, Old, Dried and Damaged)* [Data set]. Mendeley Data,
   Version 1. https://doi.org/10.17632/6nxnjbn9w6.1
   — Licence **CC BY 4.0**. The record used to confirm licensing and category names.

3. Project-AgML. *VegNet_quality_classification* [Data set]. Hugging Face.
   https://huggingface.co/datasets/Project-AgML/VegNet_quality_classification
   — The 6,150-image, 256×256 redistribution actually used in this project;
   licence field `cc-by-4.0`.

## Image processing

4. Zuiderveld, K. (1994). *Contrast Limited Adaptive Histogram Equalization.*
   In: Heckbert, P.S. (ed.) **Graphics Gems IV**, Academic Press, pp. 474–485.
   — The CLAHE algorithm implemented via `cv2.createCLAHE`.

5. Immerkær, J. (1996). *Fast Noise Variance Estimation.* **Computer Vision and
   Image Understanding**, 64(2), 300–302.
   — Basis of the Laplacian-MAD noise estimate used to compare filters.

6. Hasler, D., & Süsstrunk, S. (2003). *Measuring colourfulness in natural images.*
   **Proc. SPIE 5007, Human Vision and Electronic Imaging VIII**, 87–95.
   — The colourfulness metric used in the colour feature set.

7. Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms.*
   **IEEE Transactions on Systems, Man, and Cybernetics**, 9(1), 62–66.
   — Used for the foreground/background split in segmentation.

## Texture and shape features

8. Haralick, R.M., Shanmugam, K., & Dinstein, I. (1973). *Textural Features for
   Image Classification.* **IEEE Transactions on Systems, Man, and Cybernetics**,
   SMC-3(6), 610–621.
   — GLCM descriptors (contrast, dissimilarity, homogeneity, energy, correlation, ASM).

9. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). *Multiresolution gray-scale
   and rotation invariant texture classification with local binary patterns.*
   **IEEE Transactions on Pattern Analysis and Machine Intelligence**, 24(7),
   971–987.
   — Uniform LBP used at two radii.

## Deep learning (used in Phase 2)

10. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018).
    *MobileNetV2: Inverted Residuals and Linear Bottlenecks.*
    **CVPR 2018**, 4510–4520.

11. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for
    Image Recognition.* **CVPR 2016**, 770–778.

12. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009).
    *ImageNet: A Large-Scale Hierarchical Image Database.* **CVPR 2009**.
    — Origin of the mean/std normalisation and pretrained weights used here.

13. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
    (2017). *Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization.* **ICCV 2017**, 618–626.
    — Candidate method for the explanation panel in Phase 2/3.

## Learning with imbalance

14. He, H., & Garcia, E.A. (2009). *Learning from Imbalanced Data.*
    **IEEE Transactions on Knowledge and Data Engineering**, 21(9), 1263–1284.
    — Justifies macro F1 and class weighting over accuracy.

## Tools

15. Bradski, G. (2000). *The OpenCV Library.* **Dr. Dobb's Journal of Software Tools.**
    — OpenCV 5.0.0 used for all image I/O, colour conversion, filtering, morphology
    and contour analysis.

16. van der Walt, S., et al. (2014). *scikit-image: Image processing in Python.*
    **PeerJ**, 2, e453.
    — GLCM, LBP and SSIM implementations.

17. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.*
    **JMLR**, 12, 2825–2830.
    — Feature screening, cross-validation and Random Forest experiments.

18. Abadi, M., et al. (2016). *TensorFlow: A System for Large-Scale Machine
    Learning.* **OSDI 2016**, 265–283.
    — TensorFlow 2.21.0 (CPU) used for Phase 2 training.
