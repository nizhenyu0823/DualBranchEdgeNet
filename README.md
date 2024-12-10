# DualBranchEdgeNet
Colonoscopy is widely regarded as an important method for the early
 diagnosis of colorectal cancer. Polyp segmentation is crucial for accurately analyzing
 the morphology and size of polyps, thus contributing to the early diagnosis of
 colorectal cancer. Although manual segmentation can achieve accurate results, the
 process is time-consuming and labor-intensive. Automated polyp segmentation using
 deep learning technology can significantly improve segmentation efficiency and has
 high accuracy and robustness. However, the challenges of complex backgrounds,
 variations in size and shape, and blurred boundaries make accurate polyp segmentation
 difficult. We propose a dual-path polyp segmentation network for colonoscopy
 images.
 The network adopts a Transformer and fully convolutional dual-path
 structure. The Transformer branch based on quadtree attention realizes the full
 extraction of global information, while the fully convolutional branch complements
 the local information extracted by the Transformer branch, achieving complementary
 advantages.
 Furthermore, a edge enhancement module (EEM) based on LoG
 features is introduced to enhance the network’s focus on edge information, improving
 segmentation accuracy in complex backgrounds and blurred edge regions. Finally, a
 gated fusion module is used to organically fuse the two-path information. Experimental
 results on the Kvasir-SEG dataset and the CVC-ClinicDB dataset show that the model
 outperforms baseline methods in both qualitative and quantitative indicators.
