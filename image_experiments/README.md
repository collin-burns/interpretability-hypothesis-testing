This is the code for replicating the image experiments. 

Requirements
---
- Python 3 with standard packages (e.g., numpy).
- PyTorch 1.0.
- [Generative Inpainting and its requirements.](https://github.com/JiahuiYu/generative_inpainting)

Usage
---
1. Run select_bounding_boxes.py to choose the bounding boxes for the images of interest (or use those selected for the paper in the bounding_boxes directory).
2. Run generate_imgs.py to generate a new image using the inpainter for each selected bounding box (and its corresponding image).
3. Run hypothesis_test.py to determine which patches to reject.
4. Run visualize_results.py to visualize the rejected bounding boxes.

Finally, to generate the LIME result for Figure 1, run lime_experiment.py.