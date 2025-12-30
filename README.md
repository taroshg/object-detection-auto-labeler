### Object detection labeling software (auto-labeling)

I started this project to quickly label chess pieces using object detection models (auto-labeling).

Currently the `label.py` script provides a FastAPI server that serves images from the `data/` directory and displays the object detection results from the Faster R-CNN model.

I plan to extend this to allow manual correction of the object detection labels and save the corrected labels for training purposes, ultimately building a dataset for model fine-tuning.