# Swimmer-Segmentation-and-Pose-Estimation-with-SAM-and-YOLOv8
README - Swimmer Segmentation and Pose Estimation Pipeline
==============================================================

This project implements a modular computer vision pipeline to segment underwater swimmers and classify 
swim strokes based on pose estimation. It integrates Meta's Segment Anything Model (SAM) for precise 
segmentation and YOLOv8 for multi-joint pose detection. After extracting 17 anatomical keypoints per 
swimmer frame, a convolutional neural network (CNN) classifies the stroke type with high accuracy. 
The system addresses challenges typical in underwater video such as noise, occlusion, and motion blur.

The pipeline is structured into stages: environment setup, swimmer segmentation, pose estimation, 
JSON/filename adjustments, CSV dataset generation, and CNN-based accuracy evaluation.


--------------------------------------------------------------
I. Activate the Environment
--------------------------------------------------------------
- Platform: Windows Command Prompt
- Command:
    267_Project\sam2\env1\Scripts\activate.bat


--------------------------------------------------------------
II. Segment Anything Method (SAM)
--------------------------------------------------------------

1. Execute `step1_SAM.py`:
- Modify the following lines to reflect the stroke type:

    Line 24: video_folder = r"C:\267_Project\Term_project\Dataset\original_image\{StrokeType}-training-original"
    Line 103: output_folder = r"C:\267_Project\Term_project\Segmented_SAM_Images\{StrokeType}-training-sam"

- INPUT: 
    C:\267_Project\Term_project\Dataset\original_image\{StrokeType}-training-original

- OUTPUT: 
    C:\267_Project\Term_project\Segmented_SAM_Images\{StrokeType}-training-sam

- Replace {StrokeType} with: back / fly / breast / freestyle
- Run this step four times (one per stroke type)

--------------------------------------------------------------

2. Execute `step2_joint_estimation.py`

a) Using SAM Images for Joint Estimation:
- Modify:
    Line 8:  INPUT_FOLDER  = r"C:\267_Project\Term_project\Segmented_SAM_Images\{StrokeType}-training-sam"
    Line 9:  OUTPUT_FOLDER = r"C:\267_Project\Term_project\Joint_Estimation_SAM_Images\{StrokeType}-training-pose-out"

b) Using Original Images for Joint Estimation:
- Modify:
    Line 8:  INPUT_FOLDER  = r"C:\267_Project\Term_project\Dataset\original_image\{StrokeType}-training-original"
    Line 9:  OUTPUT_FOLDER = r"C:\267_Project\Term_project\Joint_Estimation_SAM_Images\{StrokeType}-training-pose-out"

- Replace {StrokeType} = back / fly / breast / freestyle
- Run four times per mode (SAM + Original)


--------------------------------------------------------------
III. Rename Images and Update JSON
--------------------------------------------------------------

3. Execute `step3_rename.py`:
- Modify:
    Line 5:  IMAGE_FOLDER  = r"C:\267_Project\Term_project\Joint_Estimation_SAM_Images\{StrokeType}-training-pose-out"
    Line 6:  OLD_JSON_PATH = r"C:\267_Project\Term_project\Joint_Estimation_SAM_Images\{StrokeType}-training-pose-out\joint_coordinates.json"
    Line 7:  NEW_JSON_PATH = r"C:\267_Project\Term_project\{StrokeType}_joint_coordinates_update.json"
    Line 8:  RENAME_PREFIX = "{StrokeType}_"

- Run four times for each stroke type


4. Execute `step4_output_cvs_file.py`:
- Modify:
    Line 6: INPUT_FOLDER = r"C:\267_Project\Term_project"
    Line 7: OUTPUT_CSV   = r"C:\267_Project\Term_project\all_strokes_joint_data.csv"


--------------------------------------------------------------
IV. Test Classification Accuracy
--------------------------------------------------------------

5. Execute `step5_accuracy_test.py`:
- Modify:
    Line 19: data = pd.read_csv(r"C:\267_Project\Term_project\all_strokes_joint_data.csv")

- This step trains and evaluates the CNN-based stroke classifier.
