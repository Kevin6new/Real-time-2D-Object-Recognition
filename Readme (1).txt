Real-time 2-D Object Recognition 

This project implements a real-time object recognition system that utilizes background subtraction, morphological operations, and contour analysis to extract and classify objects from a video stream. The system identifies objects based on Hu Moments and explores different distance metrics for classification, providing insights into the effectiveness of various classifiers and distance measures in distinguishing objects.

Development Environment
Operating System: Windows 11
IDE for C++: Visual Studio 2022

Running the Code

Ensure that OpenCV is linked properly by going to Project Properties.
Under Linker -> Additional Dependencies, add `dirent.h`, `csv_util.h` in the files.
Build the solution by selecting Build -> Build Solution.
Run the program by hitting Local Windows Debugger.

To run the different distance functions please change the fucntion name in the program.

Links:

Training Data: https://drive.google.com/drive/folders/1z3rvuaK2W3vZI6BOFsvNITWIXP3xXRBP?usp=sharing

Classification Data: https://drive.google.com/drive/folders/1Y-7ptw9LWrxnUj__ts8mZ-X5I-HbuiRi?usp=sharing

Time Travel Days Used : 0

Authors
- Kevin Sani
- Basil Reji
