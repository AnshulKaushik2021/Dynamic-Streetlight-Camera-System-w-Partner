# FinalProject

# Project Description and Background

Current streetlight systems are often based on a light sensor located on the top of the light to dictate whether the lamp should be turned on or off. This has little flaws in most scenarios, such as rural areas and highways. 

However, in cities and areas with tall buildings providing shade, problems start to arise with this approach. Thus, we wanted to implement a dynamic streetlight system that is dependent on a camera that supervises the area of the streetlight and that can use NIR imagery to detect a level of shadow and adjust the light accordingly.


Current streetlight systems have scenarios in big cities in which the streetlight sensor is telling it to stay off but there remain areas in heavy shadow that result in increased crime and cameras being less able to detect and make out specifics.

We want to utilize NIR imagery via a camera in order to remove shadow from images and dynamically adjust a nearby streetlight for improved safety and reduction of blind spots in cities.


# List of Files:

Final_Main.py - Main file used for running the dual camera system, creating propagation box, and sending brightness commands to light.

IoU_Code.py - Code to test the test shadow maps using IoU metric.

Shadow_Detect_Final.py - Supporting code used for detection of Shadow_Detect_Final

Soda_can.py - File used to find a soda can with both cameras and printing the difference of the center of said can to show the pixel difference between each camera

alignment.py - This is used in image combination and alignment

light.py - Initial code used for testing smart light
