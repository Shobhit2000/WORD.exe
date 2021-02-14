# WORD.exe

## Inspiration
One of the major problems that the military face in a compact hostage situation or terrorists attack situation is that it is very difficult to overcome the situation considering the civilians and that too taking care that there is no casualty.

## What it does
We have come up with an idea that can eliminate this problem by using the remote access autonomous aim and fire robots that can differentiate between the terrorists and civilians and disarm the terrorists making sure that there is no one who is injured and the civilians are rescued without any harm and at the same time capture the terrorists from whom a large amount of information can be gathered. This can be done by using a controlled robot that analyzes the environment and disarms the terrorists by actually aiming at their guns by using the auto-aim focus gun mechanism that automatically aims at the terrorist's guns and firing at it thereby creating a safe spot for the military to take control thereon without harming anyone and also having control over the situation.

## How I built it
The major attraction of our project is the remote access controlled robot with an auto-aim feature based on a deep learning mechanism. Firstly we have built the CAD model for our robot which is designed to run on any kind of terrain and at the same time incorporating the gun which is an automatic aim assist robot. The auto-aim feature has been implemented based on the deep learning concept which is designed to detect guns in its view using Convolutional Neural Networks which takes in the live video feed using the RPi camera and returns the bounding box coordinates from which we can find the region of interest which tells the gun to focus at that particular point that is at the weapon thereby disarming the terrorist so that the military or police can take control thereon. The gun is mounted on the top of the robot through servo motors which are controlled incorporating the feedback loop using PID and Stanley controller. Irrespective of the movement of the robot and auto-aim gun can fire in any direction when it detects terrorists with guns. 

## Challenges I ran into
* Completing the hardware part of the project as all the hardware components were with only one member.
* Integrating the Deep Learning model in Raspberry pi taking into consideration the computation power of the microprocessor.
* Completing and visualizing the intended conceptual CAD design in 24 hrs which was itself quite challenging

## Accomplishments that I'm proud of
* Training and deploying a Tensor Flow lite model on a raspberry pi
* Object tracking using servos and camera
* Designing a shooting mechanism 
* End to end integration of software and hardware
* The mechanisms, visualization, and the implementation of the Ground Robot CAD model which can run on any terrain. 

## What I learned
* Control of servos using RPi
* Implementing TensorFlow lite to improve performance
* Basics of object tracking
* Using RPi purely on the command line
* Cooperating with teammates remotely

## What's next for Word.exe
With this working prototype ready the next step would be
* Manufacturing the bot
* Increasing the accuracy of the model
* Adding live video streaming to remote channels
* Autonomous running
