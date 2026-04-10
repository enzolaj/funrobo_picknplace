# KINOVA Arm Documentation

This is the documentation guide for the KINOVA Gen3 Ultra Lightweight robot arm. This arm is very expensive, so please read this documentation carefully and completely. If there are any questions ask the Funrobo teaching team or consult the offical KINOVA documentation.

Physical Setup
==============

The KINOVA arm should be installed on a table for you already in the back of the classroom. As a general overview, the physical setup includes screwing the base onto the robot, and clamping the robot arm to the table.

For physical setup, you do need to:

*   Ensure that the blue ethernet cable is plugged into the back of the robot arm and into your computer

*   Ensure that the power supply brick is plugged into the ESTOP
*   Ensure that the ESTOP is depressed (twist it in the directions of the arrows to depress it)
*   The arm is clear of any obstacles that might damage it

After you have checked all of the above, you can turn on the robot arm by holding the silver power button on the top left of the arm control panel for 3 seconds

Communication Setup
===================

The first step of setting up your computer will be making sure you can talk to the KINOVA arm. Verify that the ethernet cable is plugged into the arm and your computer, and that the robot is on.

Windows Instructions
--------------------

1.  On your computer, open Control Panel > Network and Internet > Network and Sharing Center
2.  Select Change adapter settings![](images/image2.png)![](images/image6.png)
3.  Select wired Ethernet adapter (i.e. Local Area Connection) and choose Properties.  
    
4.  Select Internet Protocol Version 4 (TCP/IPv4) and choose Properties.  
    ![](images/image5.png)
5.  Select Use the following IP address and enter IPv4 address and the Subnet mask. IPv4 address is 192.168.1.11. Subnet mask is 255.255.255.0
6.  Press OK![](images/image1.png)

Linux Instructions
------------------

1.  Go to “Network” Settings & Press the “Settings Icon”![](images/image3.png)

2.  Go to IPv4 Settings. Set the Method to “Manual”, Address to 192.168.1.11, and Netmask to 255.255.255.0

![](images/image4.png)

3.  Press Apply

Mac Instructions
----------------

TODO, I dont have a Mac :(

Computational Setup
===================

Install UV
----------

This codebase uses UV as its python environment manager. UV is similar to conda, but way faster and a lot easier to use. You should probably look into UV to manage all of your python projects since its quickly becoming the standard.

### Windows Instructions

Run the following command to install UV:  

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

### Linux Instructions

Run the following command to install UV:

curl -LsSf https://astral.sh/uv/install.sh | sh

Codebase
--------

First fork the kinova-control-system repo from github (link). Then clone it onto your local filesystem:

git clone [https://github.com/](https://www.google.com/url?q=https://github.com/&sa=D&source=editors&ust=1775837108154458&usg=AOvVaw2JCby1-6AkyDpJLjoLfDWv)<username>/kinova-control-system

Now you have to get the python environment setup. Since we are using UV, it is as simple as running:

uv sync

This will take a second, but once it is done, everything is all setup for you. You can enter the virtual environment normally by running:

source .venv/bin/activate

To verify that everything worked correctly, run:

python -m backend.kinova.py

NOTE: This will ONLY work if you are in the UV virtual environment, if you are not in the virtual environment (it does not say (kinova-control-system) in your terminal) either activate it by following the above instructions or run uv run python -m backend.kinova.py)

The expected output should be:

Testing Environment...

Environment is ready to go

Have fun using the Kinova Robot Arm!

NOTE: This will ONLY work if you are actually connected to the KINOVA robot. Follow the Physical and Communication Setup instructions to connect to the robot.

Codebase Documentation
======================

Overview
--------

This codebase is set up to be similar to how ARDUINO CODE is written. There is an abstraction layer made for you that handles all of the robot connection code.

The backend code is stored in the backend/ folder. I recommend that you DO NOT TOUCH ANY CODE IN THIS FOLDER. The Kinova library is pretty intuitive, but if you don’t know what you are doing you could seriously damage the very expensive robot. There are failsafes and safety checks in the abstraction layer that hopefully stop you from doing any irreparable damage.

There is an [example.py](https://www.google.com/url?q=http://example.py&sa=D&source=editors&ust=1775837108158859&usg=AOvVaw2zeiwRFWNvwONfzVGi9uP6) file that shows how to get the arm to move. You should look at this file, but make any new code in the [main.py](https://www.google.com/url?q=http://main.py&sa=D&source=editors&ust=1775837108159041&usg=AOvVaw0k7kZG4sL5XsqwaR0F--YX) file to be consistent with Python standards. The only 2 functions you should have to change in the [main.py](https://www.google.com/url?q=http://main.py&sa=D&source=editors&ust=1775837108159211&usg=AOvVaw0CujmWJOc-pQ1pst9YDEX4) file are:

*   start()
*   loop()

These behave exactly like the Arduino start and loop functions:

*   The start() function will run exactly once, right after the Kinova arm is initialized.
*   The loop() function will run periodically, at a specific loop\_rate.
*   By default the loop\_rate is set to be 20Hz, which you can change in the Main() class definition.

You can use the Kinova arm with these public methods:

*   set\_joint\_angles()
*   get\_joints\_angles()
*   stop()

NOTE: Angles in the abstraction layer all work in radians, but the actual arm operates in degrees. This means if you mess around with the backend, be very careful since you can very easily forget that the actual Kinova library works in degrees and break it.

Usage
-----

To actually use this code, you will have to make an instance of the Main class:

[var] = Main()

Then you MUST have a while True loop. This loop doesn’t have to do anything, but it keeps the background thread running which actually runs your code:

try:

        while True:

                pass

except KeyboardInterrupt:

        [var].shutdown()

Not only will this keep the main thread alive, but it will also allow you to exit any program running by pressing CTRL+C