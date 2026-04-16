# KINOVA Arm Documentation

This is the documentation guide for the KINOVA Gen3 Ultra Lightweight robot arm. This arm is very expensive, so please read this documentation carefully and completely. If you have any questions, ask the Funrobo teaching team or consult the official KINOVA documentation.

---

## Physical Setup - 7DOF

The 7DOF KINOVA arm should already be installed on a table in the back of the classroom. Before turning it on, check the following:

- The blue ethernet cable is plugged into the back of the robot arm and into your computer
- The power supply brick is plugged into the ESTOP
- The ESTOP is depressed (twist it in the direction of the arrows to depress it)
- The arm is clear of any obstacles that might damage it

Once all of the above are verified, turn on the robot arm by holding the silver power button on the top left of the arm control panel for 3 seconds.

---

## Physical Setup - 6DOF

The 6DOF KINOVA arm should already be installed on a table in the back of the classroom. Before turning it on, check the following:

- The micro-usb cable is plugged into the back of the robot arm and the usb is in your computer
- The power supply brick is plugged into the arm
- The arm is clear of any obstacles that might damage it

Once all of the above are verified, turn on the robot arm by switching the switch to the ON position.

---

## Communication Setup - 7DOF

Before anything else, make sure you can talk to the KINOVA arm. Verify that the ethernet cable is plugged into both the arm and your computer, and that the robot is powered on.

### Windows

**Step 1.** Open `Control Panel > Network and Internet > Network and Sharing Center`.

![Network and Sharing Center](images/image2.png)

**Step 2.** Click **Change adapter settings** on the left sidebar.

![Change adapter settings](images/image6.png)

**Step 3.** Select the wired Ethernet adapter (e.g. `Local Area Connection`) and click **Properties**.

**Step 4.** Select **Internet Protocol Version 4 (TCP/IPv4)** and click **Properties**.

![IPv4 Properties](images/image5.png)

**Step 5.** Select **Use the following IP address** and enter the following:

- **IPv4 address:** `192.168.1.11`
- **Subnet mask:** `255.255.255.0`

**Step 6.** Click **OK**.

![IP Address settings](images/image1.png)

---

### Linux

**Step 1.** Open **Settings** and go to **Network**. Click the gear icon next to the wired connection.

![Network settings](images/image3.png)

**Step 2.** Go to the **IPv4** tab. Set the method to **Manual** and enter the following:

- **Address:** `192.168.1.11`
- **Netmask:** `255.255.255.0`

![IPv4 Manual settings](images/image4.png)

**Step 3.** Click **Apply**.

---

### Mac

TODO, I don't have a Mac :(

---

## Communication Setup - 6DOF

Since we are using USB connection, the arm automatically handles all communication setup.

---

## Computational Setup

### Install UV

This codebase uses UV as its Python environment manager. UV is similar to conda, but faster and easier to use.

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### Codebase Setup

First, fork the `kinova-control-system` repo from GitHub, then clone it:

```bash
git clone https://github.com/<username>/kinova-control-system
```

Set up the Python environment with UV:

```bash
uv sync
```

This will install everything automatically. Once it is done, activate the virtual environment:

```bash
source .venv/bin/activate
```

To verify that everything worked correctly, run:

```bash
python -m backend.kinova
```

> **NOTE:** This will only work if you are inside the UV virtual environment. Your terminal prompt should show `(kinova-control-system)`. If it does not, either activate the environment using the command above or run `uv run python -m backend.kinova` instead.

> **NOTE:** This will only work if you are physically connected to the KINOVA robot. Follow the Physical and Communication Setup sections above first.

The expected output is:

```
Testing Environment...

Environment is ready to go

Have fun using the Kinova Robot Arm!
```

---

## Codebase Documentation

### Overview

This codebase is structured similarly to how Arduino code is written. An abstraction layer handles all of the robot connection code for you.

The backend code lives in the `backend/` folder. **Do not modify anything in this folder.** The abstraction layer includes safety checks that help prevent you from damaging the robot. If you do not know what you are doing in there, you could cause serious and expensive damage.

There is an `example.py` file that shows how to move the arm. Look at it as a reference, but write your own code in `main.py`. The only two functions you should need to change are:

- `start()`
- `loop()`

These behave exactly like their Arduino equivalents:

- `start()` runs exactly once, right after the Kinova arm is initialized.
- `loop()` runs periodically at a fixed loop rate (default: **20Hz**, configurable in the `Main()` class definition).

---

### Public Methods

You can control the arm using these methods on `self.kinova_robot`:

| Method | Description |
|---|---|
| `set_joint_angles(angles, gripper_percentage=None)` | Move the arm to a target joint configuration (radians). Optionally set gripper position (0-100). |
| `get_joint_angles()` | Returns the current joint angles as a list (radians). |
| `set_gripper(percentage)` | Set the gripper position directly (0 = open, 100 = closed). |
| `open_gripper()` | Fully opens the gripper. |
| `close_gripper()` | Fully closes the gripper. |
| `set_torque(enable)` | Enables or disables torque/admittance mode. |

> **NOTE:** All angles in the abstraction layer use **radians**. The physical arm operates in degrees internally. If you ever dig into the backend, do not mix them up.

---

### Action Queue and the `wait` Parameter

Every call to `set_joint_angles()`, `set_gripper()`, `open_gripper()`, and `close_gripper()` puts a command into an internal action queue. Commands execute one at a time, in order. The arm will not start the next command until the current one finishes.

By default, `wait=True` on all of these methods. This means the call blocks until the action is fully complete before your code moves on. This is almost always what you want.

If you pass `wait=False`, the call returns immediately and the command runs in the background. Be careful with this: if you fire off several `wait=False` commands in quick succession inside `loop()`, they will pile up in the queue and the arm will keep moving long after your loop has stopped asking it to.

---

### 6DOF vs 7DOF

There are two versions of the KINOVA arm in the classroom. Check which one you are using before writing your joint angle arrays, since they have different numbers of joints.

- The **large arm** is 7DOF and takes an array of **7 angles**.
- The **small arm** is 6DOF and takes an array of **6 angles**.

Passing the wrong size array will either error or silently move fewer joints than you intended. See `example.py` for reference home positions for both arms.

---

### Suction Gripper

If your arm has the suction cup attachment instead of the two-finger gripper, pass `is_suction=True` when creating your `Main` instance:

```python
final_project = Main(is_suction=True)
```

The gripper methods (`set_gripper`, `open_gripper`, `close_gripper`) still work the same way from your code. The backend handles the difference in hardware automatically.

---

### Simulation Mode

You can run the codebase in simulation using PyBullet without connecting to a real robot. Pass `simulate=True` and a `urdf_path` to `Main()`:

```python
# 6DOF simulation
final_project = Main(simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf")

# 7DOF simulation
final_project = Main(simulate=True, urdf_path="visualizer/7dof/urdf/7dof.urdf")
```

A PyBullet window will open showing the robot. This is a good way to test your joint angle sequences before running them on the real hardware.

> **NOTE:** The simulation runs at 240Hz internally but your `loop()` still runs at whatever `loop_rate` you set. The arm movement speed in simulation is capped at roughly 60 degrees/second to match real-world behavior.

---

### Usage

To use this code, create an instance of `Main` and keep the main thread alive with a `while True` loop:

```python
final_project = Main()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    final_project.shutdown()
```

The `while True` loop keeps the background thread running, which is what actually executes your `loop()` function. Press `CTRL+C` to exit cleanly.

`shutdown()` will automatically put the arm into a safe torque-off state and stop all motion before exiting. Do not just kill the terminal without calling it, as the arm may stay in a powered hold state.

The `Main()` constructor accepts the following optional arguments:

| Argument | Default | Description |
|---|---|---|
| `loop_rate` | `20` | How often `loop()` runs, in Hz. |
| `is_suction` | `False` | Set to `True` if using the suction gripper. |
| `simulate` | `False` | Set to `True` to run in PyBullet simulation. |
| `urdf_path` | `None` | Path to the URDF file (required when simulating). |

---

## Common Issues

**The arm does not move but no error is thrown.**
Commands are queued. If a previous `set_joint_angles()` call with `wait=False` is still running, new commands will wait behind it. If the queue seems stuck, restart the program.

**`ImportError` or `ModuleNotFoundError` on startup.**
You are not in the UV virtual environment. Run `source .venv/bin/activate` and try again.

**The arm starts moving and does not stop.**
Press `CTRL+C` immediately to trigger `shutdown()`. If that does not work, hit the ESTOP. This is why the ESTOP must always be within reach when the arm is running.

**PyBullet window opens but the arm does not appear.**
The `urdf_path` is wrong. Double-check the path relative to the project root.

**`get_joint_angles()` returns an empty list.**
The background connection thread has not finished initializing yet. There is a short startup delay after `Main()` is created. If you call `get_joint_angles()` in `start()`, add a small `time.sleep()` first.