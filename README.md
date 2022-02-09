# ProjectComputerVision
Project ComputerVision group 7

Overleaf link: https://www.overleaf.com/9968814498nynmpsvpntnb

## Setting up
To get started with developing/using the project, some setup is required

It is recommended to first set up an isolated virtual Python environment for the project. This can be done with eg. Python's built in [venv](https://docs.python.org/3/library/venv.html)

```bash
# Navigate to the project directory
$ cd <project_dir>
# Create the 
$ python -m venv .env
# Now most IDE's will automatically recognize the environment but if it doesnt or you use the cli, just use the following to activate it on the current terminal prompt

# Windows
.\.env\Scripts\activate
# Linux/git Bash/maybe Mac?
source ./bin/activate
# If succesful, the terminal should now look like this in Windows
# (.env) PS C:\
```

After setting up the virtual environment, install the dependency packages with `pip`
```
pip install -r requirements.txt
```
## Running Program
In order to run the demo, the keypoints database needs to be generated using following command:
```
py .\src\main.py generate_database -d <PAINTINGS_DATABASE_PATH>
```

Once the databse has been generated, the demo can be started using following command:
```
py .\src\main.py video -f <VIDEO_FILE_PATH> [-c <CALIBRATION_FILE_PATH>]
```
(Precalibrated GoPro parameters available in "resources/calibration_matrix.yaml")
