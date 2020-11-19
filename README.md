# Controller for Self-Driving License-Plate-Reader Robot
#### For UBC ENPH353 2020T1 Competition

## Run

In order to generate the world you will also need this repository:
`https://github.com/ENPH353/2020T1_competition`

After launching the competition simulation, run in a terminal:
`roslaunch controller control.launch`

- To run autonomous steering only:      `roslaunch controller steer.launch`
- To launch license plate reader only:  `roslaunch controller read.launch`

`roslaunch controller sim.launch` will generate the world and launch this package, but cannot be controlled by command line arguments (unlike the competition launcher).






