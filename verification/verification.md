Verification results for HP cycle were obtained from an edited version of Martin T. White's PocketTHERM code
The original version is readily available through 
- https://pockettherm.github.io/
- https://github.com/ElsevierSoftwareX/SOFTX-D-24-00232/tree/main
- https://doi.org/10.1016/j.softx.2024.101806
And a local copy can be found through D:\nexus\02_learning\00_university_education\04_MSc_TUDelft\05_thesis_nexus\02_resources\03_everything_software\03_pocketTHERM

The edited version can be found in the same folder. 

The software has been edited to return pinch point data of a cycle necessary as input for my HP calculation tool, the PR ratio used is quite precise but was later found to be arbitrary. 

For generation of the necessary validation data (= performance data and pp data for your cycle input), run scripts_and_examples/Python_examples/HP_CoolProp_example.py. You can see some print statements. The cycle performance gives you the cycle performance parameters in the order displayed, and the PP give you the pp in the order displayed. Uncomment pp_i for the pinch points corresponding the the arbitrary pressure ratio.


