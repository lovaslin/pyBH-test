# pyBH-test

This repository rasemble all the codes used to perform the test in the arcicle "pyBumpHunter : A model independent bump hunting tool for High Energy Physics" published ...

The results were obtained, using pyBumpHunter v0.4.0.

[Link](https://github.com/scikit-hep/pyBumpHunter) to pyBumpHunter repository  
Link to the paper [not yet]


## Instruction to run the tests

* Create a new python environement
```bash
python3 -m venv .env
source .env/bin/activate
```

* Install the required packages
```bash
pip install -r requirement.txt
```

* Run the test you want (example for signal injection)
```bash
cd Signal-injection
python injection_test.py
```

**Note** : There is a special argument for the multi-channel test in order to choose the number of channel.
The default value (used in the paper) is 2.

To try other values, you can use the following syntax (example for 3 channels) :
```bash
cd Multi-channel_combination
python multi_test.py --Nchan 3
```
