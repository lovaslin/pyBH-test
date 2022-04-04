# pyBH-test

This repository rasemble all the codes used to perform the test in the arcicle "pyBumpHunter : A model independent bump hunting tool for High Energy Physics" published ...

The results were obtained, using pyBumpHunter v0.4.0.

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
cd signal-injection
python injection_test.py
```

