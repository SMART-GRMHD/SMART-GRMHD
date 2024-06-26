# Physics Informed Neural Networks (PINNs) for Black Hole Simulations

This repository contains the code for a research project aiming to enhance our understanding of the accretion disks surrounding black holes observed in the SgrA* and M87 galaxies through the EHT. We leverage Physics Informed Neural Networks (PINNs) to streamline expensive GRMHD simulations and generate novel predictions.

## Table of Contents

- [Background](#background)
- [Setup](#setup)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Background

Recent observations of black holes in the center of the SgrA* and M87 galaxies by the Event Horizon Telescope (EHT) have provided incredible insights into these cosmic phenomena. However, refining predictions for the future and understanding the intricacies of the accretion disks around these black holes demands high computational costs using the General Relativistic Magnetohydrodynamics (GRMHD) framework.

This project introduces an alternative approach, using machine learning and more specifically, Physics Informed Neural Networks (PINNs). This novel methodology seeks to reduce computational costs while generating robust and accurate predictions.

## Setup 

We recommend creating a virual environment when running this project. You can create a virtual environment in the root directory of this project by execuitng:
```
python3 -m venv .
```
which you can then activate with:

```
source bin/activate
```

You can always deactivate the `venv` by executing the `deactivate` command in the terminal.

This  package uses `Pytorch`. You can install all dependencies by running:

```
pip install -r requirements.txt
```
## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request.

## Acknowledgements

We want to thank the researchers, engineers, and the community who have made this project possible. Special mention to the Event Horizon Telescope team for their ground-breaking observations that have provided a significant motivation for this work.
