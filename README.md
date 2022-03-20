![Logo](https://github.com/Wp-Zhang/HandyRec/blob/master/logo.png)

![TF Depend](https://img.shields.io/badge/TensorFlow-2.1+-orange)
![License Badge](https://img.shields.io/badge/license-Apache%202-green)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9e1122d78ad345acb7fa5d9c72b64d91)](https://www.codacy.com/gh/Wp-Zhang/HandyRec/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Wp-Zhang/HandyRec&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/9e1122d78ad345acb7fa5d9c72b64d91)](https://www.codacy.com/gh/Wp-Zhang/HandyRec/dashboard?utm_source=github.com&utm_medium=referral&utm_content=Wp-Zhang/HandyRec&utm_campaign=Badge_Coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📝 Description

**HandyRec** is a package of deep-learning recommendation models implemented with **TF2.5** ✨. It is meant to be an **easy-to-use** and **easy-to-read** package for people who want to use or learn deep-learning recommendation models.

It is currently a personal project for learning purposes. I recently started to learn deep-learning recommendation algorithms and design patterns💦. I'll try to implement some classical algorithms along with example notebooks here.

## 💡Models

### Matching

| Model      | Paper                                                                                                                 | Example                                                                                                                                                    |
| :--------- | :-------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| YouTubeDNN | \[RecSys 2016] [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/pdf/10.1145/2959100.2959190) | [![Jupyer](https://img.shields.io/badge/Jupyter%20Notebook-grey?logo=jupyter)](https://github.com/Wp-Zhang/HandyRec/blob/master/examples/YouTubeDNN.ipynb) |

### Ranking

| Model      | Paper                                                                                                                    | Example                                                                                                                                                    |
| :--------- | :----------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| YouTubeDNN | \[RecSys 2016] [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/pdf/10.1145/2959100.2959190)    | [![Jupyer](https://img.shields.io/badge/Jupyter%20Notebook-grey?logo=jupyter)](https://github.com/Wp-Zhang/HandyRec/blob/master/examples/YouTubeDNN.ipynb) |
| DeepFM     | \[IJCAI, 2017] [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf) | [![Jupyer](https://img.shields.io/badge/Jupyter%20Notebook-grey?logo=jupyter)](https://github.com/Wp-Zhang/HandyRec/blob/master/examples/DeepFM.ipynb) |

## ℹ️ Usage

Examples can be found [here](https://github.com/Wp-Zhang/HandyRec/tree/master/examples) and the table above.

This project is under development and has not been packaged yet😣. Please download the source code and import it as a local module.

## 🗺️ TODO List

-   [ ] 🎨 redesign the structure of the whole project
-   [ ] 🛠️ add unit testing
-   [ ] 🚧 package this project (when >10 models are implemented)

## 🛎️ Acknowledgments

Especially thanks to [DeepMatch](https://github.com/shenweichen/DeepMatch) and [DeepCTR](https://github.com/shenweichen/DeepCTR). I got much inspiration about code structure and model implementation from these projects.

The logo of this project is generated by [AdobeLogoMaker](https://www.adobe.com/express/create/logo)
