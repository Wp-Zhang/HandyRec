---
title: Quick Start
permalink: docs/en/quick-start
key: docs-quick-start
---

HandyRec is a package of deep-learning recommendation models implemented with **TF2.5** ✨. It is meant to be an **easy-to-use** and **easy-to-read** package for people who want to use or learn deep-learning recommendation models.

<!-- In this document, you will learn how to **install the theme**, **setup your site**, **local preview** for development, **build** and **publish**. -->

This project is under development and has not been packaged yet😣. Please download the source code and import it as a local module as shown below:

Download source code:
```bash
git clone https://github.com/Wp-Zhang/HandyRec.git
```

Then add this line in the front of your code:
```python
import sys
sys.append('/YOUR/PATH/HandeyRec/')
```

Now you can import HandyRec as a local module!
```python
from handyrec.models import DeepFM
```

**Follow up:**

The Jupyter Notebook [here](https://github.com/Wp-Zhang/HandyRec/blob/master/examples/YouTubeDNN.ipynb) tested the model proposed in [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/pdf/10.1145/2959100.2959190) on the common dataset **MovieLens1M**. The notebook can be directly opened in Google Colab, please try it out 😃!

All other examples can be found [here](https://github.com/Wp-Zhang/HandyRec/tree/master/examples).