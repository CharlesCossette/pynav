
Getting Started
---------------

Welcome to the navlie tutorial! The following few pages will go through a toy localization problem, where we will be running state estimation algorithms using navlie's framework. The first step is to install this package. This should be done by directly cloning the git repo and performing a local pip install:

.. code-block:: bash

   git clone git@github.com:decargroup/navlie.git 
   cd navlie && pip install -e .

All the dependencies should get installed by this command and the package should now be ready to use. Use the column on the left to go to next page of the tutorial.


.. note::

   Although this package is currently registed with PyPi, installation via ``pip install navlie`` will not work. We're still figuring out how to set this up properly. Sorry! Feel free to help.


.. toctree::
   :hidden:

   1. Getting Started <self>
   2. Toy Problem - Traditional <./tutorial/traditional.ipynb>
   3. Toy Problem - Lie groups <./tutorial/lie_groups.ipynb>
   4. Specifying Jacobians <./tutorial/jacobians.ipynb>
   4. Composite States <./tutorial/composite.ipynb>
   5. Batch Estimation <./tutorial/batch.ipynb>