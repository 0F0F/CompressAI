CompressAI
==========

:mod:`CompressAI` (*compress-ay*) is a machine learning library for end-to-end
data compression research.

CompressAI is built on top of PyTorch and provides:

* custom operations, layers and models for deep learning based data compression

* a partial port of the official `TensorFlow compression
  <https://github.com/tensorflow/compression>`_ library

* pre-trained end-to-end compression models for learned image compression

* evaluation scripts to compare learned models against classical image/video
  compression codecs


.. toctree::
   :hidden:

   self


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial_quickstart
   tutorial_train


.. toctree::
   :maxdepth: 1
   :caption: Library API

   compressai
   datasets
   entropy_models
   ans
   layers
   models
   ops

.. toctree::
   :maxdepth: 2
   :caption: Model Zoo

   zoo

.. toctree::
  :maxdepth: 2
  :caption: Utils

  cli_usage


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
