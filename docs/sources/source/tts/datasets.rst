Datasets
========

.. _ljspeech:

LJSpeech
--------

`LJSpeech <https://keithito.com/LJ-Speech-Dataset/>`__ is a speech dataset that
consists of a single female, English speaker. It contains approximately 24
hours of speech.

Obtaining and prepocessing the data for NeMo can be done with
`our helper script <https://github.com/NVIDIA/NeMo/blob/master/scripts/get_ljspeech_data.py>`_:

.. code-block:: bash

    python scripts/get_ljspeech_data.py --data_root=<where_you_want_to_save_data>
